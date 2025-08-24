#!/usr/bin/env python3
"""
Storage Consistency Checker for Avi
Checks consistency between file system, metadata, and vector database
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.append('/Users/will/AVI BUILD/Able3_Main_WithVoiceMode/backend')

from services.storage_service import StorageService
from services.vector_service import VectorService

def check_storage_consistency():
    """Check consistency between all storage systems."""
    
    print("🔍 Avi Storage Consistency Check")
    print("=" * 50)
    
    # Initialize services
    storage_service = StorageService()
    vector_service = VectorService()
    
    # 1. Check actual files in sources directory
    sources_dir = Path("/Users/will/AVI BUILD/Able3_Main_WithVoiceMode/data/sources")
    actual_files = list(sources_dir.glob("*"))
    actual_files = [f for f in actual_files if f.is_file()]
    
    print(f"📁 Files in sources directory: {len(actual_files)}")
    for file in actual_files:
        print(f"   - {file.name}")
    
    # 2. Check metadata.json
    metadata = storage_service.load_metadata()
    print(f"\n📋 Documents in metadata.json: {len(metadata)}")
    
    metadata_files = {}
    for doc_id, doc in metadata.items():
        file_path = Path(doc.file_path)
        exists = file_path.exists()
        metadata_files[doc_id] = {
            'name': doc.name,
            'file_path': str(file_path),
            'exists': exists,
            'created_at': doc.created_at
        }
        status = "✅" if exists else "❌"
        print(f"   {status} {doc.name} -> {file_path.name}")
    
    # 3. Check vector database
    vector_stats = vector_service.get_stats()
    print(f"\n🔍 Vector database stats:")
    print(f"   - Total chunks: {vector_stats.get('total_chunks', 0)}")
    print(f"   - Unique documents: {vector_stats.get('unique_documents', 0)}")
    
    # Get all vector database documents
    try:
        all_vector_results = vector_service.collection.get(include=["metadatas"])
        vector_doc_ids = set()
        vector_doc_names = {}
        
        if all_vector_results['metadatas']:
            for metadata in all_vector_results['metadatas']:
                doc_id = metadata.get('document_id')
                doc_name = metadata.get('document_name')
                if doc_id:
                    vector_doc_ids.add(doc_id)
                    vector_doc_names[doc_id] = doc_name
        
        print(f"\n📊 Documents in vector database: {len(vector_doc_ids)}")
        for doc_id in vector_doc_ids:
            print(f"   - {vector_doc_names.get(doc_id, 'Unknown')} ({doc_id[:8]}...)")
    
    except Exception as e:
        print(f"   ❌ Error accessing vector database: {e}")
        vector_doc_ids = set()
    
    # 4. Consistency Analysis
    print(f"\n🔍 CONSISTENCY ANALYSIS")
    print("=" * 30)
    
    # Files vs Metadata
    actual_file_names = {f.name for f in actual_files}
    metadata_file_names = set()
    
    for doc_id, info in metadata_files.items():
        file_path = Path(info['file_path'])
        metadata_file_names.add(file_path.name)
    
    # Find inconsistencies
    orphaned_files = actual_file_names - metadata_file_names
    missing_files = metadata_file_names - actual_file_names
    
    print(f"📁 Files without metadata: {len(orphaned_files)}")
    for file in orphaned_files:
        print(f"   ❌ {file}")
    
    print(f"📋 Metadata without files: {len(missing_files)}")
    for file in missing_files:
        print(f"   ❌ {file}")
    
    # Metadata vs Vector DB
    metadata_doc_ids = set(metadata.keys())
    
    print(f"🔍 Metadata docs not in vector DB: {len(metadata_doc_ids - vector_doc_ids)}")
    for doc_id in metadata_doc_ids - vector_doc_ids:
        doc_info = metadata_files.get(doc_id, {})
        doc_name = doc_info.get('name', 'Unknown')
        print(f"   ❌ {doc_name} ({doc_id[:8]}...)")
    
    print(f"🔍 Vector DB docs not in metadata: {len(vector_doc_ids - metadata_doc_ids)}")
    for doc_id in vector_doc_ids - metadata_doc_ids:
        doc_name = vector_doc_names.get(doc_id, 'Unknown')
        print(f"   ❌ {doc_name} ({doc_id[:8]}...)")
    
    # Check for duplicate entries in metadata
    print(f"\n🔍 Checking for duplicate entries...")
    name_counts = {}
    for doc_id, info in metadata_files.items():
        name = info['name']
        if name in name_counts:
            name_counts[name].append(doc_id)
        else:
            name_counts[name] = [doc_id]
    
    duplicates = {name: ids for name, ids in name_counts.items() if len(ids) > 1}
    if duplicates:
        print(f"📋 Found {len(duplicates)} duplicate document names:")
        for name, ids in duplicates.items():
            print(f"   ❌ '{name}' appears {len(ids)} times:")
            for doc_id in ids:
                info = metadata_files[doc_id]
                status = "exists" if info['exists'] else "missing"
                print(f"      - {doc_id[:8]}... ({status})")
    else:
        print(f"✅ No duplicate document names found")
    
    # 5. Summary and Recommendations
    print(f"\n📊 SUMMARY")
    print("=" * 20)
    print(f"Actual files: {len(actual_files)}")
    print(f"Metadata entries: {len(metadata)}")
    print(f"Vector DB documents: {len(vector_doc_ids)}")
    
    total_issues = len(orphaned_files) + len(missing_files) + len(metadata_doc_ids - vector_doc_ids) + len(vector_doc_ids - metadata_doc_ids)
    
    if total_issues == 0:
        print("✅ All storage systems are consistent!")
    else:
        print(f"❌ Found {total_issues} consistency issues")
        
        print(f"\n🔧 RECOMMENDED FIXES:")
        if orphaned_files:
            print("1. Remove orphaned files or add them to metadata")
        if missing_files:
            print("2. Remove metadata entries for missing files")
        if metadata_doc_ids - vector_doc_ids:
            print("3. Remove metadata entries not in vector DB")
        if vector_doc_ids - metadata_doc_ids:
            print("4. Clean up orphaned vector DB entries")
    
    return {
        'actual_files': len(actual_files),
        'metadata_entries': len(metadata),
        'vector_db_docs': len(vector_doc_ids),
        'total_issues': total_issues,
        'orphaned_files': list(orphaned_files),
        'missing_files': list(missing_files),
        'metadata_without_vector': list(metadata_doc_ids - vector_doc_ids),
        'vector_without_metadata': list(vector_doc_ids - metadata_doc_ids),
        'duplicates': duplicates if 'duplicates' in locals() else {}
    }

if __name__ == "__main__":
    results = check_storage_consistency()