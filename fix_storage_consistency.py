#!/usr/bin/env python3
"""
Storage Consistency Fixer for Avi
Fixes inconsistencies between file system, metadata, and vector database
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.append('/Users/will/AVI BUILD/Able3_Main_WithVoiceMode/backend')

from services.storage_service import StorageService
from services.vector_service import VectorService

def fix_storage_consistency():
    """Fix all storage inconsistencies."""
    
    print("🔧 Avi Storage Consistency Fixer")
    print("=" * 50)
    
    # Initialize services
    storage_service = StorageService()
    vector_service = VectorService()
    
    # Get current state
    sources_dir = Path("/Users/will/AVI BUILD/Able3_Main_WithVoiceMode/data/sources")
    actual_files = [f for f in sources_dir.glob("*") if f.is_file() and f.name != ".DS_Store"]
    metadata = storage_service.load_metadata()
    
    print(f"📊 Current state:")
    print(f"   - Actual files: {len(actual_files)}")
    print(f"   - Metadata entries: {len(metadata)}")
    
    # Get vector database state
    try:
        all_vector_results = vector_service.collection.get(include=["metadatas"])
        vector_doc_ids = set()
        if all_vector_results['metadatas']:
            for metadata_entry in all_vector_results['metadatas']:
                doc_id = metadata_entry.get('document_id')
                if doc_id:
                    vector_doc_ids.add(doc_id)
        print(f"   - Vector DB documents: {len(vector_doc_ids)}")
    except Exception as e:
        print(f"   ❌ Error accessing vector database: {e}")
        return False
    
    # Step 1: Clean up metadata - remove entries for missing files
    print(f"\\n🧹 Step 1: Cleaning up metadata...")
    
    valid_metadata = {}
    actual_file_names = {f.name for f in actual_files}
    
    for doc_id, doc in metadata.items():
        file_path = Path(doc.file_path)
        if file_path.name in actual_file_names:
            valid_metadata[doc_id] = doc
            print(f"   ✅ Keeping: {doc.name}")
        else:
            print(f"   🗑️  Removing: {doc.name} (file missing)")
    
    # Handle duplicates - keep the most recent one for each file
    print(f"\\n🔍 Step 2: Handling duplicates...")
    
    file_to_docs = {}
    for doc_id, doc in valid_metadata.items():
        file_path = Path(doc.file_path)
        file_name = file_path.name
        
        if file_name not in file_to_docs:
            file_to_docs[file_name] = []
        file_to_docs[file_name].append((doc_id, doc))
    
    final_metadata = {}
    for file_name, docs in file_to_docs.items():
        if len(docs) == 1:
            doc_id, doc = docs[0]
            final_metadata[doc_id] = doc
            print(f"   ✅ {file_name}: Single entry kept")
        else:
            # Keep the most recent one
            docs.sort(key=lambda x: x[1].created_at, reverse=True)
            doc_id, doc = docs[0]
            final_metadata[doc_id] = doc
            print(f"   🔄 {file_name}: Kept most recent of {len(docs)} duplicates")
            
            # Remove others from vector DB
            for old_doc_id, old_doc in docs[1:]:
                print(f"      🗑️  Removing old duplicate: {old_doc_id[:8]}...")
                vector_service.delete_document(old_doc_id)
    
    # Step 3: Save cleaned metadata
    print(f"\\n💾 Step 3: Saving cleaned metadata...")
    
    # Create backup first
    backup_file = Path("/Users/will/AVI BUILD/Able3_Main_WithVoiceMode/data/metadata_backup.json")
    with open(backup_file, 'w') as f:
        json.dump({k: v.dict() for k, v in metadata.items()}, f, indent=2, default=str)
    print(f"   📋 Backup saved to: {backup_file}")
    
    # Save cleaned metadata
    metadata_file = Path("/Users/will/AVI BUILD/Able3_Main_WithVoiceMode/data/metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump({k: v.dict() for k, v in final_metadata.items()}, f, indent=2, default=str)
    print(f"   ✅ Cleaned metadata saved ({len(final_metadata)} entries)")
    
    # Step 4: Clean up vector database
    print(f"\\n🧹 Step 4: Cleaning up vector database...")
    
    valid_doc_ids = list(final_metadata.keys())
    cleanup_result = vector_service.cleanup_orphaned_documents(valid_doc_ids)
    
    if 'error' in cleanup_result:
        print(f"   ❌ Error cleaning vector DB: {cleanup_result['error']}")
    else:
        print(f"   ✅ Vector DB cleanup completed:")
        print(f"      - Checked: {cleanup_result.get('total_checked', 0)} chunks")
        print(f"      - Orphaned found: {cleanup_result.get('orphaned_found', 0)}")
        print(f"      - Deleted: {cleanup_result.get('deleted', 0)}")
    
    # Step 5: Final verification
    print(f"\\n✅ Step 5: Final verification...")
    
    # Reload and verify
    new_metadata = storage_service.load_metadata()
    new_vector_stats = vector_service.get_stats()
    
    print(f"   📊 Final state:")
    print(f"      - Actual files: {len(actual_files)}")
    print(f"      - Metadata entries: {len(new_metadata)}")
    print(f"      - Vector DB documents: {new_vector_stats.get('unique_documents', 0)}")
    print(f"      - Vector DB chunks: {new_vector_stats.get('total_chunks', 0)}")
    
    # Check if everything matches now
    if len(actual_files) == len(new_metadata) == new_vector_stats.get('unique_documents', 0):
        print(f"\\n🎉 SUCCESS: All storage systems are now consistent!")
        return True
    else:
        print(f"\\n⚠️  WARNING: Some inconsistencies may remain. Run the checker again.")
        return False

if __name__ == "__main__":
    success = fix_storage_consistency()
    if success:
        print(f"\\n🔧 Storage consistency fix completed successfully!")
    else:
        print(f"\\n❌ Storage consistency fix encountered issues.")