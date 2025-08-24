#!/usr/bin/env python3
"""
Migrate metadata from bloated format to optimized format.
This script converts metadata.json from storing full chunk content to storing only references.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

def main():
    print("🔄 Migrating Metadata to Optimized Format")
    print("=" * 50)
    
    # Paths
    metadata_path = Path("data/metadata.json")
    backup_path = Path("data/metadata_bloated_backup_20250824_104935.json")
    
    if not metadata_path.exists():
        print("❌ metadata.json not found!")
        return False
    
    # Load current metadata
    print(f"📖 Loading {metadata_path}")
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    print(f"📊 Found {len(metadata)} documents")
    
    # Calculate current size
    original_size = metadata_path.stat().st_size
    print(f"📏 Original size: {original_size:,} bytes ({original_size/1024:.1f}KB)")
    
    # Convert to optimized format
    optimized_metadata = {}
    total_chunks = 0
    
    for doc_id, doc_data in metadata.items():
        print(f"🔄 Processing: {doc_data.get('name', 'Unknown')}")
        
        # Handle legacy format
        if 'chunks' in doc_data:
            chunks = doc_data['chunks']
            chunk_count = len(chunks)
            chunk_ids = [chunk['id'] for chunk in chunks]
            total_chunks += chunk_count
            
            # Create optimized entry
            optimized_metadata[doc_id] = {
                "id": doc_data["id"],
                "name": doc_data["name"],
                "file_path": doc_data["file_path"], 
                "summary": doc_data["summary"],
                "created_at": doc_data["created_at"],
                "file_size": doc_data["file_size"],
                "chunk_count": chunk_count,
                "chunk_ids": chunk_ids,
                "source_type": doc_data.get("source_type", "pdf"),
                "original_url": doc_data.get("original_url")
            }
            
            print(f"  ✅ Converted {chunk_count} chunks to references")
        else:
            # Already optimized format
            optimized_metadata[doc_id] = doc_data
            chunk_count = doc_data.get("chunk_count", 0)
            total_chunks += chunk_count
            print(f"  ✅ Already optimized ({chunk_count} chunks)")
    
    # Save optimized metadata
    print(f"\n💾 Saving optimized metadata...")
    with open(metadata_path, 'w') as f:
        json.dump(optimized_metadata, f, indent=2, default=str)
    
    # Calculate new size
    new_size = metadata_path.stat().st_size
    savings = original_size - new_size
    savings_pct = (savings / original_size) * 100
    
    print(f"\n🎉 Migration Complete!")
    print(f"📊 Documents: {len(optimized_metadata)}")
    print(f"📊 Total chunks: {total_chunks}")
    print(f"📏 New size: {new_size:,} bytes ({new_size/1024:.1f}KB)")
    print(f"💰 Space saved: {savings:,} bytes ({savings/1024:.1f}KB)")
    print(f"📈 Reduction: {savings_pct:.1f}%")
    print(f"🛡️ Backup available: {backup_path}")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)