#!/usr/bin/env python3
"""
Simple cleanup script to remove .DS_Store from sources directory
"""

import os
from pathlib import Path

def cleanup_ds_store():
    """Remove .DS_Store files from sources directory."""
    
    sources_dir = Path("/Users/will/AVI BUILD/Able3_Main_WithVoiceMode/data/sources")
    
    print("🧹 Cleaning up .DS_Store files...")
    
    ds_store_files = list(sources_dir.glob(".DS_Store"))
    
    if ds_store_files:
        for ds_store in ds_store_files:
            try:
                ds_store.unlink()
                print(f"   ✅ Removed: {ds_store.name}")
            except Exception as e:
                print(f"   ❌ Failed to remove {ds_store.name}: {e}")
    else:
        print("   ✅ No .DS_Store files found")
    
    # List remaining files
    remaining_files = [f for f in sources_dir.glob("*") if f.is_file()]
    print(f"\n📁 Files in sources directory: {len(remaining_files)}")
    for file in remaining_files:
        print(f"   - {file.name}")

if __name__ == "__main__":
    cleanup_ds_store()