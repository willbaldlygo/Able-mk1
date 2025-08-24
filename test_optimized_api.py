#!/usr/bin/env python3
"""Test the optimized API endpoints directly."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from services.storage_service import StorageService

def main():
    print("🧪 Testing Optimized API Functions")
    print("=" * 40)
    
    storage_service = StorageService()
    
    # Test metadata loading
    print("1. Testing metadata loading...")
    metadata = storage_service.load_metadata()
    print(f"   ✅ Loaded {len(metadata)} documents")
    for doc in metadata.values():
        print(f"   • {doc.name} ({doc.chunk_count} chunks)")
    
    # Test document summaries
    print("\n2. Testing document summaries...")
    summaries = storage_service.get_document_summaries()
    print(f"   ✅ Generated {len(summaries)} summaries")
    for summary in summaries:
        print(f"   • {summary.name} ({summary.chunk_count} chunks)")
    
    print("\n✅ All tests passed! API should work correctly.")

if __name__ == "__main__":
    main()