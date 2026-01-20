#!/usr/bin/env python3
"""
Build memory-mapped dictionary files for LowMemorySymSpell.

Run this on a desktop machine to pre-build the mmap files:

    python build_mmap_files.py --output ./mmap_data

Then copy the generated files to your iOS app bundle:
- words.bin
- deletes.bin  
- bigrams.bin

At runtime, use LowMemorySymSpell with data_dir pointing to these files.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from low_memory_symspell import LowMemorySymSpell


def main():
    parser = argparse.ArgumentParser(
        description="Build mmap files for LowMemorySymSpell"
    )
    parser.add_argument(
        "--output", "-o",
        default="./mmap_data",
        help="Output directory for mmap files"
    )
    parser.add_argument(
        "--dictionary", "-d",
        default=None,
        help="Path to frequency dictionary (default: bundled English)"
    )
    parser.add_argument(
        "--bigrams", "-b", 
        default=None,
        help="Path to bigram dictionary (default: bundled English)"
    )
    parser.add_argument(
        "--max-edit-distance", "-e",
        type=int,
        default=2,
        help="Max edit distance (default: 2)"
    )
    parser.add_argument(
        "--prefix-length", "-p",
        type=int,
        default=7,
        help="Prefix length (default: 7)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Only include top N most frequent words (reduces size)"
    )
    args = parser.parse_args()
    
    # Default dictionary paths
    if args.dictionary is None:
        args.dictionary = os.path.join(
            os.path.dirname(__file__),
            "symspellpy",
            "frequency_dictionary_en_82_765.txt"
        )
    
    if args.bigrams is None:
        args.bigrams = os.path.join(
            os.path.dirname(__file__),
            "symspellpy",
            "frequency_bigramdictionary_en_243_342.txt"
        )
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Building mmap files...")
    print(f"  Dictionary: {args.dictionary}")
    print(f"  Bigrams: {args.bigrams}")
    print(f"  Output: {args.output}")
    print(f"  Max edit distance: {args.max_edit_distance}")
    print(f"  Prefix length: {args.prefix_length}")
    if args.top_n:
        print(f"  Top N words: {args.top_n}")
    print()
    
    # Build
    spell = LowMemorySymSpell(
        max_dictionary_edit_distance=args.max_edit_distance,
        prefix_length=args.prefix_length,
        data_dir=args.output,
    )
    
    if args.top_n:
        # Load with word limit
        success = spell.load_dictionary_top_n(
            args.dictionary, 
            n=args.top_n
        )
    else:
        success = spell.load_dictionary(args.dictionary)
    
    if not success:
        print(f"ERROR: Failed to load dictionary: {args.dictionary}")
        return 1
    
    print(f"  ✓ Loaded {spell.word_count:,} words")
    
    # Load bigrams
    success = spell.load_bigram_dictionary(args.bigrams)
    if not success:
        print(f"  ⚠ Failed to load bigrams (optional)")
    else:
        print(f"  ✓ Loaded {spell.bigram_count:,} bigrams")
    
    # Show file sizes
    print(f"\nGenerated files:")
    for name in ["words.bin", "deletes.bin", "bigrams.bin"]:
        path = os.path.join(args.output, name)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            size_mb = size_kb / 1024
            if size_mb >= 1:
                print(f"  {name}: {size_mb:.1f} MB")
            else:
                print(f"  {name}: {size_kb:.1f} KB")
    
    total_size = sum(
        os.path.getsize(os.path.join(args.output, f))
        for f in ["words.bin", "deletes.bin", "bigrams.bin"]
        if os.path.exists(os.path.join(args.output, f))
    )
    print(f"\n  Total: {total_size / (1024*1024):.1f} MB")
    
    # Cleanup (close mmap files)
    spell.close()
    
    print(f"\n✅ Done! Copy {args.output}/*.bin to your iOS app bundle.")
    print(f"   At runtime, use: LowMemorySymSpell(data_dir='path/to/files')")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
