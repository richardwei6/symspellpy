"""
Low-memory SymSpell wrapper using memory-mapped files.

This module provides a memory-efficient alternative to the standard SymSpell
by storing dictionaries in memory-mapped binary files instead of RAM.
"""

import gc
import mmap
import os
import struct
import tempfile
from pathlib import Path
from typing import Optional, List, BinaryIO

from symspellpy import Verbosity
from symspellpy.editdistance import EditDistance, DistanceAlgorithm
from symspellpy.suggest_item import SuggestItem
from symspellpy.composition import Composition


class MMapDictionary:
    """
    Memory-mapped dictionary for word frequencies.
    
    Binary format:
    - Header: [num_words: 4 bytes]
    - Word index: [offset: 4 bytes] * num_words (points into data section)
    - Data section: [word_len: 1 byte][word: variable][count: 8 bytes] * num_words
    
    Words are stored sorted for binary search.
    """
    
    HEADER_SIZE = 4  # num_words
    INDEX_ENTRY_SIZE = 4  # offset
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file: Optional[BinaryIO] = None
        self.mm: Optional[mmap.mmap] = None
        self.num_words = 0
        self.index_start = self.HEADER_SIZE
        self.data_start = 0
        self._word_cache: dict[str, int] = {}  # Small LRU cache
        self._cache_max = 1000
    
    def build(self, words: list[tuple[str, int]]):
        """Build the mmap file from a list of (word, count) pairs."""
        # Sort words alphabetically
        words.sort(key=lambda x: x[0])
        self.num_words = len(words)
        
        # Calculate offsets
        self.data_start = self.HEADER_SIZE + (self.num_words * self.INDEX_ENTRY_SIZE)
        
        with open(self.file_path, 'wb') as f:
            # Write header
            f.write(struct.pack('<I', self.num_words))
            
            # First pass: calculate offsets and write index
            current_offset = self.data_start
            offsets = []
            for word, count in words:
                offsets.append(current_offset)
                # word_len (1 byte) + word + count (8 bytes)
                current_offset += 1 + len(word.encode('utf-8')) + 8
            
            # Write index
            for offset in offsets:
                f.write(struct.pack('<I', offset))
            
            # Write data
            for word, count in words:
                word_bytes = word.encode('utf-8')
                f.write(struct.pack('<B', len(word_bytes)))
                f.write(word_bytes)
                f.write(struct.pack('<Q', count))
    
    def open(self):
        """Open the mmap file for reading."""
        if not os.path.exists(self.file_path):
            return False
        
        self.file = open(self.file_path, 'rb')
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Read header
        self.num_words = struct.unpack('<I', self.mm[0:4])[0]
        self.data_start = self.HEADER_SIZE + (self.num_words * self.INDEX_ENTRY_SIZE)
        
        return True
    
    def close(self):
        """Close the mmap file."""
        if self.mm:
            self.mm.close()
            self.mm = None
        if self.file:
            self.file.close()
            self.file = None
    
    def _get_word_at_index(self, idx: int) -> tuple[str, int]:
        """Get word and count at index."""
        if idx < 0 or idx >= self.num_words or not self.mm:
            return ("", 0)
        
        # Get offset from index
        idx_pos = self.HEADER_SIZE + (idx * self.INDEX_ENTRY_SIZE)
        offset = struct.unpack('<I', self.mm[idx_pos:idx_pos + 4])[0]
        
        # Read word
        word_len = self.mm[offset]
        word = self.mm[offset + 1:offset + 1 + word_len].decode('utf-8')
        count = struct.unpack('<Q', self.mm[offset + 1 + word_len:offset + 9 + word_len])[0]
        
        return (word, count)
    
    def get(self, word: str) -> int:
        """Get count for a word using binary search."""
        if not self.mm:
            return 0
        
        # Check cache
        if word in self._word_cache:
            return self._word_cache[word]
        
        # Binary search
        left, right = 0, self.num_words - 1
        while left <= right:
            mid = (left + right) // 2
            mid_word, mid_count = self._get_word_at_index(mid)
            
            if mid_word == word:
                # Add to cache
                if len(self._word_cache) >= self._cache_max:
                    # Simple cache eviction: clear half
                    keys = list(self._word_cache.keys())[:self._cache_max // 2]
                    for k in keys:
                        del self._word_cache[k]
                self._word_cache[word] = mid_count
                return mid_count
            elif mid_word < word:
                left = mid + 1
            else:
                right = mid - 1
        
        return 0
    
    def __contains__(self, word: str) -> bool:
        return self.get(word) > 0


class MMapDeletes:
    """
    Memory-mapped deletes index for spell checking.
    
    Binary format:
    - Header: [num_entries: 4 bytes]
    - Offset index: [offset: 4 bytes] * num_entries (for binary search without loading keys)
    - Entries: sorted by delete_key
      - [key_len: 1 byte][key: variable][num_suggestions: 2 bytes][suggestion_indices: 4 bytes each]
    
    We do NOT load keys into memory - we read them from mmap during binary search.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file: Optional[BinaryIO] = None
        self.mm: Optional[mmap.mmap] = None
        self.num_entries = 0
        self.index_start = 4  # After header
        self.data_start = 0
    
    def build(self, deletes: dict[str, list[int]]):
        """Build the mmap file from deletes dict."""
        # Sort keys
        sorted_keys = sorted(deletes.keys())
        self.num_entries = len(sorted_keys)
        
        # Calculate data start (after header + index)
        self.data_start = 4 + (self.num_entries * 4)
        
        with open(self.file_path, 'wb') as f:
            # Write header
            f.write(struct.pack('<I', self.num_entries))
            
            # First pass: calculate offsets
            offsets = []
            current_offset = self.data_start
            for key in sorted_keys:
                offsets.append(current_offset)
                key_bytes = key.encode('utf-8')
                num_suggestions = len(deletes[key])
                current_offset += 1 + len(key_bytes) + 2 + (num_suggestions * 4)
            
            # Write offset index
            for offset in offsets:
                f.write(struct.pack('<I', offset))
            
            # Write entries
            for key in sorted_keys:
                suggestions = deletes[key]
                key_bytes = key.encode('utf-8')
                
                f.write(struct.pack('<B', len(key_bytes)))
                f.write(key_bytes)
                f.write(struct.pack('<H', len(suggestions)))
                for idx in suggestions:
                    f.write(struct.pack('<I', idx))
    
    def open(self):
        """Open mmap file."""
        if not os.path.exists(self.file_path):
            return False
        
        self.file = open(self.file_path, 'rb')
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Read header
        self.num_entries = struct.unpack('<I', self.mm[0:4])[0]
        self.data_start = 4 + (self.num_entries * 4)
        
        return True
    
    def close(self):
        """Close the mmap file."""
        if self.mm:
            self.mm.close()
            self.mm = None
        if self.file:
            self.file.close()
            self.file = None
    
    def _get_key_at_index(self, idx: int) -> str:
        """Read key at index from mmap (no memory allocation for storage)."""
        if not self.mm or idx < 0 or idx >= self.num_entries:
            return ""
        
        # Get offset from index
        idx_pos = 4 + (idx * 4)
        offset = struct.unpack('<I', self.mm[idx_pos:idx_pos + 4])[0]
        
        # Read key
        key_len = self.mm[offset]
        key = self.mm[offset + 1:offset + 1 + key_len].decode('utf-8')
        return key
    
    def _get_suggestions_at_index(self, idx: int) -> list[int]:
        """Read suggestions at index from mmap."""
        if not self.mm or idx < 0 or idx >= self.num_entries:
            return []
        
        # Get offset from index
        idx_pos = 4 + (idx * 4)
        offset = struct.unpack('<I', self.mm[idx_pos:idx_pos + 4])[0]
        
        # Skip key, read suggestions
        key_len = self.mm[offset]
        num_suggestions = struct.unpack('<H', 
            self.mm[offset + 1 + key_len:offset + 3 + key_len])[0]
        
        suggestions = []
        data_start = offset + 3 + key_len
        for i in range(num_suggestions):
            suggestion_idx = struct.unpack('<I', self.mm[data_start + i*4:data_start + (i+1)*4])[0]
            suggestions.append(suggestion_idx)
        return suggestions
    
    def get(self, key: str) -> list[int]:
        """Get suggestion indices for a delete key using binary search."""
        if not self.mm or self.num_entries == 0:
            return []
        
        # Binary search - read keys from mmap, don't store in memory
        left, right = 0, self.num_entries - 1
        while left <= right:
            mid = (left + right) // 2
            mid_key = self._get_key_at_index(mid)
            
            if mid_key == key:
                return self._get_suggestions_at_index(mid)
            elif mid_key < key:
                left = mid + 1
            else:
                right = mid - 1
        
        return []


class LowMemorySymSpell:
    """
    Memory-efficient SymSpell using memory-mapped files.
    
    Args:
        memory_limit_mb: Target memory limit in MB (affects cache sizes)
        max_dictionary_edit_distance: Max edit distance for lookups
        prefix_length: Length of word prefixes for spell checking
        data_dir: Directory for mmap files (uses temp if None)
    """
    
    def __init__(
        self,
        memory_limit_mb: float = 50.0,
        max_dictionary_edit_distance: int = 2,
        prefix_length: int = 7,
        data_dir: Optional[str] = None,
    ):
        self.memory_limit_mb = memory_limit_mb
        self.max_edit_distance = max_dictionary_edit_distance
        self.prefix_length = prefix_length
        
        # Create data directory
        if data_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="symspell_mmap_")
            self.data_dir = self.temp_dir
        else:
            self.temp_dir = None
            self.data_dir = data_dir
            os.makedirs(data_dir, exist_ok=True)
        
        # File paths
        self.words_path = os.path.join(self.data_dir, "words.bin")
        self.deletes_path = os.path.join(self.data_dir, "deletes.bin")
        self.bigrams_path = os.path.join(self.data_dir, "bigrams.bin")
        
        # Memory-mapped structures
        self.words = MMapDictionary(self.words_path)
        self.deletes = MMapDeletes(self.deletes_path)
        self.bigrams = MMapDictionary(self.bigrams_path)
        
        # Cache sizes based on memory limit (very small to stay under limit)
        cache_budget = max(100, int(memory_limit_mb * 10))  # ~10 entries per MB
        self.words._cache_max = cache_budget
        
        # Statistics
        self.word_count = 0
        self.bigram_count = 0
        
        # For edit distance calculation (no need for full SymSpell)
        self._distance_comparer = EditDistance(DistanceAlgorithm.DAMERAU_OSA_FAST)
    
    def load_prebuilt(self) -> bool:
        """
        Load pre-built mmap files without building.
        
        Use this for iOS/low-memory devices where files are pre-built
        and shipped with the app. Returns True if pre-built files exist.
        """
        # Check if pre-built files exist
        if not os.path.exists(self.words_path):
            return False
        if not os.path.exists(self.deletes_path):
            return False
        
        # Just open the mmap files - no building needed
        if not self.words.open():
            return False
        self.word_count = self.words.num_words
        
        if not self.deletes.open():
            return False
        
        # Bigrams are optional
        if os.path.exists(self.bigrams_path):
            self.bigrams.open()
            self.bigram_count = self.bigrams.num_words
        
        return True
    
    def load_dictionary(
        self,
        corpus: str,
        term_index: int = 0,
        count_index: int = 1,
        separator: str = " ",
        encoding: Optional[str] = None,
    ) -> bool:
        """Load dictionary from file into mmap structures."""
        corpus_path = Path(corpus)
        if not corpus_path.exists():
            return False
        
        # Read all words
        words_list: list[tuple[str, int]] = []
        
        with open(corpus_path, "r", encoding=encoding) as f:
            for line in f:
                parts = line.rstrip().split(separator)
                if len(parts) < 2:
                    continue
                
                term = parts[term_index]
                try:
                    count = int(parts[count_index])
                except ValueError:
                    continue
                
                words_list.append((term, count))
        
        # Build words mmap (sorted)
        words_list.sort(key=lambda x: x[0])
        self.words.build(words_list)
        self.words.open()
        self.word_count = self.words.num_words
        
        # Build deletes index
        deletes_dict: dict[str, list[int]] = {}
        
        for idx, (term, _) in enumerate(words_list):
            # Add empty string for short words
            if len(term) <= self.max_edit_distance:
                if "" not in deletes_dict:
                    deletes_dict[""] = []
                deletes_dict[""].append(idx)
            
            # Generate deletes from prefix
            prefix = term[:self.prefix_length] if len(term) > self.prefix_length else term
            deletes = self._generate_deletes(prefix, 0, set())
            deletes.add(prefix)
            
            for delete in deletes:
                if delete not in deletes_dict:
                    deletes_dict[delete] = []
                deletes_dict[delete].append(idx)
        
        # Free words_list - no longer needed
        del words_list
        
        # Build deletes mmap
        self.deletes.build(deletes_dict)
        self.deletes.open()
        
        # Free deletes_dict - no longer needed
        del deletes_dict
        
        # Force garbage collection to reclaim memory
        gc.collect()
        
        return True
    
    def _generate_deletes(self, word: str, edit_distance: int, deletes: set) -> set:
        """Generate all deletes within edit distance."""
        edit_distance += 1
        if not word:
            return deletes
        
        for i in range(len(word)):
            delete = word[:i] + word[i + 1:]
            if delete not in deletes:
                deletes.add(delete)
                if edit_distance < self.max_edit_distance:
                    self._generate_deletes(delete, edit_distance, deletes)
        
        return deletes
    
    def load_dictionary_top_n(
        self,
        corpus: str,
        n: int,
        term_index: int = 0,
        count_index: int = 1,
        separator: str = " ",
        encoding: Optional[str] = None,
    ) -> bool:
        """
        Load only the top N most frequent words from dictionary.
        
        This significantly reduces memory usage and file sizes.
        Recommended for iOS: n=10000-30000 for good spell checking.
        """
        corpus_path = Path(corpus)
        if not corpus_path.exists():
            return False
        
        # Read all words with counts
        words_list: list[tuple[str, int]] = []
        
        with open(corpus_path, "r", encoding=encoding) as f:
            for line in f:
                parts = line.rstrip().split(separator)
                if len(parts) < 2:
                    continue
                
                term = parts[term_index]
                try:
                    count = int(parts[count_index])
                except ValueError:
                    continue
                
                words_list.append((term, count))
        
        # Sort by count descending and take top N
        words_list.sort(key=lambda x: -x[1])
        words_list = words_list[:n]
        
        # Now sort alphabetically for binary search
        words_list.sort(key=lambda x: x[0])
        
        # Build words mmap
        self.words.build(words_list)
        self.words.open()
        self.word_count = self.words.num_words
        
        # Build deletes index
        deletes_dict: dict[str, list[int]] = {}
        
        for idx, (term, _) in enumerate(words_list):
            if len(term) <= self.max_edit_distance:
                if "" not in deletes_dict:
                    deletes_dict[""] = []
                deletes_dict[""].append(idx)
            
            prefix = term[:self.prefix_length] if len(term) > self.prefix_length else term
            deletes = self._generate_deletes(prefix, 0, set())
            deletes.add(prefix)
            
            for delete in deletes:
                if delete not in deletes_dict:
                    deletes_dict[delete] = []
                deletes_dict[delete].append(idx)
        
        del words_list
        
        self.deletes.build(deletes_dict)
        self.deletes.open()
        
        del deletes_dict
        gc.collect()
        
        return True
    
    def load_bigram_dictionary(
        self,
        corpus: str,
        term_index: int = 0,
        count_index: int = 2,
        separator: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> bool:
        """Load bigram dictionary into mmap."""
        corpus_path = Path(corpus)
        if not corpus_path.exists():
            return False
        
        bigrams_list: list[tuple[str, int]] = []
        min_parts = 3 if separator is None else 2
        
        with open(corpus_path, "r", encoding=encoding) as f:
            for line in f:
                parts = line.rstrip().split(separator)
                if len(parts) < min_parts:
                    continue
                
                try:
                    count = int(parts[count_index])
                except ValueError:
                    continue
                
                if separator is None:
                    key = f"{parts[term_index]} {parts[term_index + 1]}"
                else:
                    key = parts[term_index]
                
                bigrams_list.append((key, count))
        
        self.bigrams.build(bigrams_list)
        self.bigrams.open()
        self.bigram_count = self.bigrams.num_words
        
        # Free bigrams_list
        del bigrams_list
        gc.collect()
        
        return True
    
    def _get_word_by_index(self, idx: int) -> tuple[str, int]:
        """Get word and count by index (reads from mmap)."""
        return self.words._get_word_at_index(idx)
    
    def lookup(
        self,
        phrase: str,
        verbosity: Verbosity,
        max_edit_distance: Optional[int] = None,
        include_unknown: bool = False,
        transfer_casing: bool = False,
        **kwargs,
    ) -> List[SuggestItem]:
        """Find suggested spellings for a word."""
        if max_edit_distance is None:
            max_edit_distance = self.max_edit_distance
        
        if max_edit_distance > self.max_edit_distance:
            raise ValueError("distance too large")
        
        suggestions: List[SuggestItem] = []
        phrase_len = len(phrase)
        
        original_phrase = phrase
        if transfer_casing:
            phrase = phrase.lower()
        
        # Check for exact match
        count = self.words.get(phrase)
        if count > 0:
            if transfer_casing:
                suggestions.append(SuggestItem(original_phrase, 0, count))
            else:
                suggestions.append(SuggestItem(phrase, 0, count))
            if verbosity != Verbosity.ALL:
                return suggestions
        
        if max_edit_distance == 0:
            if include_unknown and not suggestions:
                suggestions.append(SuggestItem(phrase, max_edit_distance + 1, 0))
            return suggestions
        
        # Generate candidates
        considered_suggestions: set = {phrase}
        max_edit_distance_2 = max_edit_distance
        candidates = []
        
        phrase_prefix_len = min(phrase_len, self.prefix_length)
        candidates.append(phrase[:phrase_prefix_len])
        
        candidate_pointer = 0
        while candidate_pointer < len(candidates):
            candidate = candidates[candidate_pointer]
            candidate_pointer += 1
            candidate_len = len(candidate)
            len_diff = phrase_prefix_len - candidate_len
            
            if len_diff > max_edit_distance_2:
                if verbosity == Verbosity.ALL:
                    continue
                break
            
            # Get suggestions from deletes index
            suggestion_indices = self.deletes.get(candidate)
            
            for idx in suggestion_indices:
                suggestion, suggestion_count = self._get_word_by_index(idx)
                if not suggestion or suggestion == phrase:
                    continue
                
                suggestion_len = len(suggestion)
                
                if (abs(suggestion_len - phrase_len) > max_edit_distance_2 or
                    suggestion_len < candidate_len or
                    (suggestion_len == candidate_len and suggestion != candidate)):
                    continue
                
                if suggestion in considered_suggestions:
                    continue
                considered_suggestions.add(suggestion)
                
                # Calculate edit distance
                distance = self._distance_comparer.compare(
                    phrase, suggestion, max_edit_distance_2
                )
                
                if distance < 0:
                    continue
                
                if distance <= max_edit_distance_2:
                    item = SuggestItem(suggestion, distance, suggestion_count)
                    
                    if suggestions:
                        if verbosity == Verbosity.CLOSEST:
                            if distance < max_edit_distance_2:
                                suggestions = []
                        elif verbosity == Verbosity.TOP:
                            if distance < max_edit_distance_2 or suggestion_count > suggestions[0].count:
                                max_edit_distance_2 = distance
                                suggestions[0] = item
                            continue
                    
                    if verbosity != Verbosity.ALL:
                        max_edit_distance_2 = distance
                    suggestions.append(item)
            
            # Generate more candidates
            if len_diff < max_edit_distance and candidate_len <= self.prefix_length:
                if verbosity != Verbosity.ALL and len_diff >= max_edit_distance_2:
                    continue
                for i in range(candidate_len):
                    delete = candidate[:i] + candidate[i + 1:]
                    if delete not in considered_suggestions:
                        candidates.append(delete)
        
        if len(suggestions) > 1:
            suggestions.sort()
        
        if transfer_casing:
            suggestions = [
                SuggestItem(
                    self._transfer_casing(original_phrase, s.term),
                    s.distance,
                    s.count,
                )
                for s in suggestions
            ]
        
        if include_unknown and not suggestions:
            suggestions.append(SuggestItem(phrase, max_edit_distance + 1, 0))
        
        return suggestions
    
    def _transfer_casing(self, source: str, target: str) -> str:
        """Transfer casing from source to target."""
        result = []
        for i, char in enumerate(target):
            if i < len(source) and source[i].isupper():
                result.append(char.upper())
            else:
                result.append(char)
        return "".join(result)
    
    def lookup_compound(
        self,
        phrase: str,
        max_edit_distance: int,
        transfer_casing: bool = False,
        **kwargs,
    ) -> List[SuggestItem]:
        """Compound word correction."""
        words = phrase.split()
        corrected_words = []
        total_distance = 0
        
        for word in words:
            suggestions = self.lookup(
                word, Verbosity.TOP, max_edit_distance,
                transfer_casing=transfer_casing
            )
            if suggestions:
                corrected_words.append(suggestions[0].term)
                total_distance += suggestions[0].distance
            else:
                corrected_words.append(word)
                total_distance += max_edit_distance + 1
        
        result_term = " ".join(corrected_words)
        return [SuggestItem(result_term, total_distance, 1)]
    
    def word_segmentation(
        self,
        phrase: str,
        max_edit_distance: Optional[int] = None,
        **kwargs,
    ) -> Composition:
        """Word segmentation."""
        if max_edit_distance is None:
            max_edit_distance = self.max_edit_distance
        
        phrase = phrase.lower().replace(" ", "")
        result_parts = []
        i = 0
        total_distance = 0
        
        while i < len(phrase):
            best_word = None
            best_len = 0
            best_count = 0
            
            # Try different lengths (longest first)
            for length in range(min(20, len(phrase) - i), 0, -1):
                word = phrase[i:i + length]
                count = self.words.get(word)
                
                if count > 0:
                    if length > best_len or (length == best_len and count > best_count):
                        best_word = word
                        best_len = length
                        best_count = count
            
            if best_word:
                result_parts.append(best_word)
                i += best_len
            else:
                # Try with spelling correction
                test_word = phrase[i:i + min(10, len(phrase) - i)]
                suggestions = self.lookup(test_word, Verbosity.TOP, max_edit_distance)
                
                if suggestions and suggestions[0].distance <= max_edit_distance:
                    result_parts.append(suggestions[0].term)
                    total_distance += suggestions[0].distance
                    i += len(test_word)
                else:
                    result_parts.append(phrase[i])
                    total_distance += 1
                    i += 1
        
        corrected = " ".join(result_parts)
        return Composition(
            segmented_string=corrected,
            corrected_string=corrected,
            distance_sum=total_distance,
            log_prob_sum=-50.0,
        )
    
    def get_db_size_mb(self) -> float:
        """Get total size of mmap files in MB."""
        total = 0
        for path in [self.words_path, self.deletes_path, self.bigrams_path]:
            if os.path.exists(path):
                total += os.path.getsize(path)
        return total / (1024 * 1024)
    
    def close(self):
        """Close all mmap files and cleanup."""
        self.words.close()
        self.deletes.close()
        self.bigrams.close()
        
        # Cleanup temp files
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.close()
        except:
            pass
