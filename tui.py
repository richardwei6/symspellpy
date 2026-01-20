#!/usr/bin/env python3
"""
Interactive TUI for testing symspellpy

Shows suggestions as you type with auto-replace for high-confidence corrections.

Usage:
    cd /path/to/symspellpy
    source .venv/bin/activate   # if using virtual environment
    python tui.py

Requirements:
    pip install editdistpy

Controls:
    Ctrl+A  - Toggle auto-replace
    Ctrl+Z  - Undo last auto-replace
    Ctrl+V  - Cycle verbosity (TOP / CLOSEST / ALL)
    Ctrl+E  - Cycle max edit distance (1 / 2)
    Ctrl+T  - Toggle transfer casing
    Ctrl+U  - Clear input
    TAB     - Accept top suggestion
    Enter   - Accept selected suggestion
    ↑/↓     - Navigate suggestions
    ESC     - Quit
"""

import curses
import os
import sys
import time
from pathlib import Path

# Add the parent directory to path so we can import symspellpy locally
sys.path.insert(0, str(Path(__file__).parent))

from symspellpy import SymSpell, Verbosity

# Import low-memory version
try:
    from low_memory_symspell import LowMemorySymSpell
    LOW_MEMORY_AVAILABLE = True
except ImportError:
    LOW_MEMORY_AVAILABLE = False

# Try to import psutil for resource monitoring (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Fallback: use resource module on Unix
try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False


class SymSpellTUI:
    """Terminal UI for testing symspellpy spell checking."""

    # Auto-replace threshold: suggestions with confidence >= this are auto-applied
    AUTO_REPLACE_THRESHOLD = 0.75  # 75%
    MEMORY_LIMIT_MB = 40.0  # Target memory limit for low-memory mode

    def __init__(self, stdscr, low_memory_mode: bool = False):
        self.stdscr = stdscr
        self.text_buffer = ""
        self.cursor_pos = 0
        self.selected = 0
        self.verbosity = Verbosity.ALL
        self.max_edit_distance = 2
        self.transfer_casing = False
        self.include_unknown = True
        self.status_message = None
        self.auto_replace_enabled = True
        self.last_auto_replacement = None  # (original, replacement, position)
        self.auto_replace_message = None
        
        # Memory mode
        self.low_memory_mode = low_memory_mode and LOW_MEMORY_AVAILABLE
        self.low_memory_spell = None
        self.db_size_mb = 0.0

        # Resource monitoring
        self.process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None
        self.start_time = time.time()
        self.last_cpu_check = time.time()
        self.cpu_percent = 0.0
        
        # Capture memory before loading dictionaries
        self.pre_dict_memory = self._get_memory_mb()

        # Dictionary paths
        pkg_dir = Path(__file__).parent / "symspellpy"
        self.dictionary_path = pkg_dir / "frequency_dictionary_en_82_765.txt"
        self.bigram_path = pkg_dir / "frequency_bigramdictionary_en_243_342.txt"

        if self.low_memory_mode:
            # Use low-memory mmap-backed version
            # Check for prebuilt files first (much lower memory)
            prebuilt_dir = Path(__file__).parent / "mmap_data_full"
            
            self.low_memory_spell = LowMemorySymSpell(
                memory_limit_mb=self.MEMORY_LIMIT_MB,
                max_dictionary_edit_distance=2,
                prefix_length=7,
                data_dir=str(prebuilt_dir) if prebuilt_dir.exists() else None,
            )
            
            # Try prebuilt first (no memory spike!)
            if prebuilt_dir.exists() and self.low_memory_spell.load_prebuilt():
                self.dict_loaded = True
                self.bigram_loaded = self.low_memory_spell.bigram_count > 0
            else:
                # Fall back to building (uses more memory during build)
                self.dict_loaded = self.low_memory_spell.load_dictionary(
                    str(self.dictionary_path), term_index=0, count_index=1
                )
                self.bigram_loaded = self.low_memory_spell.load_bigram_dictionary(
                    str(self.bigram_path), term_index=0, count_index=2
                )
            
            self.sym_spell = None
            self.db_size_mb = self.low_memory_spell.get_db_size_mb()
        else:
            # Use standard in-memory SymSpell
            self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            self.dict_loaded = self.sym_spell.load_dictionary(
                self.dictionary_path, term_index=0, count_index=1
            )
            self.bigram_loaded = self.sym_spell.load_bigram_dictionary(
                self.bigram_path, term_index=0, count_index=2
            )
        
        # Capture memory after loading dictionaries
        self.post_dict_memory = self._get_memory_mb()
        self.dict_memory = max(0, self.post_dict_memory - self.pre_dict_memory)

        # Lookup timing stats
        self.last_lookup_time_ms = 0.0

        # Setup curses
        curses.curs_set(1)
        curses.start_color()
        curses.use_default_colors()

        # Color scheme - warm terracotta/rust aesthetic
        curses.init_pair(1, curses.COLOR_CYAN, -1)      # Headers/labels
        curses.init_pair(2, curses.COLOR_GREEN, -1)     # Valid/success
        curses.init_pair(3, curses.COLOR_YELLOW, -1)    # Scores/info
        curses.init_pair(4, curses.COLOR_RED, -1)       # Corrections/warnings
        curses.init_pair(5, curses.COLOR_MAGENTA, -1)   # Current word/highlight
        curses.init_pair(6, curses.COLOR_BLUE, -1)      # Muted info
        curses.init_pair(7, curses.COLOR_WHITE, -1)     # Normal text

        self.stdscr.keypad(True)

    @property
    def speller(self):
        """Get the active spell checker (low-memory or standard)."""
        return self.low_memory_spell if self.low_memory_mode else self.sym_spell
    
    @property
    def word_count(self) -> int:
        """Get word count from active spell checker."""
        if self.low_memory_mode and self.low_memory_spell:
            return self.low_memory_spell.word_count
        elif self.sym_spell:
            return self.sym_spell.word_count
        return 0

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            if PSUTIL_AVAILABLE and self.process:
                return self.process.memory_info().rss / (1024 * 1024)
            elif RESOURCE_AVAILABLE:
                # resource.getrusage returns memory in KB on Linux, bytes on macOS
                usage = resource.getrusage(resource.RUSAGE_SELF)
                if sys.platform == 'darwin':
                    return usage.ru_maxrss / (1024 * 1024)  # bytes to MB
                else:
                    return usage.ru_maxrss / 1024  # KB to MB
        except Exception:
            pass
        return 0.0

    def _get_cpu_percent(self) -> float:
        """Get CPU usage percentage."""
        try:
            if PSUTIL_AVAILABLE and self.process:
                # Update CPU percent (non-blocking)
                now = time.time()
                if now - self.last_cpu_check >= 0.5:  # Update every 0.5s
                    self.cpu_percent = self.process.cpu_percent(interval=None)
                    self.last_cpu_check = now
                return self.cpu_percent
        except Exception:
            pass
        return 0.0

    def _get_resource_stats(self) -> dict:
        """Get current resource usage statistics."""
        memory_mb = self._get_memory_mb()
        cpu_pct = self._get_cpu_percent()
        uptime = time.time() - self.start_time
        
        return {
            'memory_mb': memory_mb,
            'dict_memory_mb': self.dict_memory,
            'cpu_percent': cpu_pct,
            'uptime_sec': uptime,
        }

    def run(self):
        """Main event loop."""
        while True:
            self.draw_ui()

            try:
                key = self.stdscr.getch()
            except:
                continue

            # Clear messages on keypress (except for undo)
            if key != 26:  # Not Ctrl+Z
                self.auto_replace_message = None
            if key not in (curses.KEY_UP, curses.KEY_DOWN):
                self.status_message = None

            if key == 27:  # ESC
                break
            elif key == 1:  # Ctrl+A - toggle auto-replace
                self.auto_replace_enabled = not self.auto_replace_enabled
                self.status_message = f"Auto-replace: {'ON' if self.auto_replace_enabled else 'OFF'}"
            elif key == 26:  # Ctrl+Z - undo last auto-replacement
                self.undo_auto_replace()
                self.auto_replace_message = None
            elif key == 21:  # Ctrl+U - clear input
                self.text_buffer = ""
                self.cursor_pos = 0
                self.selected = 0
                self.last_auto_replacement = None
            elif key == 22:  # Ctrl+V - cycle verbosity
                self.cycle_verbosity()
            elif key == 20:  # Ctrl+T - toggle transfer casing
                self.transfer_casing = not self.transfer_casing
                self.status_message = f"Transfer casing: {'ON' if self.transfer_casing else 'OFF'}"
            elif key == 5:  # Ctrl+E - cycle edit distance
                self.max_edit_distance = (self.max_edit_distance % 2) + 1
                self.status_message = f"Max edit distance: {self.max_edit_distance}"
            elif key in (curses.KEY_BACKSPACE, 127, 8):  # Backspace
                self.handle_backspace()
                self.selected = 0
            elif key == curses.KEY_LEFT:
                self.cursor_pos = max(0, self.cursor_pos - 1)
            elif key == curses.KEY_RIGHT:
                self.cursor_pos = min(len(self.text_buffer), self.cursor_pos + 1)
            elif key == curses.KEY_HOME:
                self.cursor_pos = 0
            elif key == curses.KEY_END:
                self.cursor_pos = len(self.text_buffer)
            elif key == curses.KEY_UP:
                self.selected = max(0, self.selected - 1)
            elif key == curses.KEY_DOWN:
                self.selected += 1
            elif key == 9:  # TAB - accept top suggestion
                self.accept_suggestion(index=0)
            elif key in (curses.KEY_ENTER, 10):  # ENTER - accept selected suggestion
                self.accept_suggestion(index=self.selected)
            elif 32 <= key <= 126:  # Printable characters
                self.handle_char(chr(key))
                self.selected = 0

    def cycle_verbosity(self):
        """Cycle through verbosity levels."""
        verbosities = [Verbosity.TOP, Verbosity.CLOSEST, Verbosity.ALL]
        current_idx = verbosities.index(self.verbosity)
        self.verbosity = verbosities[(current_idx + 1) % len(verbosities)]
        self.status_message = f"Verbosity: {self.verbosity.name}"

    def handle_char(self, char: str):
        """Insert character at cursor position."""
        # Check for auto-replacement when space is typed
        if char == ' ':
            self.try_auto_replace()
        
        self.text_buffer = (
            self.text_buffer[:self.cursor_pos] +
            char +
            self.text_buffer[self.cursor_pos:]
        )
        self.cursor_pos += 1

    def calculate_confidence(self, suggestion, all_suggestions) -> float:
        """
        Calculate a confidence score for a suggestion (0.0 to 1.0).
        
        Score is based on:
        - Edit distance (lower = higher confidence)
        - Relative frequency among suggestions at same distance
        """
        if not suggestion or not all_suggestions:
            return 0.0
        
        distance = suggestion.distance
        count = suggestion.count
        
        # Base score from edit distance (distance 0 = 1.0, distance 1 = 0.6, distance 2 = 0.3)
        distance_score = max(0, 1.0 - (distance * 0.4))
        
        # Get max frequency at this distance for normalization
        same_distance = [s for s in all_suggestions if s.distance == distance]
        max_count = max(s.count for s in same_distance) if same_distance else count
        
        # Frequency boost (0.0 to 0.3 based on relative frequency)
        if max_count > 0:
            freq_score = 0.3 * (count / max_count)
        else:
            freq_score = 0.0
        
        # If it's an exact match (distance 0), it's 100% confident
        if distance == 0:
            return 1.0
        
        return min(1.0, distance_score + freq_score)

    def try_auto_replace(self):
        """Auto-replace the current word if top suggestion has >= 75% confidence."""
        if not self.auto_replace_enabled or not self.text_buffer:
            return
        
        word = self.get_current_word()
        if not word:
            return
        
        best_replacement = None
        best_confidence = 0.0
        replacement_type = None
        
        try:
            # Check lookup suggestions
            suggestions = self.speller.lookup(
                word,
                self.verbosity,
                max_edit_distance=self.max_edit_distance,
                include_unknown=False,
                transfer_casing=self.transfer_casing,
            )
            
            if suggestions:
                top = suggestions[0]
                confidence = self.calculate_confidence(top, suggestions)
                
                # Only consider if misspelled (distance > 0)
                if top.distance > 0 and confidence > best_confidence:
                    best_replacement = top.term
                    best_confidence = confidence
                    replacement_type = "fix"
        except Exception:
            pass
        
        try:
            # Check if word can be split (for joined words)
            if len(word) >= 4:
                segment_result = self.speller.word_segmentation(
                    word.lower(),
                    max_edit_distance=self.max_edit_distance,
                )
                
                if segment_result:
                    corrected = segment_result.corrected_string.strip()
                    if ' ' in corrected:
                        # Calculate split confidence
                        split_confidence = 0.85 if segment_result.distance_sum == 0 else 0.7
                        
                        if split_confidence > best_confidence:
                            best_replacement = corrected
                            best_confidence = split_confidence
                            replacement_type = "split"
        except Exception:
            pass
        
        # Apply auto-replacement if confidence meets threshold
        if best_replacement and best_confidence >= self.AUTO_REPLACE_THRESHOLD:
            start, end = self.get_word_boundaries()
            
            # Store for undo
            self.last_auto_replacement = (word, best_replacement, start)
            
            # Replace
            self.text_buffer = (
                self.text_buffer[:start] +
                best_replacement +
                self.text_buffer[end:]
            )
            self.cursor_pos = start + len(best_replacement)
            
            # Show message
            pct = int(best_confidence * 100)
            self.auto_replace_message = f'Auto ({replacement_type}): "{word}" → "{best_replacement}" ({pct}%)'

    def undo_auto_replace(self):
        """Undo the last auto-replacement."""
        if not self.last_auto_replacement:
            return
        
        original, replacement, position = self.last_auto_replacement
        
        # Find where the replacement is (after space was added)
        expected = replacement + ' '
        
        if (position + len(expected) <= len(self.text_buffer) and
            self.text_buffer[position:position + len(expected)] == expected):
            
            # Restore original + space
            self.text_buffer = (
                self.text_buffer[:position] +
                original + ' ' +
                self.text_buffer[position + len(expected):]
            )
            self.cursor_pos = position + len(original) + 1
            self.status_message = f'Undid: "{replacement}" → "{original}"'
        
        self.last_auto_replacement = None

    def get_word_boundaries(self) -> tuple[int, int]:
        """Get start and end position of current word at cursor."""
        if not self.text_buffer or self.cursor_pos == 0:
            return (0, 0)
        
        start = self.cursor_pos
        while start > 0 and not self.text_buffer[start - 1].isspace():
            start -= 1
        
        end = self.cursor_pos
        while end < len(self.text_buffer) and not self.text_buffer[end].isspace():
            end += 1
        
        return (start, end)

    def handle_backspace(self):
        """Delete character before cursor."""
        if self.cursor_pos > 0:
            self.text_buffer = (
                self.text_buffer[:self.cursor_pos - 1] +
                self.text_buffer[self.cursor_pos:]
            )
            self.cursor_pos -= 1

    def get_input_text(self) -> str:
        """Get the full input text for processing."""
        return self.text_buffer.strip()

    def get_current_word(self) -> str:
        """Extract the word at the current cursor position (for lookup mode)."""
        if not self.text_buffer:
            return ""

        start = self.cursor_pos
        while start > 0 and not self.text_buffer[start - 1].isspace():
            start -= 1

        end = self.cursor_pos
        while end < len(self.text_buffer) and not self.text_buffer[end].isspace():
            end += 1

        return self.text_buffer[start:end]

    def get_all_suggestions(self):
        """Get suggestions from lookup + check if word can be split."""
        text = self.get_input_text()
        word = self.get_current_word()
        
        results = {
            'lookup': [],
            'split_suggestion': None,  # If word can be split into multiple words
            'is_valid': False,
        }
        
        if not word:
            return text, word, results

        try:
            # Lookup - uses current word at cursor
            lookup_start = time.time()
            suggestions = self.speller.lookup(
                word,
                self.verbosity,
                max_edit_distance=self.max_edit_distance,
                include_unknown=self.include_unknown,
                transfer_casing=self.transfer_casing,
            )
            self.last_lookup_time_ms = (time.time() - lookup_start) * 1000
            
            # Add confidence scores
            results['lookup'] = [
                (s, self.calculate_confidence(s, suggestions))
                for s in suggestions
            ]
            # Check if word is valid (exact match exists)
            results['is_valid'] = any(s.distance == 0 for s in suggestions)
        except Exception:
            pass

        # Check if word can be split into multiple words (for joined words like "thequick")
        # Only do this if word is long enough and not already valid
        if len(word) >= 4:
            try:
                # Try word segmentation on the current word
                segment_result = self.speller.word_segmentation(
                    word.lower(),
                    max_edit_distance=self.max_edit_distance,
                )
                
                if segment_result:
                    corrected = segment_result.corrected_string.strip()
                    # Only show as suggestion if it actually split into multiple words
                    # and the split is different from the original
                    if ' ' in corrected and corrected.replace(' ', '') != word.lower():
                        # Calculate a confidence score for the split
                        # Lower distance_sum = better split
                        # Use log_prob_sum for confidence (less negative = more probable)
                        split_confidence = 0.7 if segment_result.distance_sum <= 1 else 0.5
                        results['split_suggestion'] = (corrected, split_confidence, segment_result)
                    elif ' ' in corrected:
                        # It split but no corrections needed
                        split_confidence = 0.85
                        results['split_suggestion'] = (corrected, split_confidence, segment_result)
            except Exception:
                pass

        return text, word, results

    def accept_suggestion(self, index: int = 0):
        """Accept a suggestion by index (lookup or split)."""
        text, word, results = self.get_all_suggestions()
        
        if not word:
            return

        # Build the same combined list as in draw_suggestions_list
        lookup_suggestions = results.get('lookup', [])
        split_sugg = results.get('split_suggestion')
        
        all_suggestions = []
        
        if split_sugg:
            split_text, split_conf, _ = split_sugg
            all_suggestions.append({
                'term': split_text,
                'confidence': split_conf,
                'distance': 0,
                'type': 'split',
            })
        
        for sugg, conf in lookup_suggestions:
            all_suggestions.append({
                'term': sugg.term,
                'confidence': conf,
                'distance': sugg.distance,
                'type': 'lookup',
            })
        
        # Sort same way as display
        all_suggestions.sort(key=lambda x: (
            0 if x['distance'] == 0 and x['type'] == 'lookup' else 1,
            -x['confidence']
        ))
        
        if not all_suggestions:
            return

        # Clamp index
        index = min(index, len(all_suggestions) - 1)
        replacement = all_suggestions[index]['term']

        # Find word boundaries
        start, end = self.get_word_boundaries()

        # Replace with space
        self.text_buffer = (
            self.text_buffer[:start] +
            replacement + ' ' +
            self.text_buffer[end:]
        )
        self.cursor_pos = start + len(replacement) + 1
        self.selected = 0

    def draw_ui(self):
        """Render the entire UI."""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()

        if h < 20 or w < 65:
            self.stdscr.addstr(0, 0, "Terminal too small!")
            self.stdscr.addstr(1, 0, f"Need 65x20, got {w}x{h}")
            self.stdscr.refresh()
            return

        # ═══════════════════════════════════════════════════════════════
        # Header with resource monitor
        # ═══════════════════════════════════════════════════════════════
        title = "═" * min(w-1, 75)
        self.stdscr.addstr(0, 0, title, curses.color_pair(5))
        
        header = " SymSpellPy Interactive Tester "
        self.stdscr.addstr(1, 0, header, curses.color_pair(5) | curses.A_BOLD)
        
        # Resource stats on the right side of header
        stats = self._get_resource_stats()
        mem_str = f"RAM: {stats['memory_mb']:.1f}MB"
        cpu_str = f"CPU: {stats['cpu_percent']:.1f}%"
        uptime_min = int(stats['uptime_sec'] // 60)
        uptime_sec = int(stats['uptime_sec'] % 60)
        time_str = f"{uptime_min}:{uptime_sec:02d}"
        
        lookup_str = f"Lookup: {self.last_lookup_time_ms:.1f}ms"
        resource_info = f"│ {mem_str} │ {cpu_str} │ {lookup_str} │ ⏱ {time_str}"
        resource_start = w - len(resource_info) - 1
        if resource_start > len(header):
            self.stdscr.addstr(1, resource_start, resource_info[:w-resource_start-1], curses.color_pair(6))
        
        self.stdscr.addstr(2, 0, title, curses.color_pair(5))

        # Dictionary info line
        dict_status = "✓" if self.dict_loaded else "✗"
        bigram_status = "✓" if self.bigram_loaded else "✗"
        mode_str = "LOW-MEM" if self.low_memory_mode else "FULL"
        if self.low_memory_mode:
            dict_info = f"[{mode_str}] Dict:{dict_status} │ Words: {self.word_count:,} │ DB: {self.db_size_mb:.1f}MB"
        else:
            dict_info = f"[{mode_str}] Dict:{dict_status} Bigram:{bigram_status} │ Words: {self.word_count:,} │ RAM: {stats['dict_memory_mb']:.1f}MB"
        self.stdscr.addstr(3, 0, dict_info[:w-1], curses.color_pair(6))

        # ═══════════════════════════════════════════════════════════════
        # Controls line
        # ═══════════════════════════════════════════════════════════════
        controls = "TAB=accept │ ↑↓=select │ ^Z=undo │ ^A=auto │ ESC=quit"
        self.stdscr.addstr(5, 0, controls[:w-1], curses.color_pair(6))

        # Auto-replace status
        threshold_pct = int(self.AUTO_REPLACE_THRESHOLD * 100)
        self.stdscr.addstr(6, 0, f"Auto-replace (≥{threshold_pct}%): ", curses.color_pair(6))
        auto_status = "ON" if self.auto_replace_enabled else "OFF"
        auto_color = curses.color_pair(2) if self.auto_replace_enabled else curses.color_pair(4)
        self.stdscr.addstr(auto_status, auto_color | curses.A_BOLD)

        self.stdscr.addstr(" │ ", curses.color_pair(6))
        self.stdscr.addstr("Verbosity: ", curses.color_pair(6))
        self.stdscr.addstr(self.verbosity.name, curses.color_pair(3))

        self.stdscr.addstr(" │ ", curses.color_pair(6))
        self.stdscr.addstr("Edit: ", curses.color_pair(6))
        self.stdscr.addstr(str(self.max_edit_distance), curses.color_pair(3))

        # ═══════════════════════════════════════════════════════════════
        # Input Area
        # ═══════════════════════════════════════════════════════════════
        self.stdscr.addstr(8, 0, ">>> ", curses.color_pair(2) | curses.A_BOLD)
        prompt_len = 4

        # Display input text
        display_text = self.text_buffer[:w - prompt_len - 1] if self.text_buffer else ""
        self.stdscr.addstr(8, prompt_len, display_text, curses.color_pair(2))

        # Auto-replace message
        if self.auto_replace_message:
            self.stdscr.addstr(9, 0, self.auto_replace_message[:w-1], 
                              curses.color_pair(2) | curses.A_BOLD)

        # Status message
        if self.status_message:
            msg_row = 10 if self.auto_replace_message else 9
            self.stdscr.addstr(msg_row, 0, self.status_message[:w-1], curses.color_pair(3))

        # ═══════════════════════════════════════════════════════════════
        # Suggestions Panel
        # ═══════════════════════════════════════════════════════════════
        start_row = 11
        self.draw_suggestions_list(start_row, h, w)

        # Position cursor
        cursor_x = min(prompt_len + self.cursor_pos, w - 1)
        self.stdscr.move(8, cursor_x)
        self.stdscr.refresh()

    def draw_suggestions_list(self, start_row: int, h: int, w: int):
        """Draw suggestions in a single list like autocorrect main.py."""
        text, word, results = self.get_all_suggestions()

        if not word:
            hint = "(type to see suggestions)"
            self.stdscr.addstr(start_row, 0, hint[:w-1], curses.color_pair(6))
            return

        row = start_row

        # ─────────────────────────────────────────────────────────────
        # Current word status
        # ─────────────────────────────────────────────────────────────
        is_valid = results.get('is_valid', False)
        split_sugg = results.get('split_suggestion')
        
        # Determine status - valid, misspelled, or joined words
        if is_valid:
            word_status = "VALID"
            status_marker = "✓"
            status_color = curses.color_pair(2)
        elif split_sugg:
            word_status = "JOINED WORDS?"
            status_marker = "⊕"
            status_color = curses.color_pair(3)
        else:
            word_status = "MISSPELLED"
            status_marker = "✗"
            status_color = curses.color_pair(4)

        self.stdscr.addstr(row, 0, "Current: ", curses.color_pair(1))
        self.stdscr.addstr(f'"{word}"', curses.color_pair(5) | curses.A_BOLD)
        self.stdscr.addstr(" ")
        self.stdscr.addstr(f"{status_marker} {word_status}", status_color)
        row += 2

        # ─────────────────────────────────────────────────────────────
        # Build combined suggestions list
        # ─────────────────────────────────────────────────────────────
        lookup_suggestions = results.get('lookup', [])
        
        # Combine lookup suggestions with split suggestion
        all_suggestions = []
        
        # Add split suggestion first if it exists and has good confidence
        if split_sugg:
            split_text, split_conf, _ = split_sugg
            all_suggestions.append({
                'term': split_text,
                'confidence': split_conf,
                'distance': 0,  # Not applicable for splits
                'type': 'split',
            })
        
        # Add lookup suggestions
        for sugg, conf in lookup_suggestions:
            all_suggestions.append({
                'term': sugg.term,
                'confidence': conf,
                'distance': sugg.distance,
                'type': 'lookup',
            })
        
        # Sort by confidence (highest first), but keep exact matches at top
        all_suggestions.sort(key=lambda x: (
            0 if x['distance'] == 0 and x['type'] == 'lookup' else 1,
            -x['confidence']
        ))
        
        if all_suggestions:
            self.selected = min(self.selected, len(all_suggestions) - 1)
            
            label = "Suggestions:"
            self.stdscr.addstr(row, 0, label[:w-1], curses.color_pair(1))
            row += 1

            for i, sugg in enumerate(all_suggestions[:10]):
                if row >= h - 2:
                    break

                # Selection arrow
                arrow = "→ " if i == self.selected else "  "
                confidence = sugg['confidence']
                
                # Color and tag based on type and confidence
                if sugg['type'] == 'split':
                    color = curses.color_pair(5)  # Magenta for splits
                    tag = "split"
                elif sugg['distance'] == 0:
                    color = curses.color_pair(2)  # Green - exact
                    tag = "exact"
                elif confidence >= 0.75:
                    color = curses.color_pair(2)  # Green - high confidence
                    tag = "high"
                elif confidence >= 0.5:
                    color = curses.color_pair(3)  # Yellow - medium
                    tag = "med"
                else:
                    color = curses.color_pair(4)  # Red - low confidence
                    tag = "low"

                attr = curses.A_BOLD if i == self.selected else 0

                # Draw line
                self.stdscr.addstr(row, 0, arrow, color | curses.A_BOLD)
                self.stdscr.addstr(f"{sugg['term']}", color | attr)

                # Confidence percentage and info
                conf_pct = int(confidence * 100)
                if sugg['type'] == 'split':
                    info = f" ({conf_pct}%) [{tag}]"
                else:
                    info = f" ({conf_pct}%) [d:{sugg['distance']}] [{tag}]"
                col = len(arrow) + len(sugg['term']) + 1
                if col < w - 25:
                    self.stdscr.addstr(row, col, info[:w-col-1], curses.color_pair(3))

                row += 1

            if len(all_suggestions) > 10:
                self.stdscr.addstr(row, 0, f"  ... and {len(all_suggestions) - 10} more", 
                                  curses.color_pair(6))
        else:
            self.stdscr.addstr(row, 0, "  (no suggestions)", curses.color_pair(6))


def main(stdscr, low_memory_mode: bool = False):
    """Entry point for curses application."""
    app = SymSpellTUI(stdscr, low_memory_mode=low_memory_mode)
    app.run()
    
    # Cleanup low-memory resources
    if app.low_memory_spell:
        app.low_memory_spell.close()


if __name__ == "__main__":
    import argparse

    # Check if running in an interactive terminal
    if not sys.stdin.isatty():
        print("Error: This TUI requires an interactive terminal.")
        print("Please run directly in a terminal, not as a background process.")
        sys.exit(1)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SymSpellPy Interactive Tester")
    parser.add_argument(
        "--low-memory", "-l",
        action="store_true",
        help=f"Use low-memory mode with mmap-backed dictionary (target: {SymSpellTUI.MEMORY_LIMIT_MB}MB)"
    )
    args = parser.parse_args()

    print("═" * 65)
    print(" SymSpellPy Interactive Tester")
    print(" with Auto-Replace & Resource Monitor")
    print("═" * 65)
    print()
    
    # Show mode
    if args.low_memory:
        if LOW_MEMORY_AVAILABLE:
            print(f"  ⚡ LOW-MEMORY MODE (mmap-backed, target: {SymSpellTUI.MEMORY_LIMIT_MB}MB)")
        else:
            print("  ⚠ Low-memory mode not available")
            args.low_memory = False
    else:
        print("  💾 FULL-MEMORY MODE (faster, uses ~150MB)")
        print("     Use --low-memory or -l for memory-limited mode")
    print()
    
    # Show resource monitoring status
    if PSUTIL_AVAILABLE:
        print("  ✓ psutil installed - full CPU/RAM monitoring")
    else:
        print("  ⚠ psutil not installed - limited monitoring")
        print("    Install with: pip install psutil")
    print()
    
    print("Features:")
    print("  • Auto-replace misspellings (≥75% confidence)")
    print("  • Auto-split joined words ('thequick' → 'the quick')")
    print("  • Real-time CPU & memory monitoring")
    print("  • Ctrl+Z to undo auto-replacements")
    print()
    print("Try typing:")
    print("  • 'memebers '     → auto-corrects to 'members'")
    print("  • 'thequick '     → auto-splits to 'the quick'")
    print()
    
    if args.low_memory:
        prebuilt_dir = Path(__file__).parent / "mmap_data_full"
        if prebuilt_dir.exists() and (prebuilt_dir / "words.bin").exists():
            print("Loading prebuilt mmap dictionary (fast, low memory)...")
        else:
            print("Building mmap dictionary (this may take a moment)...")
            print("  TIP: Run 'python build_mmap_files.py' first for lower memory usage")
    
    print("Press any key to start...")

    try:
        curses.wrapper(lambda stdscr: main(stdscr, low_memory_mode=args.low_memory))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure you're running in an interactive terminal.")
        sys.exit(1)

    print("\n✨ Goodbye!")
