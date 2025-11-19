#!/usr/bin/env python3
"""
EQ Client - 1D Pixel Array with Color and Brightness
Visualizes audio as a 1D array where each pixel has color (from pattern) and brightness (from EQ)
Color patterns rotate based on music tempo/beat detection
"""

import argparse
import curses
import socket
import time
from collections import deque

HEADER = b"EQ"
VERSION = 1

# Color patterns that will rotate based on tempo
COLOR_PATTERNS = [
    # Pattern 1: Rainbow
    [
        curses.COLOR_RED,
        curses.COLOR_YELLOW,
        curses.COLOR_GREEN,
        curses.COLOR_CYAN,
        curses.COLOR_BLUE,
        curses.COLOR_MAGENTA,
    ],
    # Pattern 2: Fire
    [
        curses.COLOR_RED,
        curses.COLOR_RED,
        curses.COLOR_YELLOW,
        curses.COLOR_YELLOW,
        curses.COLOR_WHITE,
    ],
    # Pattern 3: Ocean
    [
        curses.COLOR_BLUE,
        curses.COLOR_CYAN,
        curses.COLOR_BLUE,
        curses.COLOR_CYAN,
        curses.COLOR_WHITE,
    ],
    # Pattern 4: Forest
    [
        curses.COLOR_GREEN,
        curses.COLOR_YELLOW,
        curses.COLOR_GREEN,
        curses.COLOR_CYAN,
    ],
    # Pattern 5: Sunset
    [
        curses.COLOR_MAGENTA,
        curses.COLOR_RED,
        curses.COLOR_YELLOW,
        curses.COLOR_RED,
    ],
    # Pattern 6: Monochrome
    [
        curses.COLOR_WHITE,
        curses.COLOR_WHITE,
        curses.COLOR_CYAN,
    ],
]


class BeatDetector:
    """
    Advanced beat detector using multi-band energy analysis and spectral flux
    Combines multiple techniques for more accurate beat detection
    """

    def __init__(self, history_size=30, sensitivity=1.0):
        # Multi-band energy history
        self.low_history = deque(maxlen=history_size)  # Bass/kick (0-8)
        self.mid_history = deque(maxlen=history_size)  # Snare (8-16)
        self.high_history = deque(maxlen=history_size)  # Hi-hat (16-24)

        # Spectral flux history (measures change in spectrum)
        self.flux_history = deque(maxlen=history_size)
        self.prev_bands = None

        # Beat timing
        self.last_beat_time = 0
        self.min_beat_interval = 0.25  # 250ms = max 240 BPM
        self.beat_intervals = deque(maxlen=8)  # Track tempo

        # Sensitivity (0.5 = less sensitive, 2.0 = more sensitive)
        self.sensitivity = sensitivity

        # Adaptive thresholds
        self.beat_strength_history = deque(maxlen=10)

    def _calculate_spectral_flux(self, bands_data):
        """Calculate spectral flux (sum of positive differences)"""
        if self.prev_bands is None:
            self.prev_bands = bands_data
            return 0.0

        flux = sum(max(0, bands_data[i] - self.prev_bands[i]) for i in range(len(bands_data)))
        self.prev_bands = bands_data
        return flux / len(bands_data)

    def _get_band_energy(self, bands_data, start, end):
        """Get average energy for a frequency band range"""
        if end > len(bands_data):
            end = len(bands_data)
        if start >= end:
            return 0.0
        return sum(bands_data[start:end]) / (end - start)

    def _adaptive_threshold(self, history, multiplier):
        """Calculate adaptive threshold based on variance"""
        if len(history) < 3:
            return float("inf")

        values = list(history)
        avg = sum(values) / len(values)
        variance = sum((x - avg) ** 2 for x in values) / len(values)
        std_dev = variance**0.5

        # Use mean + std_dev for adaptive threshold
        return avg + (std_dev * multiplier * self.sensitivity)

    def process(self, bands_data):
        """
        Process bands data and detect if beat occurred
        Returns: (is_beat, beat_strength) where beat_strength is 0.0-1.0
        """
        current_time = time.time()

        # Extract multi-band energies
        low_energy = self._get_band_energy(bands_data, 0, 8)  # Bass
        mid_energy = self._get_band_energy(bands_data, 8, 16)  # Snare
        high_energy = self._get_band_energy(bands_data, 16, 24)  # Hi-hat

        # Calculate spectral flux
        flux = self._calculate_spectral_flux(bands_data)

        # Update histories
        self.low_history.append(low_energy)
        self.mid_history.append(mid_energy)
        self.high_history.append(high_energy)
        self.flux_history.append(flux)

        # Need enough history
        if len(self.low_history) < self.low_history.maxlen // 2:
            return False, 0.0

        # Calculate adaptive thresholds for each band
        low_threshold = self._adaptive_threshold(self.low_history, 1.3)
        mid_threshold = self._adaptive_threshold(self.mid_history, 1.5)
        flux_threshold = self._adaptive_threshold(self.flux_history, 1.4)

        # Multi-criteria beat detection
        beat_indicators = 0
        beat_strength = 0.0

        # Low frequency (bass/kick) - most important
        if low_energy > low_threshold:
            beat_indicators += 2  # Weight: 2
            beat_strength += min(1.0, (low_energy - low_threshold) / low_threshold)

        # Mid frequency (snare)
        if mid_energy > mid_threshold:
            beat_indicators += 1  # Weight: 1
            beat_strength += 0.5 * min(1.0, (mid_energy - mid_threshold) / mid_threshold)

        # Spectral flux (sudden change)
        if flux > flux_threshold:
            beat_indicators += 1  # Weight: 1
            beat_strength += 0.5 * min(1.0, (flux - flux_threshold) / flux_threshold)

        # Normalize strength
        beat_strength = min(1.0, beat_strength / 2.0)

        # Check timing constraint
        time_since_last = current_time - self.last_beat_time
        if time_since_last < self.min_beat_interval:
            return False, beat_strength

        # Detect beat if we have strong enough indicators
        # Need at least 2 indicators (e.g., low + mid, or low + flux)
        is_beat = beat_indicators >= 2

        if is_beat:
            # Track beat timing for tempo analysis
            if self.last_beat_time > 0:
                interval = time_since_last
                self.beat_intervals.append(interval)

            self.last_beat_time = current_time
            self.beat_strength_history.append(beat_strength)

        return is_beat, beat_strength

    def get_tempo_estimate(self):
        """Estimate current tempo in BPM based on recent beat intervals"""
        if len(self.beat_intervals) < 3:
            return 0

        # Use median to avoid outliers
        intervals = sorted(list(self.beat_intervals))
        median_interval = intervals[len(intervals) // 2]

        if median_interval > 0:
            bpm = 60.0 / median_interval
            return round(bpm)
        return 0


def init_colors():
    """Initialize curses color pairs"""
    # Create color pairs for each color with different brightness levels
    pair_id = 1
    color_pairs = {}

    for color in [
        curses.COLOR_RED,
        curses.COLOR_GREEN,
        curses.COLOR_YELLOW,
        curses.COLOR_BLUE,
        curses.COLOR_MAGENTA,
        curses.COLOR_CYAN,
        curses.COLOR_WHITE,
        curses.COLOR_BLACK,
    ]:
        color_pairs[color] = pair_id
        curses.init_pair(pair_id, color, curses.COLOR_BLACK)
        pair_id += 1

    return color_pairs


def get_brightness_char(brightness):
    """Get character representing brightness level (0.0 - 1.0)"""
    # Characters from dimmest to brightest
    chars = " ░▒▓█"
    idx = int(brightness * (len(chars) - 1))
    return chars[min(idx, len(chars) - 1)]


def draw_1d_pixel_array(stdscr, bands_data, color_pairs, current_pattern, y_offset=5):
    """
    Draw 1D pixel array with two representations:
    1. Brightness-simulated pixels (using characters ░▒▓█)
    2. Height-based pixels (normal color, height represents brightness)
    """
    H, W = stdscr.getmaxyx()
    num_pixels = len(bands_data)

    # Get current color pattern
    pattern = COLOR_PATTERNS[current_pattern]

    # Calculate pixel width
    pixel_width = max(1, W // num_pixels)

    # Row 1: Brightness-simulated pixels (simulated brightness using characters)
    y1 = y_offset
    try:
        stdscr.move(y1, 0)
        stdscr.clrtoeol()
        for i, brightness_val in enumerate(bands_data):
            brightness = brightness_val / 255.0  # Normalize to 0-1
            color = pattern[i % len(pattern)]
            char = get_brightness_char(brightness)

            x = i * pixel_width
            if x < W:
                try:
                    # Use character to simulate brightness
                    stdscr.addstr(y1, x, char * pixel_width, curses.color_pair(color_pairs[color]))
                except curses.error:
                    pass
    except curses.error:
        pass

    # Row 2-10: Height-based pixels (height represents brightness)
    max_height = 8  # Maximum height in characters
    base_y = y_offset + 10

    # Clear the area
    for row in range(max_height + 1):
        try:
            stdscr.move(base_y - row, 0)
            stdscr.clrtoeol()
        except curses.error:
            pass

    # Draw each pixel as a vertical bar
    for i, brightness_val in enumerate(bands_data):
        brightness = brightness_val / 255.0
        height = int(brightness * max_height)
        color = pattern[i % len(pattern)]

        x = i * pixel_width

        # Draw vertical bar from bottom up
        for h in range(height):
            y = base_y - h
            if 0 <= y < H and x < W:
                try:
                    stdscr.addstr(y, x, "█" * pixel_width, curses.color_pair(color_pairs[color]))
                except curses.error:
                    pass


def main_curses(stdscr, args):
    """Main curses loop"""
    # Setup curses
    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(True)  # Non-blocking input
    stdscr.timeout(0)

    # Initialize colors
    curses.start_color()
    curses.use_default_colors()
    color_pairs = init_colors()

    # Setup UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.setblocking(False)

    # Initialize beat detector
    beat_detector = BeatDetector()

    # Display initial message
    H, W = stdscr.getmaxyx()
    stdscr.clear()
    stdscr.addstr(0, 0, f"1D Pixel Array EQ - {args.host}:{args.port} | [Q]uit")
    stdscr.addstr(H - 1, 0, "Waiting for audio data...")
    stdscr.refresh()

    bands_data = None
    frame_interval = 1.0 / args.fps
    frame_count = 0
    current_pattern = 0
    beat_count = 0
    auto_switch_enabled = True  # Auto-switch on beat detection
    last_beat_strength = 0.0  # For visual feedback

    try:
        while True:
            t_start = time.time()

            # Drain all packets, keep latest
            latest = None
            while True:
                try:
                    data, _ = sock.recvfrom(4096)
                    if len(data) >= 4 and data[:2] == HEADER and data[2] == VERSION:
                        latest = data[3:]
                except BlockingIOError:
                    break

            if latest is not None:
                bands_data = list(latest)

                # Beat detection - switch pattern on beat (if auto-switch enabled)
                is_beat, beat_strength = beat_detector.process(bands_data)

                # Decay beat strength for visual feedback
                if is_beat:
                    beat_count += 1
                    last_beat_strength = beat_strength
                    if auto_switch_enabled:
                        current_pattern = (current_pattern + 1) % len(COLOR_PATTERNS)
                else:
                    # Gradually fade out beat indicator
                    last_beat_strength = max(0.0, last_beat_strength - 0.1)

            # Draw visualization
            if bands_data is not None:
                H, W = stdscr.getmaxyx()

                # Clear screen
                stdscr.clear()

                # Header with BPM info
                pattern_names = ["Rainbow", "Fire", "Ocean", "Forest", "Sunset", "Mono"]
                auto_status = "ON" if auto_switch_enabled else "OFF"
                bpm = beat_detector.get_tempo_estimate()
                bpm_str = f"{bpm} BPM" if bpm > 0 else "Detecting..."
                header = f"1D Pixel Array EQ | Pattern: {pattern_names[current_pattern]} | Auto: {auto_status} | Beats: {beat_count} | Tempo: {bpm_str}"
                try:
                    stdscr.addstr(0, 0, header[: W - 1], curses.A_BOLD)
                except curses.error:
                    pass

                # Pattern list (for reference)
                try:
                    patterns_line = (
                        "Patterns: [1]Rainbow [2]Fire [3]Ocean [4]Forest [5]Sunset [6]Mono"
                    )
                    stdscr.addstr(1, 0, patterns_line[: W - 1], curses.A_DIM)
                except curses.error:
                    pass

                # Labels and beat indicator
                try:
                    stdscr.addstr(3, 0, "Brightness-simulated (using characters):", curses.A_BOLD)

                    # Beat strength indicator (visual feedback)
                    beat_bar_len = int(last_beat_strength * 20)
                    beat_bar = "█" * beat_bar_len + "░" * (20 - beat_bar_len)
                    beat_label = f"Beat: [{beat_bar}] {int(last_beat_strength * 100)}%"
                    stdscr.addstr(
                        3,
                        W - len(beat_label) - 1,
                        beat_label,
                        curses.color_pair(color_pairs[curses.COLOR_RED])
                        if last_beat_strength > 0.5
                        else curses.A_DIM,
                    )

                    stdscr.addstr(7, 0, "Height-based (normal pixels):", curses.A_BOLD)
                except curses.error:
                    pass

                # Draw 1D pixel array
                draw_1d_pixel_array(stdscr, bands_data, color_pairs, current_pattern)

                # Footer with controls
                sensitivity_display = f"{beat_detector.sensitivity:.1f}"
                footer = f"[Q]uit | [1-6] Pattern | [←→] Nav | [SPACE] Auto | [+/-] Sensitivity: {sensitivity_display} | [R] Reset"
                try:
                    stdscr.addstr(H - 1, 0, footer[: W - 1])
                except curses.error:
                    pass

                stdscr.refresh()
                frame_count += 1

            # Check for key press
            try:
                key = stdscr.getch()
                if key in (ord("q"), ord("Q")):
                    break

                # Number keys 1-6: Select specific pattern
                elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6")):
                    pattern_idx = int(chr(key)) - 1
                    if 0 <= pattern_idx < len(COLOR_PATTERNS):
                        current_pattern = pattern_idx
                        auto_switch_enabled = False  # Disable auto-switch when manually selecting

                # Arrow keys: Navigate patterns
                elif key == curses.KEY_RIGHT:
                    current_pattern = (current_pattern + 1) % len(COLOR_PATTERNS)
                    auto_switch_enabled = False  # Disable auto-switch when manually navigating
                elif key == curses.KEY_LEFT:
                    current_pattern = (current_pattern - 1) % len(COLOR_PATTERNS)
                    auto_switch_enabled = False  # Disable auto-switch when manually navigating

                # Space: Toggle auto-switch
                elif key == ord(" "):
                    auto_switch_enabled = not auto_switch_enabled

                # A: Enable auto-switch
                elif key in (ord("a"), ord("A")):
                    auto_switch_enabled = True

                # M: Disable auto-switch (Manual mode)
                elif key in (ord("m"), ord("M")):
                    auto_switch_enabled = False

                # P: Next pattern (legacy support)
                elif key in (ord("p"), ord("P")):
                    current_pattern = (current_pattern + 1) % len(COLOR_PATTERNS)
                    auto_switch_enabled = False

                # +/=: Increase sensitivity
                elif key in (ord("+"), ord("=")):
                    beat_detector.sensitivity = min(3.0, beat_detector.sensitivity + 0.1)

                # -/_: Decrease sensitivity
                elif key in (ord("-"), ord("_")):
                    beat_detector.sensitivity = max(0.3, beat_detector.sensitivity - 0.1)

                # R: Reset to default sensitivity
                elif key in (ord("r"), ord("R")):
                    beat_detector.sensitivity = 1.0
            except curses.error:
                pass

            # Frame rate control
            dt = time.time() - t_start
            if dt < frame_interval:
                time.sleep(frame_interval - dt)

    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


def main():
    """Entry point"""
    ap = argparse.ArgumentParser(
        description="1D Pixel Array EQ - Color patterns rotate with music tempo"
    )
    ap.add_argument("--host", default="0.0.0.0", help="Listen address")
    ap.add_argument("--port", type=int, default=31337, help="UDP port")
    ap.add_argument("--fps", type=int, default=30, help="Update frequency")
    args = ap.parse_args()

    # Run curses application
    curses.wrapper(main_curses, args)
    print("\nVisualization stopped.")


if __name__ == "__main__":
    main()
