import cv2
import numpy as np
from scipy import signal
import time  # Import the time module
from scipy.fft import fft, fftfreq
import tkinter as tk
from tkinter import ttk
from collections import deque
import scipy
from datetime import datetime
import os
import mne  # To handle EDF files
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

class AdvancedFrequencyAnalyzer:
    def __init__(self, edf_file):
        self.history_length = 100
        self.running = True
        self.pattern_history = deque(maxlen=self.history_length)
        self.logged_data = []
        
        # Create captures directory
        self.capture_dir = "screen_captures"
        os.makedirs(self.capture_dir, exist_ok=True)
        
        # Analysis parameters
        self.low_freq = 0.5
        self.high_freq = 30
        self.filter_strength = 0.5
        self.fft_scale = 1.0
        self.resonance_q = 1.0
        self.filter_order = 4
        self.filter_type = 'bandpass'
        self.step_size = 256  # Default speed ~1 second
        self.pattern_space_dim = 2  # Default Pattern Space dimension

        # Load EDF file
        self.edf_file = edf_file
        self.data, self.times, self.raw_info = self.load_eeg_data()
        self.channel_idx = 0  # Default channel

        # Create visualization windows
        cv2.namedWindow('Filtered Signal', cv2.WINDOW_NORMAL)
        cv2.namedWindow('FFT Analysis', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Pattern Space', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('FFT Analysis', 1000, 400)
        cv2.resizeWindow('Filtered Signal', 1000, 400)
        cv2.resizeWindow('Pattern Space', 800, 800)
        
        self.init_gui()

    def load_eeg_data(self):
        """Load EEG data from an EDF file using MNE."""
        raw = mne.io.read_raw_edf(self.edf_file, preload=True)
        data, times = raw[:]
        print(f"Loaded EDF file with {len(raw.ch_names)} channels.")
        return data, times, raw.info
    
    def init_gui(self):
        """Initialize the control GUI."""
        self.root = tk.Tk()
        self.root.title("Advanced Frequency Analysis")
        
        # Channel selection
        channel_frame = ttk.LabelFrame(self.root, text="Channel Selection")
        channel_frame.pack(fill="x", padx=5, pady=5)
        self.channel_var = tk.IntVar(value=self.channel_idx)
        ttk.Label(channel_frame, text="Channel:").pack(side="left")
        self.channel_combo = ttk.Combobox(
            channel_frame, values=list(range(self.data.shape[0])),
            textvariable=self.channel_var, state="readonly"
        )
        self.channel_combo.pack(side="left", padx=5)
        self.channel_combo.bind("<<ComboboxSelected>>", self.update_channel)
        
        # Step size control (Speed)
        self.step_var = tk.IntVar(value=self.step_size)  # Initialize step variable
        ttk.Label(channel_frame, text="Step Size (Samples):").pack(side="left")
        self.step_slider = ttk.Scale(
            channel_frame, from_=64, to=1024, orient="horizontal",
            variable=self.step_var
        )
        self.step_slider.pack(side="left", fill="x", expand=True)
        
        # Delay control slider
        speed_frame = ttk.LabelFrame(self.root, text="Speed Control (Delay)")
        speed_frame.pack(fill="x", padx=5, pady=5)
        self.delay_var = tk.DoubleVar(value=0.1)  # Default delay in seconds
        ttk.Label(speed_frame, text="Delay (s):").pack(side="left")
        self.delay_slider = ttk.Scale(
            speed_frame, from_=0.01, to=2.0, orient="horizontal",
            variable=self.delay_var
        )
        self.delay_slider.pack(side="left", fill="x", expand=True)

        # Frequency sliders
        freq_frame = ttk.LabelFrame(self.root, text="Filter Frequencies")
        freq_frame.pack(fill="x", padx=5, pady=5)
        self.low_freq_var = tk.DoubleVar(value=self.low_freq)
        ttk.Label(freq_frame, text="Low (Hz):").pack(side="left")
        ttk.Scale(freq_frame, from_=0.1, to=50, variable=self.low_freq_var,
                command=self.update_low_freq).pack(side="left", fill="x", expand=True)
        
        self.high_freq_var = tk.DoubleVar(value=self.high_freq)
        ttk.Label(freq_frame, text="High (Hz):").pack(side="left")
        ttk.Scale(freq_frame, from_=0.1, to=50, variable=self.high_freq_var,
                command=self.update_high_freq).pack(side="left", fill="x", expand=True)
        
        # Toggle Pattern Space dimensions
        self.pattern_dim_var = tk.IntVar(value=self.pattern_space_dim)
        ttk.Label(freq_frame, text="Pattern Space Dim:").pack(side="left")
        self.pattern_dim_combo = ttk.Combobox(
            freq_frame, values=[2, 3], textvariable=self.pattern_dim_var, state="readonly"
        )
        self.pattern_dim_combo.pack(side="left", padx=5)
        self.pattern_dim_combo.bind("<<ComboboxSelected>>", self.update_pattern_dim)
        
        # Buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(btn_frame, text="Capture Screens", command=self.capture_screen).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Save Data", command=self.save_logged_data).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Quit", command=self.quit).pack(side="left", padx=5)

    def visualize_fft(self, freqs, fft_vals):
        """Create enhanced FFT visualization"""
        vis_height = 400
        vis_width = 1000
        vis = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        
        # Scale FFT values
        fft_vals = fft_vals * self.fft_scale
        
        # Plot FFT
        max_freq = 50
        freq_mask = freqs <= max_freq
        freqs = freqs[freq_mask]
        fft_vals = fft_vals[freq_mask]
        
        if len(fft_vals) > 0:
            # Normalize and apply log scaling for better visualization
            fft_normalized = np.log1p(fft_vals)
            fft_normalized = fft_normalized / (np.max(fft_normalized) + 1e-10)
            
            # Draw frequency bars with color gradient
            for i in range(len(freqs)-1):
                x = int(freqs[i] * vis_width / max_freq)
                height = int(fft_normalized[i] * vis_height)
                
                # Color based on frequency band
                if freqs[i] <= self.low_freq:
                    color = (0, 0, 255)  # Red for low frequencies
                elif freqs[i] >= self.high_freq:
                    color = (255, 0, 0)  # Blue for high frequencies
                else:
                    color = (0, 255, 0)  # Green for passband
                    
                cv2.line(vis, (x, vis_height), (x, vis_height - height), color, 2)
            
            # Draw frequency band limits
            low_x = int(self.low_freq * vis_width / max_freq)
            high_x = int(self.high_freq * vis_width / max_freq)
            cv2.line(vis, (low_x, 0), (low_x, vis_height), (255, 255, 255), 1)
            cv2.line(vis, (high_x, 0), (high_x, vis_height), (255, 255, 255), 1)
        
        # Add labels and info
        cv2.putText(vis, f"Filter: {self.filter_type}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Low: {self.low_freq:.1f}Hz High: {self.high_freq:.1f}Hz", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis

    def compute_fft(self, signal):
        n = len(signal)
        fft_vals = fft(signal)
        freqs = fftfreq(n, d=1 / 256)
        pos_mask = freqs >= 0
        return freqs[pos_mask], np.abs(fft_vals[pos_mask])

    def apply_frequency_filter(self, input_signal):
        """Apply configurable filter to signal"""
        nyq = 50.0  # Nyquist frequency
        
        if self.filter_type == 'bandpass':
            low = self.low_freq / nyq
            high = self.high_freq / nyq
            b, a = scipy.signal.butter(self.filter_order, [low, high], btype='band')
        elif self.filter_type == 'lowpass':
            cutoff = self.high_freq / nyq
            b, a = scipy.signal.butter(self.filter_order, cutoff, btype='low')
        else:  # highpass
            cutoff = self.low_freq / nyq
            b, a = scipy.signal.butter(self.filter_order, cutoff, btype='high')
        
        # Apply filter with forward-backward filtering
        filtered = scipy.signal.filtfilt(b, a, input_signal)
        
        # Apply resonance/Q factor
        if self.resonance_q > 1.0:
            sos = scipy.signal.butter(self.filter_order, [self.low_freq, self.high_freq], 
                              btype='band', fs=100, output='sos')
            filtered = scipy.signal.sosfilt(sos, filtered)
        
        # Mix original and filtered based on strength
        return input_signal * (1 - self.filter_strength) + filtered * self.filter_strength

    def update_channel(self, event):
        self.channel_idx = self.channel_var.get()
        print(f"Switched to channel {self.channel_idx}")

    def update_low_freq(self, value):
        self.low_freq = float(value)
        print(f"Low Frequency updated to {self.low_freq} Hz")

    def update_high_freq(self, value):
        self.high_freq = float(value)
        print(f"High Frequency updated to {self.high_freq} Hz")
    
    def update_pattern_dim(self, event):
        self.pattern_space_dim = self.pattern_dim_var.get()
        print(f"Updated Pattern Space dimension to {self.pattern_space_dim}")

    def visualize_fft(self, freqs, fft_vals):
        """Create enhanced FFT visualization"""
        vis_height = 400
        vis_width = 1000
        vis = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        
        # Scale FFT values
        fft_vals = fft_vals * self.fft_scale
        
        # Plot FFT
        max_freq = 50
        freq_mask = freqs <= max_freq
        freqs = freqs[freq_mask]
        fft_vals = fft_vals[freq_mask]
        
        if len(fft_vals) > 0:
            # Normalize and apply log scaling for better visualization
            fft_normalized = np.log1p(fft_vals)
            fft_normalized = fft_normalized / (np.max(fft_normalized) + 1e-10)
            
            # Draw frequency bars with color gradient
            for i in range(len(freqs)-1):
                x = int(freqs[i] * vis_width / max_freq)
                height = int(fft_normalized[i] * vis_height)
                
                # Color based on frequency band
                if freqs[i] <= self.low_freq:
                    color = (0, 0, 255)  # Red for low frequencies
                elif freqs[i] >= self.high_freq:
                    color = (255, 0, 0)  # Blue for high frequencies
                else:
                    color = (0, 255, 0)  # Green for passband
                    
                cv2.line(vis, (x, vis_height), (x, vis_height - height), color, 2)
            
            # Draw frequency band limits
            low_x = int(self.low_freq * vis_width / max_freq)
            high_x = int(self.high_freq * vis_width / max_freq)
            cv2.line(vis, (low_x, 0), (low_x, vis_height), (255, 255, 255), 1)
            cv2.line(vis, (high_x, 0), (high_x, vis_height), (255, 255, 255), 1)
        
        # Add labels and info
        cv2.putText(vis, f"Filter: {self.filter_type}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Low: {self.low_freq:.1f}Hz High: {self.high_freq:.1f}Hz", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis
    
    def save_logged_data(self):
        if not self.logged_data:
            print("No data to save.")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame(self.logged_data, columns=["Time", "Filtered Signal", "FFT Magnitude"])
        df.to_csv(f"{self.capture_dir}/{timestamp}_logged_data.csv", index=False)
        print(f"Logged data saved to {self.capture_dir}/{timestamp}_logged_data.csv")

    def quit(self):
        self.running = False
        self.root.destroy()

    def apply_frequency_filter(self, input_signal):
        """Apply configurable filter to signal"""
        nyq = 50.0  # Nyquist frequency
        
        if self.filter_type == 'bandpass':
            low = self.low_freq / nyq
            high = self.high_freq / nyq
            b, a = scipy.signal.butter(self.filter_order, [low, high], btype='band')
        elif self.filter_type == 'lowpass':
            cutoff = self.high_freq / nyq
            b, a = scipy.signal.butter(self.filter_order, cutoff, btype='low')
        else:  # highpass
            cutoff = self.low_freq / nyq
            b, a = scipy.signal.butter(self.filter_order, cutoff, btype='high')
        
        # Apply filter with forward-backward filtering
        filtered = scipy.signal.filtfilt(b, a, input_signal)
        
        # Apply resonance/Q factor
        if self.resonance_q > 1.0:
            sos = scipy.signal.butter(self.filter_order, [self.low_freq, self.high_freq], 
                              btype='band', fs=100, output='sos')
            filtered = scipy.signal.sosfilt(sos, filtered)
        
        # Mix original and filtered based on strength
        return input_signal * (1 - self.filter_strength) + filtered * self.filter_strength

    def compute_fft(self, signal):
        n = len(signal)
        fft_vals = fft(signal)
        freqs = fftfreq(n, d=1 / 256)
        pos_mask = freqs >= 0
        return freqs[pos_mask], np.abs(fft_vals[pos_mask])

    def compute_pattern_space(self, signal, dim=3, tau=10):
        """Compute pattern space embedding"""
        if len(signal) < dim * tau:
            return None
            
        N = len(signal) - (dim-1)*tau
        pattern_space = np.zeros((N, dim))
        
        for i in range(dim):
            pattern_space[:, i] = signal[i*tau:i*tau + N]
            
        return pattern_space

    def visualize_pattern_space(self, pattern_space):
        """Create pattern space visualization"""
        if pattern_space is None or pattern_space.shape[1] < 2:
            return np.zeros((800, 800, 3), dtype=np.uint8)
            
        vis = np.zeros((800, 800, 3), dtype=np.uint8)
        
        # Project to 2D and normalize
        x = pattern_space[:, 0]
        y = pattern_space[:, 1]
        x_norm = ((x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10) * 700 + 50).astype(int)
        y_norm = ((y - np.min(y)) / (np.max(y) - np.min(y) + 1e-10) * 700 + 50).astype(int)
        
        # Draw trajectory with color gradient
        points = list(zip(x_norm, y_norm))
        if len(points) > 1:
            for i in range(len(points)-1):
                progress = i / len(points)
                color = (
                    int(255 * progress),
                    int(255 * (1-progress)),
                    int(128 * np.sin(progress * np.pi))
                )
                cv2.line(vis, points[i], points[i+1], color, 1)
        
        # Add reference grid
        for i in range(50, 800, 100):
            cv2.line(vis, (i, 0), (i, 800), (40, 40, 40), 1)
            cv2.line(vis, (0, i), (800, i), (40, 40, 40), 1)
        
        # Add pattern space info
        info_text = f"Pattern Space"
        cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis

    def run(self):
        """Run the EEG analyzer."""
        signal_len = self.data.shape[1]
        i = 0
        while self.running and i < signal_len:
            self.step_size = int(self.step_var.get())
            self.delay = self.delay_var.get()  # Read delay from slider

            # Process signal
            eeg_signal = self.data[self.channel_idx, i : i + self.step_size]
            filtered_signal = self.apply_frequency_filter(eeg_signal)
            freqs, fft_vals = self.compute_fft(filtered_signal)
            pattern_space = self.compute_pattern_space(filtered_signal)

            # Log data
            self.logged_data.append([self.times[i], filtered_signal.mean(), fft_vals.sum()])

            # Visualize
            cv2.imshow("Filtered Signal", self.visualize_signal(filtered_signal, f"Channel {self.channel_idx}"))
            cv2.imshow("FFT Analysis", self.visualize_fft(freqs, fft_vals))
            cv2.imshow("Pattern Space", self.visualize_pattern_space(pattern_space))

            # Step forward
            i += self.step_size
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.quit()
            
            # Slow down loop based on delay
            time.sleep(self.delay)
            self.root.update()
        cv2.destroyAllWindows()

    def capture_screen(self):
        """Capture all visualization windows with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        windows = {
            'filtered': self.visualize_signal(self.data[self.channel_idx, :self.step_size], f"Channel {self.channel_idx}"),
            'fft': self.visualize_fft(*self.compute_fft(self.data[self.channel_idx, :self.step_size])),
            'pattern': self.visualize_pattern_space(
                self.compute_pattern_space(self.data[self.channel_idx, :self.step_size]))
        }

        for key, image in windows.items():
            if image is not None and image.size > 0:
                filename = f"{self.capture_dir}/{timestamp}_{key}.png"
                cv2.imwrite(filename, image)
                print(f"Captured {filename}")
            else:
                print(f"Warning: No content to capture for {key}")


    def visualize_signal(self, signal, title):
        vis = np.zeros((400, 1000, 3), dtype=np.uint8)
        signal_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-10)
        for i in range(len(signal_norm) - 1):
            x1, y1 = int(i * 1000 / len(signal_norm)), int((1 - signal_norm[i]) * 400)
            x2, y2 = int((i + 1) * 1000 / len(signal_norm)), int((1 - signal_norm[i + 1]) * 400)
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return vis

if __name__ == "__main__":
    edf_path = "EEG_C.edf"  # Replace with your EDF file path
    analyzer = AdvancedFrequencyAnalyzer(edf_path)
    analyzer.run()
