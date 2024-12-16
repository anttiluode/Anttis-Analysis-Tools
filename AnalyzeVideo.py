
import cv2
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq, fft2, fftshift
import tkinter as tk
import pyautogui
from tkinter import ttk
import scipy
from collections import deque
from datetime import datetime
import os
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.preprocessing import normalize

class AdvancedFrequencyAnalyzer:
    def __init__(self):
        self.history_length = 100
        self.running = True
        self.pattern_history = deque(maxlen=self.history_length)
        
        # Create captures directory
        self.capture_dir = "screen_captures"
        os.makedirs(self.capture_dir, exist_ok=True)
        
        # Analysis parameters
        self.low_freq = 0.5
        self.high_freq = 30
        self.filter_strength = 0.5
        self.fft_scale = 1.0
        self.resonance_q = 1.0  # Resonance/Q factor
        self.filter_order = 4
        self.filter_type = 'bandpass'  # 'lowpass', 'highpass', 'bandpass'
        
        # Advanced parameters
        self.fractal_dimension = 0
        self.phase_coherence = 0
        self.spectral_edge = 0
        
        # Create windows
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('FFT Analysis', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Filtered Signal', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Pattern Space', cv2.WINDOW_NORMAL)
        
        cv2.resizeWindow('FFT Analysis', 1000, 400)
        cv2.resizeWindow('Filtered Signal', 1000, 400)
        cv2.resizeWindow('Pattern Space', 800, 800)
        
        self.init_gui()


    def init_gui(self):
        self.root = tk.Tk()
        self.root.title("Advanced Frequency Analysis")
        
        # Frequency controls frame
        freq_frame = ttk.LabelFrame(self.root, text="Frequency Controls")
        freq_frame.pack(fill="x", padx=5, pady=5)
        
        # Low frequency control with entry
        low_freq_frame = ttk.Frame(freq_frame)
        low_freq_frame.pack(fill="x", padx=5)
        ttk.Label(low_freq_frame, text="Low Cutoff (Hz):").pack(side="left")
        self.low_freq_entry = ttk.Entry(low_freq_frame, width=8)
        self.low_freq_entry.insert(0, str(self.low_freq))
        self.low_freq_entry.pack(side="left", padx=5)
        self.low_freq_slider = ttk.Scale(
            freq_frame, from_=0.1, to=50,
            value=self.low_freq, orient='horizontal',
            command=self.update_low_freq
        )
        self.low_freq_slider.pack(fill="x", padx=5)

        # Speed Control Frame
        speed_frame = ttk.LabelFrame(self.root, text="Speed and Steps")
        speed_frame.pack(fill="x", padx=5, pady=5)

        # Step size slider
        ttk.Label(speed_frame, text="Step Size (Samples):").pack(side="left")
        self.step_var = tk.IntVar(value=1)  # Default step size
        self.step_slider = ttk.Scale(
            speed_frame, from_=1, to=50, orient="horizontal", variable=self.step_var
        )
        self.step_slider.pack(side="left", fill="x", expand=True, padx=5)

        # Delay slider
        ttk.Label(speed_frame, text="Delay (s):").pack(side="left")
        self.delay_var = tk.DoubleVar(value=0.03)  # Default delay in seconds
        self.delay_slider = ttk.Scale(
            speed_frame, from_=0.01, to=1.0, orient="horizontal", variable=self.delay_var
        )
        self.delay_slider.pack(side="left", fill="x", expand=True, padx=5)


        # High frequency control with entry
        high_freq_frame = ttk.Frame(freq_frame)
        high_freq_frame.pack(fill="x", padx=5)
        ttk.Label(high_freq_frame, text="High Cutoff (Hz):").pack(side="left")
        self.high_freq_entry = ttk.Entry(high_freq_frame, width=8)
        self.high_freq_entry.insert(0, str(self.high_freq))
        self.high_freq_entry.pack(side="left", padx=5)
        self.high_freq_slider = ttk.Scale(
            freq_frame, from_=0.1, to=50,
            value=self.high_freq, orient='horizontal',
            command=self.update_high_freq
        )
        self.high_freq_slider.pack(fill="x", padx=5)
        
        # Filter controls frame
        filter_frame = ttk.LabelFrame(self.root, text="Filter Controls")
        filter_frame.pack(fill="x", padx=5, pady=5)
        
        # Filter type selection
        ttk.Label(filter_frame, text="Filter Type:").pack()
        self.filter_type_var = tk.StringVar(value=self.filter_type)
        for ftype in ['lowpass', 'highpass', 'bandpass']:
            ttk.Radiobutton(
                filter_frame, text=ftype.title(),
                value=ftype, variable=self.filter_type_var,
                command=lambda: setattr(self, 'filter_type', self.filter_type_var.get())
            ).pack()
        
        # Filter strength control
        ttk.Label(filter_frame, text="Filter Strength:").pack()
        self.filter_slider = ttk.Scale(
            filter_frame, from_=0, to=1,
            value=self.filter_strength, orient='horizontal',
            command=lambda v: setattr(self, 'filter_strength', float(v))
        )
        self.filter_slider.pack(fill="x", padx=5)
        
        # Resonance control
        ttk.Label(filter_frame, text="Resonance (Q):").pack()
        self.resonance_slider = ttk.Scale(
            filter_frame, from_=0.1, to=10,
            value=self.resonance_q, orient='horizontal',
            command=lambda v: setattr(self, 'resonance_q', float(v))
        )
        self.resonance_slider.pack(fill="x", padx=5)
        
        # FFT scale control
        ttk.Label(filter_frame, text="FFT Scale:").pack()
        self.scale_slider = ttk.Scale(
            filter_frame, from_=0.1, to=5,
            value=self.fft_scale, orient='horizontal',
            command=lambda v: setattr(self, 'fft_scale', float(v))
        )
        self.scale_slider.pack(fill="x", padx=5)
        
        # Capture button
        ttk.Button(self.root, text="Capture Screens", 
                  command=self.capture_screen).pack(pady=10)

    def update_low_freq(self, value):
        """Update low frequency from slider and validate"""
        try:
            freq = float(value)
            if freq < float(self.high_freq_slider.get()):
                self.low_freq = freq
                self.low_freq_entry.delete(0, tk.END)
                self.low_freq_entry.insert(0, f"{freq:.1f}")
        except ValueError:
            pass

    def update_high_freq(self, value):
        """Update high frequency from slider and validate"""
        try:
            freq = float(value)
            if freq > float(self.low_freq_slider.get()):
                self.high_freq = freq
                self.high_freq_entry.delete(0, tk.END)
                self.high_freq_entry.insert(0, f"{freq:.1f}")
        except ValueError:
            pass

    def extract_signal(self, frame):
        """Extract signal from frame with preprocessing"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        signal = np.mean(gray, axis=0)
        signal = signal - np.mean(signal)  # Remove DC component
        
        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(len(signal))
        signal = signal * window
        
        return signal

    def compute_fft(self, signal):
        """Compute FFT with zero padding for better frequency resolution"""
        n = len(signal)
        n_padded = 2**int(np.ceil(np.log2(n)))  # Next power of 2
        signal_padded = np.pad(signal, (0, n_padded - n))
        
        fft_vals = fft(signal_padded)
        freqs = fftfreq(n_padded, 1/100)  # Assuming 100Hz sampling
        
        # Get positive frequencies
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        fft_vals = np.abs(fft_vals[pos_mask])
        
        return freqs, fft_vals

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

    def visualize_signal(self, signal, title):
        """Create enhanced signal visualization"""
        vis_height = 400
        vis_width = 1000
        vis = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        
        # Normalize signal
        signal_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-10)
        
        # Create points with history fade effect
        points = []
        for i in range(len(signal_norm)):
            x = int(i * vis_width / len(signal_norm))
            y = int((1 - signal_norm[i]) * vis_height)
            points.append((x, y))
        
        # Draw lines with color gradient
        if len(points) > 1:
            for i in range(len(points)-1):
                progress = i / len(points)
                color = (
                    int(255 * (1-progress)),  # Blue fade
                    int(255 * progress),      # Green increase
                    0
                )
                cv2.line(vis, points[i], points[i+1], color, 1)
        
        # Add title and info
        cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis

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

    def capture_screen(self):
        """Capture all visualization windows with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        windows = {
            'original': 'Original',
            'fft': 'FFT Analysis',
            'filtered': 'Filtered Signal',
            'pattern': 'Pattern Space'
        }

        for key, window_name in windows.items():
            try:
                # Check if window is visible
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                    # Get window position and size
                    x, y, w, h = cv2.getWindowImageRect(window_name)
                    
                    # Capture the region using pyautogui
                    screenshot = pyautogui.screenshot(region=(x, y, w, h))
                    
                    # Convert the screenshot to OpenCV format
                    screenshot = np.array(screenshot)
                    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                    
                    # Save the screenshot
                    filename = f"{self.capture_dir}/{timestamp}_{key}.png"
                    cv2.imwrite(filename, screenshot)
                    print(f"Captured {filename}")
            except Exception as e:
                print(f"Error capturing {window_name}: {e}")

    def compute_spectral_features(self, input_signal):
        """Compute additional spectral features"""
        freqs, psd = signal.welch(input_signal, fs=100, nperseg=min(256, len(input_signal)))
        
        # Spectral edge frequency (95% of power)
        total_power = np.sum(psd)
        cumulative_power = np.cumsum(psd)
        spectral_edge = freqs[np.where(cumulative_power >= 0.95 * total_power)[0][0]]
        
        # Spectral entropy
        psd_norm = psd / total_power
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        
        return spectral_edge, spectral_entropy
    
    def run(self, camera_index=0):
        """Main run loop"""
        cap = cv2.VideoCapture(camera_index)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every Nth frame based on step size
            step_size = int(self.step_var.get())  # Get step size from slider
            for _ in range(step_size):  # Skip frames based on step size
                cap.grab()
            
            # Extract and process signal
            signal = self.extract_signal(frame)
            filtered_signal = self.apply_frequency_filter(signal)
            freqs, fft_vals = self.compute_fft(filtered_signal)
            
            # Compute pattern space
            pattern_space = self.compute_pattern_space(filtered_signal)
            
            # Compute additional features
            spectral_edge, spectral_entropy = self.compute_spectral_features(filtered_signal)
            
            # Create visualizations
            cv2.imshow('Original', frame)
            cv2.imshow('FFT Analysis', self.visualize_fft(freqs, fft_vals))
            cv2.imshow('Filtered Signal', self.visualize_signal(filtered_signal,
                    f"Filtered Signal (Edge: {spectral_edge:.1f}Hz, Entropy: {spectral_entropy:.2f})"))
            cv2.imshow('Pattern Space', self.visualize_pattern_space(pattern_space))
            
            # Handle controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.capture_screen()
            
            # Add delay to control speed
            delay = self.delay_var.get()
            cv2.waitKey(int(delay * 1000))  # Convert to milliseconds
            
            # Update GUI
            try:
                self.root.update()
            except tk.TclError:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.root.quit()


if __name__ == "__main__":
    analyzer = AdvancedFrequencyAnalyzer()
    analyzer.run(camera_index=0)  # Adjust camera index as needed
