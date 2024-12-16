
# Antti's Analysis Tools

Antti's Analysis Tools is a suite of Python scripts designed for **advanced signal analysis** of video streams and EEG (Electroencephalography) files. These tools provide filtering, FFT (Fast Fourier Transform), and visualization capabilities for time series signals.

---

## Overview

The project includes two core tools:

1. **Video Analysis Tool**  
   Processes and analyzes real-time video input (e.g., from a webcam or a video file) for signal analysis and visualization.  
   Key features include:
   - Signal extraction from video frames  
   - FFT-based frequency analysis  
   - Filtered signal visualization  
   - Dynamic pattern space generation  
   - Customizable speed and step size controls  

2. **EDF EEG Analysis Tool**  
   Processes EEG data from EDF (European Data Format) files for frequency and signal analysis.  
   Key features include:
   - EDF file loading and channel selection  
   - Bandpass filtering and FFT analysis  
   - Interactive visualization of EEG signals  
   - Pattern space embedding for signal dynamics  
   - Screen capturing and data saving for further analysis  

---

## Dependencies

Both tools require the following Python libraries:

### Shared Dependencies
- `cv2` (OpenCV) - For video and image processing  
- `numpy` - For numerical operations  
- `scipy` - For signal processing and filtering  
- `tkinter` - For graphical user interface (GUI) controls  
- `matplotlib` - For plotting and visualizations  
- `os`, `datetime` - For file and time management  

### Video Analysis Tool Specific
- `pyautogui` - For capturing OpenCV window screenshots  
- `scipy.stats` - For additional spectral features  
- `sklearn.preprocessing` - For normalizing data  

### EDF EEG Analysis Tool Specific
- `mne` - For loading and processing EEG data from EDF files  
- `pandas` - For logging and saving analyzed data  

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/anttiluode/Anttis-Analysis-Tools.git
   cd Anttis-Analysis-Tools
   ```

2. **Install the required libraries:**
   ```bash
   pip install opencv-python numpy scipy matplotlib pyautogui mne pandas scikit-learn
   ```

---

## How to Run

### 1. Video Analysis Tool
The video analysis tool processes real-time video input (e.g., webcam).

- **Run the script:**
   ```bash
   python AnalyzeVideo.py
   ```

- **Controls:**
   - Press **`q`** to quit the analysis.  
   - Press **`s`** to capture the current visualization.  
   - Adjust filters, speed, and step size in the GUI.

---

### 2. EDF EEG Analysis Tool
The EDF EEG analysis tool processes EEG data from `.edf` files.

- **Run the script:**
   ```bash
   python AnalyzeEDF.py
   ```

- **Instructions:**
   - Load an `.edf` EEG file and choose a channel for analysis.
   - Use the GUI to adjust filtering, delay, and step sizes.
   - Save visualizations and data using the available buttons.

---

## Features

| Feature                        | Video Analysis Tool | EDF EEG Analysis Tool |
|--------------------------------|---------------------|-----------------------|
| Signal Filtering               | ✅                  | ✅                    |
| Fast Fourier Transform (FFT)   | ✅                  | ✅                    |
| Dynamic Pattern Space          | ✅                  | ✅                    |
| Adjustable Speed & Steps       | ✅                  | ✅                    |
| EDF File Loading               | ❌                  | ✅                    |
| Real-Time Video Analysis       | ✅                  | ❌                    |
| Screen Capture                 | ✅                  | ✅                    |
| Data Saving                    | ✅                  | ✅                    |

---

## File Structure

```
Anttis-Analysis-Tools/
│
├── edf_analysis.py      # EDF EEG Analysis Tool
├── video_analysis.py    # Video Analysis Tool
├── screen_captures/     # Folder for saved screenshots
├── data/                # Folder for saved analysis data (optional)
└── README.md            # Documentation
```

---

## Contribution

Feel free to open issues, submit feature requests, or contribute improvements to the codebase through pull requests.

---

## License

This project is licensed under the MIT License.  

---

## Acknowledgements

Thanks to the developers of **MNE**, **OpenCV**, **SciPy**, and other open-source tools that made this project possible.
