import numpy as np
import mne
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from scipy.signal import welch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGProcessor:
    def __init__(self, edf_path, channel_index=1, low_freq=1.0, high_freq=50.0, window_size=256, overlap=128, target_dim=256, target_samples=10000):
        """
        Parameters:
        - edf_path: Path to the EEG EDF file
        - channel_index: which channel to pick from the EDF (0-based)
        - low_freq, high_freq: bandpass filter frequencies
        - window_size, overlap: parameters for PSD feature extraction
        - target_dim: number of PSD points to interpolate to match AI latent dimension
        - target_samples: how many EEG windows to produce
        """
        self.edf_path = edf_path
        self.channel_index = channel_index
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.window_size = window_size
        self.overlap = overlap
        self.target_dim = target_dim
        self.target_samples = target_samples
        self.data = None
        self.sampling_rate = None
        self.channel_name = None

    def load_eeg_channel(self):
        raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
        if self.channel_index >= len(raw.ch_names):
            raise IndexError("Channel index out of range.")
        self.channel_name = raw.ch_names[self.channel_index]
        self.data = raw.get_data(picks=self.channel_index)[0]
        self.sampling_rate = raw.info['sfreq']
        logger.info(f"Loaded EEG channel '{self.channel_name}' with shape {self.data.shape}")
        logger.info(f"Sampling rate: {self.sampling_rate} Hz")

    def preprocess(self):
        if self.data is None:
            raise ValueError("EEG data not loaded.")
        raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
        raw.pick([self.channel_name])
        nyq = self.sampling_rate / 2.0
        high = min(self.high_freq, nyq - 1.0)  # ensure filter validity
        if high < self.high_freq:
            logger.warning(f"Adjusted high frequency to {high} Hz (Nyquist limit)")
        raw.filter(self.low_freq, high, fir_design='firwin')
        filtered_data = raw.get_data()[0]

        # Normalize
        filtered_data = (filtered_data - np.mean(filtered_data)) / (np.std(filtered_data) + 1e-10)
        self.data = filtered_data
        logger.info(f"Preprocessed EEG data for channel '{self.channel_name}'.")

    def extract_features(self):
        """
        Extract PSD features from EEG data using sliding windows to produce target_samples windows.
        Interpolate each PSD to target_dim points to match AI latent dimensions.
        """
        if self.data is None:
            raise ValueError("EEG data not preprocessed.")

        total_length = len(self.data) - self.window_size
        if total_length <= 0:
            raise ValueError("EEG data too short compared to window_size.")

        step = max(1, total_length // self.target_samples)
        features = []
        for start in range(0, total_length, step):
            if len(features) >= self.target_samples:
                break
            end = start + self.window_size
            if end > len(self.data):
                break
            window = self.data[start:end]

            # Compute PSD
            freqs, psd = welch(window, fs=self.sampling_rate, nperseg=self.window_size)
            # Interpolate PSD to target_dim
            if len(psd) != self.target_dim:
                x_original = np.linspace(0, 1, len(psd))
                x_new = np.linspace(0, 1, self.target_dim)
                psd = np.interp(x_new, x_original, psd)
            features.append(psd)

        features = np.array(features)
        logger.info(f"Extracted {features.shape[0]} feature windows from EEG data.")
        return features


def load_ai_latent_space(ai_latent_path):
    ai_latent_data = np.load(ai_latent_path)
    logger.info(f"Loaded AI latent data from '{ai_latent_path}' with shape {ai_latent_data.shape}")
    return ai_latent_data


def run_analysis(eeg_edf_path, ai_latent_path, output_eeg_projection='eeg_projection.npy', top_components=1):
    """
    - Load AI latent space
    - Load and preprocess EEG
    - Extract EEG features
    - Perform PCA -> CCA
    - Identify top correlated components
    - Project EEG onto these components
    - Save the projected EEG data as a numpy array
    """

    # Step 1: Load AI latent space
    ai_latent = load_ai_latent_space(ai_latent_path)

    # Step 2: Load and preprocess EEG
    eeg_processor = EEGProcessor(edf_path=eeg_edf_path, channel_index=1)
    eeg_processor.load_eeg_channel()
    eeg_processor.preprocess()
    eeg_features = eeg_processor.extract_features()

    # Ensure same number of samples if needed (already done by extract_features)
    min_samples = min(ai_latent.shape[0], eeg_features.shape[0])
    ai_latent = ai_latent[:min_samples]
    eeg_features = eeg_features[:min_samples]

    # Step 3: Dimensionality Reduction with PCA
    # Combine data to ensure same PCA transform
    combined = np.vstack((ai_latent, eeg_features))
    scaler = StandardScaler().fit(combined)
    ai_scaled = scaler.transform(ai_latent)
    eeg_scaled = scaler.transform(eeg_features)

    # Decide on PCA components (just enough as original code)
    n_components = min(50, ai_scaled.shape[1])  # or some chosen number
    pca = PCA(n_components=n_components)
    pca.fit(combined)
    ai_reduced = pca.transform(ai_scaled)
    eeg_reduced = pca.transform(eeg_scaled)

    logger.info("PCA reduction done on AI and EEG data.")

    # Step 4: Perform CCA
    cca_components = min(10, n_components)  # Let's do 10 CCA components
    cca = CCA(n_components=cca_components)
    cca.fit(ai_reduced, eeg_reduced)
    ai_c, eeg_c = cca.transform(ai_reduced, eeg_reduced)

    # Compute correlations
    correlations = []
    from scipy.stats import pearsonr
    for i in range(cca_components):
        corr, _ = pearsonr(ai_c[:, i], eeg_c[:, i])
        correlations.append(corr)
        logger.info(f"CCA Component {i+1}: Correlation = {corr:.3f}")

    # Step 5: Identify top correlated CCA components
    # Sort by absolute correlation
    sorted_indices = np.argsort(np.abs(correlations))[::-1]
    top_idx = sorted_indices[:top_components]

    logger.info(f"Selected top {top_components} CCA components: {top_idx + 1} with correlations {[correlations[i] for i in top_idx]}")

    # Step 6: Project EEG onto these top CCA components
    # eeg_c are the EEG data transformed into CCA space, top_idx picks the desired components
    eeg_projection = eeg_c[:, top_idx]

    # Step 7: Save the projected EEG data as a numpy array
    # This array now represents the part of the EEG latent space most aligned with the AI latent space
    np.save(output_eeg_projection, eeg_projection)
    logger.info(f"Saved projected EEG data into {output_eeg_projection}")

    return eeg_projection, correlations


if __name__ == "__main__":
    # Example usage:
    # Adjust these paths to your actual file locations.
    eeg_edf_path = "EEG_c.edf"
    ai_latent_path = "cifar10_latents.npy"

    # This code will produce an eeg_projection.npy which contains the EEG data projected onto
    # the most correlated dimensions with AI latent space.
    eeg_projection, correlations = run_analysis(eeg_edf_path, ai_latent_path)
    # Now eeg_projection is a numpy array focusing on the overlapping "bit" of the EEG and AI latent representations.
