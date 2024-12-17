import numpy as np
import mne
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, entropy, wasserstein_distance
import logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ============================
# Configure Logging
# ============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================
# EEG Processing Class
# ============================
class EEGProcessor:
    def __init__(self, edf_path, channel_index=1):
        """
        Initializes the EEGProcessor.

        Parameters:
        - edf_path: Path to the EDF file.
        - channel_index: Index of the EEG channel to process (0-based).
        """
        self.edf_path = edf_path
        self.channel_index = channel_index
        self.data = None
        self.channel_name = None
        self.sampling_rate = None

    def load_eeg_channel(self):
        """Load EEG data from a specific channel."""
        try:
            raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
            channel_names = raw.ch_names
            if self.channel_index >= len(channel_names):
                raise IndexError("Channel index out of range.")
            self.channel_name = channel_names[self.channel_index]
            self.data = raw.get_data(picks=self.channel_index)[0]
            self.sampling_rate = raw.info['sfreq']
            logger.info(f"Loaded EEG channel '{self.channel_name}' with shape {self.data.shape}")
            logger.info(f"Sampling rate: {self.sampling_rate} Hz")
        except Exception as e:
            logger.error(f"Error loading EEG data: {e}")
            raise

    def preprocess(self):
        """Preprocess EEG data (e.g., filtering, normalization)."""
        if self.data is None:
            raise ValueError("EEG data not loaded.")

        try:
            raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
            raw.pick([self.channel_name])  # Updated to use 'pick' instead of 'pick_channels'
            
            # Determine Nyquist frequency
            nyq = self.sampling_rate / 2.
            logger.info(f"Nyquist frequency: {nyq} Hz")

            # Define bandpass filter frequencies
            low = 1.0  # 1 Hz
            high = 50.0  # 50 Hz

            # Adjust high if it exceeds Nyquist frequency
            if high >= nyq:
                high = nyq - 1.0  # Set high to Nyquist - 1 Hz as a safety margin
                logger.warning(f"Adjusted highpass frequency to {high} Hz to be below Nyquist ({nyq} Hz)")

            # Apply bandpass filter
            raw.filter(low, high, fir_design='firwin')
            self.data = raw.get_data()[0]
            logger.info(f"Applied bandpass filter: {low} - {high} Hz to '{self.channel_name}'")

            # Normalize the data
            scaler = StandardScaler()
            self.data = scaler.fit_transform(self.data.reshape(-1, 1)).flatten()
            logger.info(f"Normalized EEG data for channel '{self.channel_name}'")
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise

    def extract_features(self, window_size=256, overlap=128, target_dim=256, target_samples=10000):
        """
        Extract features from EEG data using sliding windows, with target sample count.
        
        Parameters:
        - window_size: Number of samples per window
        - overlap: Number of overlapping samples
        - target_dim: Target dimension to match AI latent space
        - target_samples: Target number of windows to match AI samples
        """
        if self.data is None:
            raise ValueError("EEG data not preprocessed.")

        # Calculate required step size to get target number of samples
        total_length = len(self.data) - window_size
        step = max(1, total_length // target_samples)
        
        features = []
        for start in range(0, total_length, step):
            if len(features) >= target_samples:
                break
                
            end = start + window_size
            window = self.data[start:end]

            # Compute PSD with interpolation to target dimension
            freqs, psd = welch(window, fs=self.sampling_rate, nperseg=window_size)
            if len(psd) != target_dim:
                x_original = np.linspace(0, 1, len(psd))
                x_new = np.linspace(0, 1, target_dim)
                psd = np.interp(x_new, x_original, psd)
                
            features.append(psd)

        features = np.array(features)
        logger.info(f"Extracted {features.shape[0]} feature windows from EEG data")
        return features[:target_samples]  # Ensure exact number of samples
# ============================
# AI Latent Space Loader
# ============================
def load_ai_latent_space(ai_latent_path='cifar10_latents.npy'):
    """
    Load AI latent space data from a .npy file.

    Parameters:
    - ai_latent_path: Path to the AI latent .npy file.

    Returns:
    - ai_latent_data: NumPy array of AI latent vectors.
    """
    try:
        ai_latent_data = np.load(ai_latent_path)
        logger.info(f"Loaded AI latent data from '{ai_latent_path}' with shape {ai_latent_data.shape}")
        return ai_latent_data
    except Exception as e:
        logger.error(f"Error loading AI latent space: {e}")
        raise

# ============================
# Dimensionality Reduction
# ============================
def reduce_dimensionality(ai_latent, eeg_features, n_components=50):
    """
    Apply PCA to reduce dimensionality of both AI and EEG features.

    Parameters:
    - ai_latent: AI latent vectors.
    - eeg_features: EEG feature vectors.
    - n_components: Number of PCA components.

    Returns:
    - ai_reduced: Reduced AI latent vectors.
    - eeg_reduced: Reduced EEG feature vectors.
    """
    scaler = StandardScaler()
    combined_data = np.vstack((ai_latent, eeg_features))
    scaler.fit(combined_data)
    logger.info("Fitted StandardScaler on combined AI and EEG data.")

    ai_scaled = scaler.transform(ai_latent)
    eeg_scaled = scaler.transform(eeg_features)

    pca = PCA(n_components=n_components)
    pca.fit(combined_data)
    logger.info(f"Fitted PCA with {n_components} components on combined data.")

    ai_reduced = pca.transform(ai_scaled)
    eeg_reduced = pca.transform(eeg_scaled)
    logger.info(f"Reduced AI latent space to {n_components} dimensions.")
    logger.info(f"Reduced EEG feature space to {n_components} dimensions.")

    return ai_reduced, eeg_reduced

# ============================
# Canonical Correlation Analysis
# ============================
def perform_cca(ai_reduced, eeg_reduced, n_components=10):
    """
    Perform Canonical Correlation Analysis between AI and EEG data.

    Parameters:
    - ai_reduced: Reduced AI latent vectors.
    - eeg_reduced: Reduced EEG feature vectors.
    - n_components: Number of CCA components.

    Returns:
    - cca: Trained CCA model.
    - ai_c: Transformed AI data.
    - eeg_c: Transformed EEG data.
    - correlations: List of Pearson correlation coefficients per component.
    """
    cca = CCA(n_components=n_components)
    cca.fit(ai_reduced, eeg_reduced)
    ai_c, eeg_c = cca.transform(ai_reduced, eeg_reduced)

    correlations = []
    for i in range(n_components):
        corr, _ = pearsonr(ai_c[:, i], eeg_c[:, i])
        correlations.append(corr)
        logger.info(f"CCA Component {i+1}: Correlation = {corr:.3f}")

    return cca, ai_c, eeg_c, correlations

# ============================
# Cosine Similarity Computation
# ============================
def compute_cosine_similarity(ai_reduced, eeg_reduced):
    """
    Compute cosine similarity between AI and EEG latent spaces.

    Parameters:
    - ai_reduced: Reduced AI latent vectors.
    - eeg_reduced: Reduced EEG feature vectors.

    Returns:
    - similarity_matrix: Cosine similarity matrix.
    - average_max_similarity: Average of maximum similarities per AI vector.
    """
    # Normalize the data
    ai_norm = ai_reduced / np.linalg.norm(ai_reduced, axis=1, keepdims=True)
    eeg_norm = eeg_reduced / np.linalg.norm(eeg_reduced, axis=1, keepdims=True)
    logger.info("Normalized AI and EEG data for cosine similarity.")

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(ai_norm, eeg_norm)
    logger.info("Computed cosine similarity matrix.")

    # Compute the average maximum similarity for each AI latent vector
    max_similarities = np.max(similarity_matrix, axis=1)
    average_max_similarity = np.mean(max_similarities)
    logger.info(f"Average Maximum Cosine Similarity: {average_max_similarity:.3f}")

    return similarity_matrix, average_max_similarity

# ============================
# Earth Mover's Distance (EMD) Computation
# ============================
def compute_emd(ai_reduced, eeg_reduced):
    """
    Compute Earth Mover's Distance between two distributions.

    Parameters:
    - ai_reduced: Reduced AI latent vectors.
    - eeg_reduced: Reduced EEG feature vectors.

    Returns:
    - emd_scores: List of EMD scores per dimension.
    - average_emd: Average EMD across dimensions.
    """
    emd_scores = []
    n_dimensions = ai_reduced.shape[1]
    for i in range(n_dimensions):
        emd = wasserstein_distance(ai_reduced[:, i], eeg_reduced[:, i])
        emd_scores.append(emd)
        logger.info(f"EMD for Dimension {i+1}: {emd:.3f}")
    average_emd = np.mean(emd_scores)
    logger.info(f"Average EMD across dimensions: {average_emd:.3f}")
    return emd_scores, average_emd

# ============================
# Visualization of Latent Spaces
# ============================
def visualize_latent_spaces(ai_reduced, eeg_reduced, labels=None):
    """
    Visualize AI and EEG latent spaces using t-SNE.

    Parameters:
    - ai_reduced: Reduced AI latent vectors.
    - eeg_reduced: Reduced EEG feature vectors.
    - labels: Optional labels for coloring (e.g., class labels).

    Returns:
    - None
    """
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    logger.info("Starting t-SNE dimensionality reduction.")

    ai_tsne = tsne.fit_transform(ai_reduced)
    eeg_tsne = tsne.fit_transform(eeg_reduced)
    logger.info("Completed t-SNE transformation.")

    plt.figure(figsize=(12, 6))

    # AI Latent Space
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(ai_tsne[:, 0], ai_tsne[:, 1], c=labels, cmap='tab10', s=5, alpha=0.6)
    plt.title('AI Latent Space (t-SNE)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    if labels is not None:
        plt.colorbar(scatter, ticks=range(10), label='Class')

    # EEG Feature Space
    plt.subplot(1, 2, 2)
    plt.scatter(eeg_tsne[:, 0], eeg_tsne[:, 1], c=labels, cmap='tab10', s=5, alpha=0.6)
    plt.title('EEG Feature Space (t-SNE)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    if labels is not None:
        plt.colorbar(scatter, ticks=range(10), label='Class')

    plt.tight_layout()
    plt.show()
    logger.info("Displayed t-SNE visualizations.")

# ============================
# Complete Comparison Function
# ============================
def main_comparison(eeg_edf_path, ai_latent_path='cifar10_latents.npy'):
    """
    Perform the complete comparison between AI latent space and EEG Channel 1 data.

    Parameters:
    - eeg_edf_path: Path to the EEG EDF file.
    - ai_latent_path: Path to the AI latent .npy file.

    Returns:
    - summary: Dictionary containing comparison metrics.
    """
    # Step 1: Load AI latent space
    ai_latent = load_ai_latent_space(ai_latent_path)

    # Step 2: Load and preprocess EEG data
    eeg_processor = EEGProcessor(edf_path=eeg_edf_path, channel_index=1)
    eeg_processor.load_eeg_channel()
    eeg_processor.preprocess()
    eeg_features = eeg_processor.extract_features()

    # Step 3: Dimensionality Reduction
    ai_reduced, eeg_reduced = reduce_dimensionality(ai_latent, eeg_features, n_components=50)

    # Step 4: Similarity Analysis
    cca_model, ai_c, eeg_c, cca_correlations = perform_cca(ai_reduced, eeg_reduced, n_components=10)
    similarity_matrix, avg_max_cos_sim = compute_cosine_similarity(ai_reduced, eeg_reduced)
    emd_scores, average_emd = compute_emd(ai_reduced, eeg_reduced)

    # Step 5: Visualization
    visualize_latent_spaces(ai_reduced, eeg_reduced, labels=None)  # Pass labels if available

    # Step 6: Summary of Results
    summary = {
        'CCA Correlations': cca_correlations,
        'Average Cosine Similarity': avg_max_cos_sim,
        'EMD Scores': emd_scores,
        'Average EMD': average_emd
    }

    logger.info(f"Comparison Summary: {summary}")
    return summary

# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    # Define paths
    eeg_edf_path = 'G:/DocsHouse/98 LiveAnalyze/EEG_C.edf'  # Replace with your actual EEG EDF file path
    ai_latent_path = 'cifar10_latents.npy'  # Ensure this file exists in the specified path

    # Run comparison
    try:
        results = main_comparison(eeg_edf_path, ai_latent_path)
        logger.info(f"Final Comparison Results: {results}")
    except Exception as e:
        logger.error(f"An error occurred during comparison: {e}")
