import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 64

# -------------------------
# Define VAE Decoder
# -------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), 128, 4, 4)
        x_recon = self.deconv(z)
        return x_recon

decoder = Decoder(latent_dim).to(device)
decoder.load_state_dict(torch.load("vae_decoder.pth", map_location=device))
decoder.eval()

# Load CIFAR latents and EEG projection
cifar_latents = np.load("cifar10_latents.npy")    # (Nc, D=256)
eeg_projection = np.load("eeg_projection.npy")     # (Ne, 1)

print("CIFAR latents:", cifar_latents.shape)
print("EEG projection:", eeg_projection.shape)

# We need at least a vector of length 64 for fractal dimension or something similar.
# Let's replicate EEG dimension if it's 1D.
if eeg_projection.shape[1] == 1:
    # replicate to match latent_dim for fractal analysis as well
    eeg_projection = np.tile(eeg_projection, (1, latent_dim))
    print("Replicated EEG projection to shape:", eeg_projection.shape)

# -------------------------
# Katz Fractal Dimension (example)
# This is a simplistic approach: Katz's method for a single vector x:
# Katz FD = log10(L) / (log10(L) + log10(d))
# Where L is the total length (sum of point-to-point distances)
# and d is the maximum distance from the first point.
# -------------------------
def katz_fractal_dimension(x):
    # x is a 1D array representing the signal
    # Compute cumulative length
    # We'll interpret each dimension as a point on a line, so distance = sum of abs differences
    distances = np.abs(np.diff(x))
    L = np.sum(distances) + 1e-9  # total length
    # max distance from first point:
    d = np.max(np.abs(x - x[0])) + 1e-9
    return np.log10(L) / (np.log10(L) + np.log10(d))

# Compute fractal dimension for each EEG vector
fractal_dims = []
for i in range(eeg_projection.shape[0]):
    fd = katz_fractal_dimension(eeg_projection[i])
    fractal_dims.append(fd)

fractal_dims = np.array(fractal_dims)
# Sort EEG by fractal dimension
sorted_indices = np.argsort(fractal_dims)
eeg_projection_sorted = eeg_projection[sorted_indices]

# Adjust CIFAR latents dimension if needed
if cifar_latents.shape[1] > latent_dim:
    cifar_latents = cifar_latents[:, :latent_dim]
elif cifar_latents.shape[1] < latent_dim:
    # If needed, replicate CIFAR dimensions (unlikely needed)
    rep_factor = latent_dim // cifar_latents.shape[1]
    extended = np.tile(cifar_latents, (1, rep_factor))
    if extended.shape[1] > latent_dim:
        extended = extended[:, :latent_dim]
    cifar_latents = extended

print("Final CIFAR latents shape:", cifar_latents.shape)
print("Final EEG shape:", eeg_projection_sorted.shape)

# Compute similarity
sim_matrix = cosine_similarity(eeg_projection_sorted, cifar_latents)
best_matches = np.argmax(sim_matrix, axis=1)

# Decode some samples
sample_count = min(16, len(best_matches))
matched_cifar_vectors = cifar_latents[best_matches[:sample_count]]

matched_cifar_t = torch.tensor(matched_cifar_vectors, dtype=torch.float32, device=device)
with torch.no_grad():
    reconstructed_images = decoder(matched_cifar_t)
    save_image(reconstructed_images.cpu(), "eeg_generated_images_fractal_similarity.png", nrow=4)
    print("Images generated from fractal-sorted EEG latents (matched by similarity) saved as eeg_generated_images_fractal_similarity.png.")

print("Process completed. This approach uses fractal dimension sorting as a heuristic step before alignment.")
