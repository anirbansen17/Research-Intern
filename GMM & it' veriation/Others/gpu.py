import torch
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist

# Check if GPU is available and use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the features from a .pt file stored locally
features_path = 'imbalanced_train_features.pt'
features = torch.load(features_path).to(device)

# Perform FCM
n_clusters = 10
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(features.cpu().numpy().T, n_clusters, 2, error=0.005, maxiter=1000, init=None)

# Transfer centroids back to GPU
cntr = torch.tensor(cntr, device=device)

# Print centroids
print("Centroids:\n", cntr.cpu().numpy())

# Plot the data and centroids (Note: Matplotlib can't directly plot tensors on GPU)
plt.scatter(features.cpu().numpy()[:, 0], features.cpu().numpy()[:, 1], c='blue', marker='o', alpha=0.5)
plt.scatter(cntr.cpu().numpy()[:, 0], cntr.cpu().numpy()[:, 1], c='red', marker='x')
plt.title('Fuzzy C-Means Clustering')
plt.show()

# Initialize GMM Parameters
def initialize_gmm_parameters(data, cntr, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, means_init=cntr.cpu().numpy())
    gmm.fit(data.cpu().numpy())
    return gmm

gmm = initialize_gmm_parameters(features, cntr, n_clusters)
mu = torch.tensor(gmm.means_, device=device)
sigma = torch.tensor(gmm.covariances_, device=device)

print("Initial GMM parameters:")
print("Means:\n", mu.cpu().numpy())
print("Covariances:\n", sigma.cpu().numpy())

# Select all the 9 minority components (those with lower mixing coefficients)
mixing_coefficients = torch.tensor(gmm.weights_, device=device)
minority_indices = torch.argsort(mixing_coefficients)[:9].cpu().numpy()  
print("Minority components indices:", minority_indices)

# Apply GMM for each minority cluster
def apply_gmm_to_minority(data, majority_data, minority_indices):
    gmm_params = {}
    for index in minority_indices:
        minority_data = data[torch.argmax(u, axis=0) == index]
        if len(minority_data) < 2:  # Check if there are at least two samples
            continue
        m_comp = len(majority_data) / len(minority_data)
        n_components = max(1, int(np.round(m_comp)))  # Ensure n_components is at least 1
        gmm = GaussianMixture(n_components=min(n_components, len(minority_data)), init_params='kmeans')
        gmm.fit(minority_data.cpu().numpy())
        gmm_params[index] = (torch.tensor(gmm.means_, device=device), torch.tensor(gmm.covariances_, device=device), m_comp)
    return gmm_params

majority_class_indices = torch.argmax(u, axis=0) == 0
majority_data = features[majority_class_indices]
gmm_params = apply_gmm_to_minority(features, majority_data, minority_indices)

def find_k_nearest_neighbours(data, means, k):
    distances = cdist(data.cpu().numpy(), means.cpu().numpy())
    k_nearest_indices = np.argsort(distances, axis=1)[:, :k]
    k_nearest_means = torch.tensor(means.cpu().numpy()[k_nearest_indices], device=device)
    return k_nearest_means

k = 3

# Vectorized version of generate_new_elements function
def generate_new_elements(minority_data, k_nearest_means, sigma, k, n_c):
    minority_data = minority_data[:, np.newaxis, :]
    
    mux = (torch.sum(k_nearest_means, axis=1) + minority_data) / (k + 1)
    
    sigmax = (torch.sum(sigma, axis=0) + torch.tensor(np.cov(minority_data.squeeze().cpu().numpy().T), device=device)) / (k + 1)
    
    # Generate new elements separately for each mean vector in mux
    new_elements = []
    for mean in mux.reshape(-1, mux.shape[-1]).cpu().numpy():
        new_elements.append(np.random.multivariate_normal(mean, sigmax.cpu().numpy(), n_c))
    
    return torch.tensor(np.vstack(new_elements), device=device)

new_elements = []

for index in minority_indices:
    minority_data = features[torch.argmax(u, axis=0) == index]
    if len(minority_data) < 2:  
        continue
    
    # Retrieve the GMM parameters (means and covariances)
    means, covariances, m_comp = gmm_params[index]

    if m_comp >= k:
        # Find the k-nearest means
        k_nearest_means = find_k_nearest_neighbours(minority_data, means, k)
        n_c = max(1, (len(majority_data) - len(minority_data)) // len(minority_data))
        
        # Generate new elements using the vectorized function
        new_elements_class = generate_new_elements(minority_data, k_nearest_means, covariances, k, n_c)
        new_elements.append(new_elements_class)
    else:
        n_comp = max(1, len(majority_data) // len(minority_data))  
        gmm = GaussianMixture(n_components=min(n_comp, len(minority_data)), init_params='kmeans')
        gmm.fit(minority_data.cpu().numpy()) 
        
        # Move the newly fitted GMM parameters to GPU tensors
        means = torch.tensor(gmm.means_, device=device)
        covariances = torch.tensor(gmm.covariances_, device=device)
        
        # Find the k-nearest means
        k_nearest_means = find_k_nearest_neighbours(minority_data, means, k)
        n_c = max(1, len(minority_data) * (len(majority_data) // n_comp) // len(minority_data))
        
        # Generate new elements
        new_elements_class = generate_new_elements(minority_data, k_nearest_means, covariances, k, n_c)
        new_elements.append(new_elements_class)

new_elements = torch.vstack(new_elements)

# Transfer to CPU for plotting
plt.scatter(features.cpu().numpy()[:, 0], features.cpu().numpy()[:, 1], c='blue', marker='o', alpha=0.5, label='Original Data')
plt.scatter(new_elements.cpu().numpy()[:, 0], new_elements.cpu().numpy()[:, 1], c='green', marker='s', alpha=0.5, label='Generated Data')
plt.title('Original and Generated Data Points')
plt.legend()
plt.show()
