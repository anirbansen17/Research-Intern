import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
import skfuzzy as fuzz  



# Fuzzy C-Means Clustering Function
def Fuzzy(X, m):
    """
    Perform Fuzzy C-Means clustering on the majority class elements X.
    
    Parameters:
    - X: Majority class data (numpy array or equivalent)
    - m: Number of components (clusters)

    Returns:
    - centroids: m centroids obtained from Fuzzy C-Means
    """
    
    X_transposed = X.T
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        X_transposed, m, 2, error=0.005, maxiter=1000, init=None
    )

    return cntr




# Function to perform GMM and return the mixture parameters
def GMM(X, m):
    centroids = Fuzzy(X, m)
    
    gmm = GaussianMixture(n_components=m, means_init=centroids)
    gmm.fit(X)
    
    return gmm.means_, gmm.covariances_




# Function to generate new samples based on Gaussian sampling
def Sample(mu, sigma, n_samples):
    return np.random.multivariate_normal(mu, sigma, n_samples)




# Main function 
def augment_data(X_majority, X_minority, k=3):
    M = len(X_majority)  
    GEN = []  
    
    for class_c_data in X_minority:      
        n_c = len(class_c_data)
        m = M / n_c  # ratio

        # Perform GMM on majority class 
        mu, sigma = GMM(X_majority, int(m))

        genc = [] 

        if m > k:
            # Case when m > k
            for x in class_c_data:
                # Find k-nearest centroids (mu's)
                dists = [distance.euclidean(x, mu_i) for mu_i in mu]
                k_nearest_indices = np.argsort(dists)[:k]
                mu_k = mu[k_nearest_indices]
                sigma_k = sigma[k_nearest_indices]

                # Calculate mux and sigmax
                mux = (np.sum(mu_k, axis=0) + x) / (k + 1)
                sigmax = (np.sum(sigma_k, axis=0) + np.cov(class_c_data.T)) / (k + 1)

                # Number of samples to generate for each element
                n_to_generate = int((M - n_c) / n_c)
                gen_x = Sample(mux, sigmax, n_to_generate)

                genc.append(gen_x)

        else:
            # Case when m < k
            n_comp = int(M / n_c)
            mu_comp, sigma_comp = GMM(class_c_data, n_comp)

            for mu_i, sigma_i in zip(mu_comp, sigma_comp):
                n_to_generate = int(n_c * (M / n_comp))
                gen_comp = Sample(mu_i, sigma_i, n_to_generate)
                genc.append(gen_comp)
        
        GEN.append(genc)
    
    return GEN




# Load datasets 
features = torch.load('imbalanced_train_features.pt')  
labels = torch.load('imbalanced_train_labels.pt')     

# Function to find majority and minority classes
def prepare_data(features, labels):
    
    unique_classes, class_counts = torch.unique(labels, return_counts=True)
    
    # find Majority class 
    majority_class = unique_classes[torch.argmax(class_counts)]
    X_majority = features[labels == majority_class].numpy()  

    # find minority classes
    X_minority = [features[labels == uc].numpy() for uc in unique_classes if uc != majority_class]

    
    print(f"Majority class: {majority_class.item()} with {class_counts[torch.argmax(class_counts)].item()} samples.")
    for uc, count in zip(unique_classes, class_counts):
        if uc != majority_class:
            print(f"Minority class: {uc.item()} with {count.item()} samples.")
    
    return X_majority, X_minority




# Separate into majority and minority class 
X_majority, X_minority = prepare_data(features, labels)

# Perform k-nearest neighbors
k = 3
generated_samples = augment_data(X_majority, X_minority, k)


generated_samples_tensors = [torch.tensor(np.concatenate(gen_class)) for gen_class in generated_samples]


print("Generated samples:", generated_samples_tensors)
