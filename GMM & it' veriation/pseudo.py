Fuzzy(X, m)
{
    // Fuzzy clustering to find centroids
    return m-centroids
}

GMM(X, m)
{
    // Initialize centroids using Fuzzy clustering
    m-centroids = Fuzzy(X, m)

    // Initialize and train GMM with m centroids
    Initialize GMM with m centroids

    // Return m GMM components with mean and covariance
    return [(mu1, sigma1), (mu2, sigma2), ..., (mum, sigmam)]
}

main()
{
    M = |class 0|  // Majority class size
    GEN = []       // List to store generated data
    k = 3          // Number of neighbors

    for each class c in (1, C):  // Loop through all minority classes
    {
        m = M / |class c|  // Compute oversampling factor for class c
        
        // Train GMM with majority class data
        m-mixture-parameters = GMM(X_maj, m)
        
        if m > k:
        {
            // Generate synthetic samples with Gaussian components
            mu, sigma = gaussian(class c, 1) // Gaussian initialization for class c
            n_c = (M - |class c|) / |class c|  // Elements to generate per sample
            
            gen_c = []  // List for generated samples for class c
            
            for each element x in class c:
            {
                // Find k nearest neighbors among GMM means
                neighbors = find_k_nearest_neighbors(x, (mu1, mu2, ..., mum))
                
                // Calculate mean and covariance for new sample generation
                mux = (sum(neighbors.mu) + x) / (k + 1)
                sigmax = (sum(neighbors.sigma) + sigma) / (k + 1)
                
                // Generate synthetic samples
                gen_x = Sample((mux, sigmax), n_c)
                
                // Append generated samples
                gen_c.append(gen_x)
            }
        }
        else:
        {
            // Generate synthetic samples with fewer components
            n_comp = M / |class c|  // Number of GMM components for minority class
            
            // Train GMM on minority class data
            n_comp-mixture-parameters = GMM(class c, n_comp)
            
            n_c = |class c| * (M / n_comp)  // Elements to generate per component
            
            for each component (mu_comp, sigma_comp) in n_comp-mixture-parameters:
            {
                // Generate synthetic samples for each component
                gen_comp = Sample((mu_comp, sigma_comp), n_c)
                
                // Append generated samples
                gen_c.append(gen_comp)
            }
        }
        
        // Append generated data for class c to the overall list
        GEN.append(gen_c)
    }
}
