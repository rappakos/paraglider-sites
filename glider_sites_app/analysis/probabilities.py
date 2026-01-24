import numpy as np
from scipy.stats import lognorm, weibull_min
from scipy.optimize import minimize


def weibull_objective_function(params, bin_edges, target_probs):
    shape, scale = params
    # Ensure both are positive
    if shape <= 0 or scale <= 0: 
        return 1e10
    
    # Calculate the CDF values at the edges
    cdf_values = weibull_min.cdf(bin_edges, c=shape, scale=scale)
    
    # Calculate the model's probabilities for each bin
    model_probs = np.diff(np.insert(cdf_values, 0, 0))
    # Handle the last tail (150 to infinity)
    model_probs = np.append(model_probs, 1 - cdf_values[-1])
    
    # Minimize the difference (Mean Squared Error)
    return np.sum((model_probs - target_probs)**2)

def fit_weibull_to_probabilities(bin_edges, target_probs):
    res = minimize(weibull_objective_function, x0=[1.0, 30.0], args=(bin_edges, target_probs))
    shape_fit, scale_fit = res.x
    return shape_fit, scale_fit

def log_norm_objective_function(params, bin_edges, target_probs):
    mu, sigma = params
    # Ensure sigma is positive
    if sigma <= 0: return 1e10
    
    # Calculate the CDF values at the edges
    # For lognorm, s is the shape (sigma), and scale is exp(mu)
    cdf_values = lognorm.cdf(bin_edges, s=sigma, scale=np.exp(mu))
    
    # Calculate the model's probabilities for each bin
    # [p1, p2, p3, p4]
    model_probs = np.diff(np.insert(cdf_values, 0, 0))
    # Handle the last tail (150 to infinity)
    model_probs = np.append(model_probs, 1 - cdf_values[-1])
    
    # Minimize the difference (Mean Squared Error)
    return np.sum((model_probs - target_probs)**2)

def fit_lognorm_to_probabilities(bin_edges, target_probs):
    res = minimize(log_norm_objective_function, x0=[4.0, 1.0], args=(bin_edges, target_probs))
    mu_fit, sigma_fit = res.x
    return mu_fit, sigma_fit


if __name__ == "__main__":
    # Calculate Expected Value (Mean of Log-Normal)
    bin_edges = np.array([9, 45, 120])
    target_probs_list = np.array([
        #[0.93,0.01,0.05,0.01],
        [0.84,0.13,0.02,0.01],
        [0.6, 0.2, 0.1, 0.1],        
        [0.35,0.25,0.25,0.15],
        [0.73, 0.02, 0.22,0.03]
        ])  # Example probabilities for the bins

    print("=== Weibull Distribution ===")
    for i, target_probs in enumerate(target_probs_list):
        shape_fit, scale_fit = fit_weibull_to_probabilities(bin_edges, target_probs)
        # Weibull mean: scale * Gamma(1 + 1/shape)
        from scipy.special import gamma
        expected_airtime = scale_fit * gamma(1 + 1/shape_fit)
        print(f"{i+1}. Expected Airtime: {expected_airtime:.2f} minutes (shape={shape_fit:.3f}, scale={scale_fit:.2f})")
    
    print("\n=== Log-Normal Distribution ===")
    for i, target_probs in enumerate(target_probs_list[1:], 1):  # Skip first one
        mu_fit, sigma_fit = fit_lognorm_to_probabilities(bin_edges, target_probs)
        expected_airtime = np.exp(mu_fit + (sigma_fit**2 / 2))
        print(f"{i+1}. Expected Airtime: {expected_airtime:.2f} minutes (mu={mu_fit:.3f}, sigma={sigma_fit:.3f})")