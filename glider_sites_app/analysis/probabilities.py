import numpy as np
import pandas as pd
from scipy.stats import lognorm, weibull_min, norm
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
import plotly.express as px
import plotly.graph_objects as go



def fit_gmm_to_log_durations(log_durations, n_components=2, random_state=42):
    """
    Fit a Gaussian Mixture Model to log-transformed flight durations.
    
    Args:
        log_durations: Array of log-transformed flight durations
        n_components: Number of Gaussian components (default: 2)
        random_state: Random state for reproducibility
    
    Returns:
        gmm: Fitted GaussianMixture model
        dict: Dictionary with means, stds, weights, and other stats
    """
    X = log_durations.reshape(-1, 1)
    
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(X)
    
    # Extract parameters
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_
    
    # Sort by mean (ascending)
    sort_idx = np.argsort(means)
    means = means[sort_idx]
    stds = stds[sort_idx]
    weights = weights[sort_idx]
    
    stats = {
        'means': means,
        'stds': stds,
        'weights': weights,
        'n_components': n_components,
        'bic': gmm.bic(X),
        'aic': gmm.aic(X)
    }
    
    return gmm, stats


def gmm_cdf(x, means, stds, weights):
    """Calculate the CDF of a Gaussian Mixture Model."""
    cdf = np.zeros_like(x)
    for mean, std, weight in zip(means, stds, weights):
        cdf += weight * norm.cdf(x, loc=mean, scale=std)
    return cdf


def fit_weights_from_bins(mu_1, mu_2, sigma_1, sigma_2, bin_edges, target_probs):
    """
    Fit GMM weights given fixed means and stds, based on target bin probabilities.
    
    Args:
        mu_1, mu_2: Fixed means in log-space
        sigma_1, sigma_2: Fixed standard deviations in log-space
        bin_edges: Array of bin edges in minutes (will be converted to log seconds)
        target_probs: Array of target probabilities for each bin
    
    Returns:
        tuple: (weight_1, weight_2) fitted weights
    """
    # Convert bin edges from minutes to log(seconds)
    log_bin_edges = np.log(bin_edges * 60)
    
    def objective(weight_1):
        # weight_2 = 1 - weight_1
        weight_2 = 1 - weight_1
        
        # Ensure weight is valid
        if weight_1 <= 0 or weight_1 >= 1:
            return 1e10
        
        weights = np.array([weight_1, weight_2])
        means = np.array([mu_1, mu_2])
        stds = np.array([sigma_1, sigma_2])
        
        # Calculate CDF at bin edges
        cdf_values = gmm_cdf(log_bin_edges, means, stds, weights)
        
        # Calculate model probabilities for each bin
        model_probs = np.diff(np.insert(cdf_values, 0, 0))
        # Handle the last tail (last bin to infinity)
        model_probs = np.append(model_probs, 1 - cdf_values[-1])
        
        # Minimize MSE
        return np.sum((model_probs - target_probs)**2)
    
    # Optimize weight_1
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(objective, bounds=(0.001, 0.999), method='bounded')
    weight_1 = res.x
    weight_2 = 1 - weight_1
    
    return weight_1, weight_2


def load_site_params_and_fit_weights(site_name, bin_edges, target_probs, json_path='flight_durations_gmm.json'):
    """
    Load GMM parameters for a site from JSON and fit weights based on target bin probabilities.
    
    Args:
        site_name: Name of the site
        bin_edges: Array of bin edges in minutes
        target_probs: Array of target probabilities for each bin
        json_path: Path to the JSON file with GMM parameters
    
    Returns:
        dict: Dictionary with all parameters including fitted weights
    """
    import json
    
    # Load parameters from JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        sites_gmm = json.load(f)
    
    if site_name not in sites_gmm:
        raise ValueError(f"Site '{site_name}' not found in {json_path}")
    
    params = sites_gmm[site_name]
    
    # Extract mu and sigma (these are already in log-space)
    mu_1 = params['mu_1']
    mu_2 = params['mu_2']
    sigma_1 = params['sigma_1']
    sigma_2 = params['sigma_2']
    
    # Fit weights based on bins
    weight_1, weight_2 = fit_weights_from_bins(mu_1, mu_2, sigma_1, sigma_2, bin_edges, target_probs)
    
    result = {
        'mu_1': mu_1,
        'mu_2': mu_2,
        'sigma_1': sigma_1,
        'sigma_2': sigma_2,
        'weight_1': weight_1,
        'weight_2': weight_2,
        'weight_1_original': params['weight_1'],
        'weight_2_original': params['weight_2'],
        'n_flights': params['n_flights']
    }
    
    print(f"\n=== Fitted Weights for {site_name} ===")
    print(f"Original weights: w1={params['weight_1']:.3f}, w2={params['weight_2']:.3f}")
    print(f"Fitted weights:   w1={weight_1:.3f}, w2={weight_2:.3f}")
    print(f"Using mu1={mu_1:.3f}, sigma1={sigma_1:.3f}, mu2={mu_2:.3f}, sigma2={sigma_2:.3f}")
    
    return result


def fit_site_duration_distribution(df, n_components=2):
    """
    Fit a Gaussian Mixture Model to flight durations for a site.
    
    Args:
        df: DataFrame with 'FlightDuration' column
        n_components: Number of Gaussian components (default: 2)
    
    Returns:
        dict: Dictionary with GMM parameters (mu_i, sigma_i, weights) and statistics
    """
    # Calculate log durations
    log_durations = np.log(df['FlightDuration'].values)
    
    # Fit GMM to log durations
    gmm, stats = fit_gmm_to_log_durations(log_durations, n_components=n_components)
    
    # Add additional info
    stats['n_flights'] = len(df)
    stats['mu_1'] = stats['means'][0]
    stats['mu_2'] = stats['means'][1]
    stats['sigma_1'] = stats['stds'][0]
    stats['sigma_2'] = stats['stds'][1]
    stats['weight_1'] = stats['weights'][0]
    stats['weight_2'] = stats['weights'][1]
    
    print(f"\n=== GMM Fitting Results ===")
    print(f"Number of flights: {stats['n_flights']}")
    print(f"Component 1: mu={stats['mu_1']:.3f}, sigma={stats['sigma_1']:.3f}, weight={stats['weight_1']:.3f}")
    print(f"Component 2: mu={stats['mu_2']:.3f}, sigma={stats['sigma_2']:.3f}, weight={stats['weight_2']:.3f}")
    print(f"BIC: {stats['bic']:.2f}, AIC: {stats['aic']:.2f}")
    
    return stats


def plot_flight_duration_distribution(df, stats, site_name, output_path=None):
    """
    Plot flight duration distribution with GMM fit.
    
    Args:
        df: DataFrame with 'FlightDuration' column
        stats: Dictionary with GMM statistics from fit_site_duration_distribution
        site_name: Name of the site for the plot title
        output_path: Optional path to save the plot (default: flight_durations_{site_name}.png)
    
    Returns:
        fig: Plotly figure object
    """
    # Sort flights by duration and create ordinal positions
    df_sorted = df.sort_values('FlightDuration').reset_index(drop=True)
    df_sorted['LogDuration'] = np.log(df_sorted['FlightDuration'])
    df_sorted['ordinal'] = np.arange(1, len(df_sorted) + 1) / len(df_sorted)
    
    # Create x range for plotting curves
    x_range = np.linspace(df_sorted['LogDuration'].min(), df_sorted['LogDuration'].max(), 300)
    
    # Calculate GMM CDF
    gmm_cdf_values = gmm_cdf(x_range, stats['means'], stats['stds'], stats['weights'])
    
    # Create scatter plot of actual data
    fig = px.scatter(
        df_sorted, 
        x='LogDuration', 
        y='ordinal',
        title=f'Flight Duration Distribution for {site_name} (n={len(df)})',
        labels={'LogDuration': 'Log of Flight Duration (minutes)', 'ordinal': 'Cumulative Probability'}
    )
    
    # Add GMM curve
    fig.add_scatter(
        x=x_range, 
        y=gmm_cdf_values, 
        mode='lines', 
        name='2-Component GMM', 
        line=dict(color='red', width=2)
    )
    
    # Add individual components
    for i, (mean, std, weight) in enumerate(zip(stats['means'], stats['stds'], stats['weights']), 1):
        component_cdf = norm.cdf(x_range, loc=mean, scale=std)
        fig.add_scatter(
            x=x_range,
            y=component_cdf,
            mode='lines',
            name=f'Component {i} (w={weight:.2f})',
            line=dict(dash='dash', width=1),
            opacity=0.5
        )
    
    fig.update_layout(
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        width=1200,
        height=600
    )
    
    # Save if output path provided
    if output_path is None:
        output_path = f"flight_durations_{site_name.replace(' ', '_')}.png"
    fig.write_image(output_path)
    print(f"Plot saved to: {output_path}")
    
    return fig


if __name__ == "__main__":
    # Example: Fit weights based on bin probabilities
    bin_edges = np.array([9, 45, 120])  # in minutes
    target_probs = np.array([0.6, 0.1, 0.2, 0.1])  # probabilities for [0-9, 9-45, 45-120, 120+]
    
    site_name = 'Rammelsberg NW'
    
    # Load site parameters and fit weights
    result = load_site_params_and_fit_weights(site_name, bin_edges, target_probs)
    
    # Verify the fit by calculating resulting probabilities
    means = np.array([result['mu_1'], result['mu_2']])
    stds = np.array([result['sigma_1'], result['sigma_2']])
    weights = np.array([result['weight_1'], result['weight_2']])
    
    log_bin_edges = np.log(bin_edges * 60)
    cdf_values = gmm_cdf(log_bin_edges, means, stds, weights)
    model_probs = np.diff(np.insert(cdf_values, 0, 0))
    model_probs = np.append(model_probs, 1 - cdf_values[-1])
    
    print(f"\nTarget probabilities: {target_probs}")
    print(f"Model probabilities:  {model_probs}")
    print(f"Error: {np.sum((model_probs - target_probs)**2):.6f}")

