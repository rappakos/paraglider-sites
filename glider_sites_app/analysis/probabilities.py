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
    # Calculate Expected Value (Mean of Log-Normal)
    bin_edges = np.array([9, 45, 120])
    target_probs_list = np.array([
        #[0.93,0.01,0.05,0.01],
        [0.84,0.13,0.02,0.01],
        [0.6, 0.2, 0.1, 0.1],        
        [0.35,0.25,0.25,0.15],
        [0.73, 0.02, 0.22,0.03]
        ])  # Example probabilities for the bins

