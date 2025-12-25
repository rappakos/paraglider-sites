# analysis/bayes_network.py

import logging
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

from glider_sites_app.analysis.data_preparation import prepare_training_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def discretize_data(df):
    """
    Converts raw weather numbers into Discrete Physics States.
    Returns a new DataFrame suitable for the Bayesian Network.
    """
    data = pd.DataFrame()
    
    # --- 1. WIND STATE (The Mechanical Energy) ---
    # Thresholds based on your Random Forest findings
    # 0-5: Too weak (Parawaiting)
    # 5-25: Perfect
    # 25-30: Strong (Experts only)
    # >30: Nuke (Danger)
    data['Wind_State'] = pd.cut(
        df['avg_wind_speed'], 
        bins=[-np.inf, 5, 25, 30, np.inf], 
        labels=['Calm', 'Perfect', 'Strong', 'Nuke']
    )

    # --- 2. GUST FACTOR (The Turbulence) ---
    # Gust Factor = Max Gust / Avg Speed
    # < 1.5: Smooth Laminar
    # > 1.8: Dangerous Turbulence
    gust_factor = df['max_wind_gust'] / (df['avg_wind_speed'] + 1) # +1 avoids div/0
    data['Turbulence_State'] = pd.cut(
        gust_factor, 
        bins=[-np.inf, 1.5, 1.8, np.inf], 
        labels=['Smooth', 'Gusty', 'Dangerous']
    )
    
    # --- 3. ALIGNMENT (The Geometry) ---
    # Your COS metric (1.0 = Perfect, 0.0 = Cross)
    # > 0.8: Good launch
    # < 0.5: Unflyable crosswind
    data['Alignment_State'] = pd.cut(
        df['avg_wind_alignment'], 
        bins=[-np.inf, 0.5, 0.8, np.inf], 
        labels=['Cross', 'Okay', 'Perfect']
    )

    # --- 4. ENGINE (Lapse Rate / Lift) ---
    # Temp diff between 2m and 850hPa (approx 1500m)
    # < 4C: Inversion / Stable (Sled ride)
    # > 8C: Unstable (Good XC)
    if 'max_lapse_rate' in df.columns:
        lr_col = 'max_lapse_rate'
    else:
        # Fallback calculation if column missing
        lr_col = 'calculated_lapse'
        data[lr_col] = df['temperature_2m'] - df['temperature_850hPa']
        
    data['Thermal_Quality'] = pd.cut(
        df[lr_col],
        bins=[-np.inf, 4, 8, np.inf],
        labels=['Stable', 'Weak', 'Booming']
    )

    # --- 5. CEILING (BLH) ---
    # < 800m: Scratching
    # > 1500m: XC Highway
    data['Ceiling_State'] = pd.cut(
        df['max_boundary_layer_height'],
        bins=[-np.inf, 800, 1500, np.inf],
        labels=['Low', 'Average', 'High']
    )

    # --- 6. TARGETS (What we want to predict) ---
    # Binary Flyability
    data['Is_Flyable'] = df['flight_count'].apply(lambda x: 'Yes' if x > 0 else 'No')
    
    # XC Potential (Optional: Define based on km or duration)
    # Simple proxy: If distinct pilots > 5, it's a "Good Day"
    data['Day_Potential'] = df['flight_count'].apply(lambda x: 'High' if x > 10 else 'Low')

    return data.dropna() # BNs hate NaNs

def build_and_train_network(df_discrete):
    # Define the DAG (Directed Acyclic Graph)
    # Syntax: (Parent, Child)
    model = BayesianNetwork([
        # --- LAYER 1: Physics to Intermediate States ---
        # Safety depends on Wind Speed and Turbulence
        ('Wind_State',       'Launch_Safety'),
        ('Turbulence_State', 'Launch_Safety'),
        
        # Mechanics depends on Geometry (Alignment)
        ('Alignment_State',  'Site_Mechanics'),
        
        # Lift depends on Lapse Rate and Ceiling
        ('Thermal_Quality',  'Lift_Potential'),
        ('Ceiling_State',    'Lift_Potential'),

        # --- LAYER 2: Intermediate to Decisions ---
        # Can we fly? (Requires Safety AND Mechanics)
        ('Launch_Safety',    'Is_Flyable'),
        ('Site_Mechanics',   'Is_Flyable'),

        # --- LAYER 3: Output ---
        # How good is the flight? (Requires Flyability AND Lift)
        ('Is_Flyable',       'XC_Result'),
        ('Lift_Potential',   'XC_Result')
    ])

    # Connect the learned data to the nodes
    # This automatically maps column names to nodes.
    # We must rename our DF columns to match these node names exactly.
    # (Mapping logic is below in the main execution block)
    
    print("Training Bayesian Network...")
    model.fit(df_discrete, estimator=MaximumLikelihoodEstimator)
    
    return model



async def flight_predictor(site_name: str, main_direction: int):

    # Prepare data
    df = await prepare_training_data(site_name,main_direction)
    
    if len(df) < 50:
        logger.error(f"Insufficient data: only {len(df)} days available")
        return None

    # 2. Discretize
    df_bn = discretize_data(df)

    # 3. Rename columns to match the Network Nodes strictly
    # This tells the model which column belongs to which node
    df_bn = df_bn.rename(columns={
        'Day_Potential': 'XC_Result' 
        # Note: Intermediate nodes like 'Launch_Safety' are Latent (Hidden).
        # If they don't exist in data, we can't train them directly with MaximumLikelihood.
        # STRATEGY: For Phase 1, we will SIMPLIFY the network to map Inputs -> Output directly
        # OR we must manually engineer the 'Launch_Safety' column in the discretization step.
    })

    # --- IMPORTANT: Creating the Intermediate "Truth" Columns ---
    # Since we are training, we need to tell the model what "Launch_Safety" actually WAS historically.
    # We create these synthetic ground-truth columns based on logic.
    def label_safety(row):
        if row['Wind_State'] == 'Nuke' or row['Turbulence_State'] == 'Dangerous':
            return 'Unsafe'
        return 'Safe'

    df_bn['Launch_Safety'] = df_bn.apply(label_safety, axis=1)
    df_bn['Site_Mechanics'] = df_bn['Alignment_State'].apply(lambda x: 'On' if x in ['Perfect', 'Okay'] else 'Off')
    df_bn['Lift_Potential'] = df_bn.apply(lambda x: 'Good' if x['Thermal_Quality'] == 'Booming' else 'Bad', axis=1)

    # 4. Train
    model = build_and_train_network(df_bn)

    # 5. Query the Model (Inference)
    infer = VariableElimination(model)

    print("\n=== SCENARIO 1: The 'Windy but Aligned' Day ===")
    print("Wind: Strong (25-30kmh), Alignment: Perfect, Lapse: Stable")
    q1 = infer.query(
        variables=['Is_Flyable', 'XC_Result'], 
        evidence={
            'Wind_State': 'Strong', 
            'Alignment_State': 'Perfect', 
            'Thermal_Quality': 'Stable'
        }
    )
    print(q1)

    print("\n=== SCENARIO 2: The 'Perfect Thermal' Day ===")
    print("Wind: Perfect, Lapse: Booming, Ceiling: High")
    q2 = infer.query(
        variables=['XC_Result'], 
        evidence={
            'Wind_State': 'Perfect', 
            'Thermal_Quality': 'Booming', 
            'Ceiling_State': 'High',
            'Is_Flyable': 'Yes' # We assume we launched
        }
    )
    print(q2)



if __name__ == '__main__':
    import asyncio
    
    asyncio.run(flight_predictor('Rammelsberg NW', 315))
    #asyncio.run(flight_predictor('Königszinne', 270))
    #asyncio.run(flight_predictor('Börry', 180))