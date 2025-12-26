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
    # 5-15: Perfect
    # 15-25: Strong (Experts only)
    # >30: Nuke (Danger)
    data['Wind_State'] = pd.cut(
        df['avg_wind_speed'], 
        bins=[-np.inf, 5, 15, 25, np.inf], 
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
    # > 0.8: Good launch (45 deg -> sqrt 2 / 2 is 0.707)
    # < 0.5: Unflyable crosswind
    data['Alignment_State'] = pd.cut(
        df['avg_wind_alignment'], 
        bins=[-np.inf,0.5 , 0.8, np.inf], 
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

    # Create the 'Shear' variable (Absolute difference in speed)
    # Ideally vector difference, but scalar diff is a good proxy
    df['wind_shear'] = (df['wind_speed_850hPa'] - df['avg_wind_speed']).abs()

    data['Shear_State'] = pd.cut(
        df['wind_shear'],
        bins=[-np.inf, 10, 20, np.inf],
        labels=['Low', 'Moderate', 'High'] # >20km/h diff usually kills thermals
    )

    data['Wind_850_State'] = pd.cut(
        df['wind_speed_850hPa'],
        bins=[-np.inf, 15, 30, np.inf],
        labels=['Light', 'Drift', 'Strong']
    )

    data['Social_Window'] = df['is_workingday'].map({1: 'Low', 0: 'High'})


    # --- TARGETS (What we want to predict) ---
    # Binary Flyability
    data['Is_Flyable'] = df['flight_count'].apply(lambda x: 'Yes' if x > 0 else 'No')
    
    # --- NEW: XC POTENTIAL (The True Target) ---
    # Assuming 'max_daily_score' is the best flight of the day in points/km
    data['XC_Result'] = pd.cut(
        df['max_daily_score'].fillna(0), # 0 if nobody flew
        bins=[-np.inf, 5, 15, 50, np.inf], 
        labels=['A-Sled', 'B-Local', 'C-XC', 'D-Hammer']
    )

    return data.dropna() # BNs hate NaNs

def build_and_train_network(df_discrete):
    # Define the DAG (Directed Acyclic Graph)
    # Syntax: (Parent, Child)
    model = BayesianNetwork([
        # --- LAYER 1: Physics to Intermediate States ---
        # Safety depends on Wind Speed and Turbulence
        ('Wind_State',       'Launch_Safety'),
        ('Turbulence_State', 'Launch_Safety'),
        #('Wind_850_State',   'Shear_State'),   # 850 Wind drives Shear
        
        # Mechanics depends on Geometry (Alignment)
        ('Alignment_State',  'Site_Mechanics'),
        
        # Lift depends on Lapse Rate and Ceiling
        ('Thermal_Quality',  'Lift_Potential'),
        ('Ceiling_State',    'Lift_Potential'),
        #('Shear_State',      'Lift_Potential'), # High shear kills thermals

        # --- LAYER 2: Intermediate to Decisions ---
        # Can we fly? (Requires Safety AND Mechanics AND Pilots)
        ('Launch_Safety',    'Is_Flyable'),
        ('Site_Mechanics',   'Is_Flyable'),
        ('Social_Window',   'Is_Flyable'),

        # --- LAYER 3: Output ---
        # How good is the flight? (Requires Flyability AND Lift)
        ('Is_Flyable',       'XC_Result'),
        ('Lift_Potential',   'XC_Result')
    ])

    # Connect the learned data to the nodes
    # This automatically maps column names to nodes.
    # We must rename our DF columns to match these node names exactly.
    # (Mapping logic is below in the main execution block)
    
    logger.info("Training Bayesian Network...")
    model.fit(df_discrete, estimator=MaximumLikelihoodEstimator)
    
    return model


async def flight_predictor(site_name: str, main_direction: int):

    # Prepare data
    df = await prepare_training_data(site_name,main_direction,use_workingdays=False)
    
    if len(df) < 50:
        logger.error(f"Insufficient data: only {len(df)} days available")
        return None

    # 2. Discretize
    df_bn = discretize_data(df)

    logger.debug("\n=== Data Distribution ===")
    for col in df_bn.columns:
        logger.debug(f"{col}: {df_bn[col].value_counts().to_dict()}")


    # 3. Create Intermediate "Truth" Columns
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

    logger.info("\n=== SCENARIO 1: The 'Windy but Aligned' Day ===")
    logger.info("Wind: Strong, Alignment: Perfect, Lapse: Stable")
    q1 = infer.query(
        variables=['Is_Flyable', 'XC_Result'], 
        evidence={
            'Wind_State': 'Strong', 
            'Alignment_State': 'Perfect', 
            'Thermal_Quality': 'Stable',
            #'Wind_850_State': 'Strong',
            'Social_Window': 'High' # assume weekend
        }
    )
    logger.info(q1)

    logger.info("\n=== SCENARIO 2: The 'Perfect Thermal' Day ===")
    logger.info("Wind: Perfect, Lapse: Booming, Ceiling: High")
    q2 = infer.query(
        variables=['XC_Result'], 
        evidence={
            'Wind_State': 'Perfect', 
            'Thermal_Quality': 'Booming', 
            'Ceiling_State': 'High',
            #'Wind_850_State': 'Light',
            'Is_Flyable': 'Yes' # We assume we launched
        }
    )
    logger.info(q2)


if __name__ == '__main__':
    import asyncio
    
    #asyncio.run(flight_predictor('Rammelsberg NW', 315))
    #asyncio.run(flight_predictor('Königszinne', 270))
    asyncio.run(flight_predictor('Börry', 180))