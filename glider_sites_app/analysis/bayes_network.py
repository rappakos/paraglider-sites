# analysis/bayes_network.py

import asyncio
import logging
import pandas as pd
import numpy as np
from glider_sites_app.services.weather_service import gust_factor
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator,BayesianEstimator
from pgmpy.inference import VariableElimination

from glider_sites_app.analysis.data_preparation import prepare_training_data
from glider_sites_app.analysis.model_loader import save_bayesian_model, load_bayesian_model, load_site_model, save_site_prior_counts, load_site_prior_counts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the DAG (Directed Acyclic Graph)
# Syntax: (Parent, Child)
NETWORK_GRAPH = [
        # --- LAYER 1: Physics to Intermediate States ---
        # Safety depends on Wind Speed and Turbulence
        ('Wind_State',       'Launch_Safety'),
        ('Turbulence_State', 'Launch_Safety'),
        #('Wind_850_State',   'Shear_State'),   # 850 Wind drives Shear
        
        # Mechanics depends on Geometry (Alignment)
        ('Alignment_State',  'Site_Mechanics'),
        ('Wind_State',      'Site_Mechanics'),
        
        # Lift depends on Lapse Rate and Ceiling
        ('Thermal_Quality',  'Lift_Potential'),
        ('Ceiling_State',    'Lift_Potential'),
        #('Shear_State',      'Lift_Potential'), # High shear kills thermals

        # --- LAYER 2: Intermediate to Decisions ---
        # Can we fly? (Requires Safety AND Mechanics AND Pilots)
        ('Launch_Safety',    'Is_Flyable'),
        ('Site_Mechanics',   'Is_Flyable'),
        ('Social_Window',   'Is_Flyable'),
        ('RF_Flyability_Confidence', 'Is_Flyable'),


        # --- LAYER 3: Output ---
        # How loing are the best flights?
        ('Is_Flyable', 'Avg_Flight_Duration'),
        ('Lift_Potential', 'Avg_Flight_Duration'),
        ('Site_Mechanics', 'Avg_Flight_Duration'),

        # How good is the flight? (Requires Flyability AND Lift)
        ('Is_Flyable',       'XC_Result'),
        ('Avg_Flight_Duration',   'XC_Result'),
        ('Pilot_Skill_Present', 'XC_Result')
    ]

STATE_NAMES = {
    'Wind_State':        ['Calm', 'Ideal', 'Strong', 'Very Strong', 'Extreme'],
    'Turbulence_State':  ['Smooth','OK', 'Gusty', 'Dangerous'],
    'Alignment_State':   ['Cross', 'Okay', 'Perfect'],
    'Thermal_Quality':   ['Stable', 'Weak', 'OK', 'Great'],
    'Ceiling_State':     ['Low', 'Average', 'High'],
    'Wind_850_State':   ['Light', 'Drift', 'Strong'],
    'Launch_Safety':    ['Safe', 'Unsafe'],
    'Site_Mechanics':   ['On', 'Off'],
    'Lift_Potential':   ['Good', 'Bad'],
    'Social_Window':    ['Low', 'High'],
    'Pilot_Skill_Present': ['Basic', 'Intermediate', 'Pro'],
    'RF_Flyability_Confidence': ['Low', 'Medium', 'High'],
    'Is_Flyable':       ['No', 'Yes'],
    'XC_Result':        ['A-Sled', 'B-Local', 'C-XC', 'D-Hammer'],
    'Avg_Flight_Duration': ['A-Abgleiter', 'B-Soaring', 'C-Good', 'D-Epic']
}

ESS = 50  # Equivalent Sample Size for Bayesian Estimation

def discretize_data(df):
    """
    Converts raw weather numbers into Discrete Physics States.
    Returns a new DataFrame suitable for the Bayesian Network.
    """
    data = pd.DataFrame()

    filter = df['date'].astype(str) >= '2022-01-01' # wind @ 850hPa and other data not available before
    df = df[filter]
    
    # ChatGPT Suggested Discretization Bins:
    # | Node                | Bins        |
    # | ------------------- | ----------- |
    # | Wind_State          | **4–6**     |
    # | Turbulence_State    | **3–4**     |
    # | Thermal_Quality     | **4–5**     |
    # | Ceiling_State       | **4**       |
    # | Alignment_State     | 3           |
    # | Launch_Safety       | 3           |
    # | Site_Mechanics      | 3           |
    # | Lift_Potential      | 3           |
    # | Social_Window       | 2–3         |
    # | Pilot_Skill_Present | 2–3         |
    # | Is_Flyable          | 2–3         |
    # | XC_Result           | 3 (maybe 4) |

    # --- 1. WIND STATE (The Mechanical Energy) ---
    # < 1.5 m/s  Calm             -> < 5.4 km/h
    # 1.5 -3.5 m/s  Ideal         -> 5.4 - 12.6 km/h
    # 3.5 - 5.5 m/s Strong        -> 12.6 - 19.8 km/h
    # 5.5 - 7.5  m/s Very Strong  -> 19.8 - 27 km/h
    # > 7.5  m/s Extreme          -> > 27 km/h
    #
    data['Wind_State'] = pd.cut(
        df['avg_wind_speed'], 
        bins=[-np.inf, 5.4, 12.6, 19.8, 27, np.inf], 
        #labels=['Calm', 'Perfect', 'Strong', 'Nuke']
        labels=STATE_NAMES['Wind_State']
    )

    # --- 2. GUST FACTOR (The Turbulence) ---    
    # Gust Factor 
    # Non-trivial expression based on base wind and max gust
    gust_fac = df.apply(lambda row: gust_factor(row['avg_wind_speed'], row['max_wind_gust']), axis=1)
    data['Turbulence_State'] = pd.cut(
        gust_fac, 
        bins=[-np.inf, 1.,2., 3., np.inf], 
        labels=STATE_NAMES['Turbulence_State']
    )
    
    # --- 3. ALIGNMENT (The Geometry) ---
    # Your COS metric (1.0 = Perfect, 0.0 = Cross)
    # > 0.8: Good launch (45 deg -> sqrt 2 / 2 is 0.707)
    # < 0.5: Unflyable crosswind (60 deg -> 0.5)
    data['Alignment_State'] = pd.cut(
        df['avg_wind_alignment'], 
        bins=[-np.inf,0.5 , 0.8, np.inf], 
        labels=STATE_NAMES['Alignment_State']
    )

    # --- 4. ENGINE (Lapse Rate / Lift) ---
    # Temp diff between 2m and 850hPa (approx 1500m)
    # < 4 °C/1000m: Stable 
    # 4-6 °C/1000m: Weak
    # 6-8 °C/1000m: OK
    # > 8 °C/1000m: Great
    # Rammi factor: Delta T / Delta Height * 1000=  Delta T / (1500-610) *1000 = Delta T * 1.12
    if 'max_lapse_rate' in df.columns:
        lr_col = 'max_lapse_rate'
    else:
        # Fallback calculation if column missing
        lr_col = 'calculated_lapse'
        data[lr_col] = df['temperature_2m'] - df['temperature_850hPa']
        
    data['Thermal_Quality'] = pd.cut(
        df[lr_col],
        bins=[-np.inf, 3.57,5.36, 7.14, np.inf],
        labels=STATE_NAMES['Thermal_Quality']
    )

    # --- 5. CEILING (BLH) ---
    # TODO Adjust bins
    # < 800m: Scratching
    # > 1500m: XC Highway
    data['Ceiling_State'] = pd.cut(
        df['max_boundary_layer_height'],
        bins=[-np.inf, 800, 1500, np.inf],
        labels=STATE_NAMES['Ceiling_State']
    )

    # Create the 'Shear' variable (Absolute difference in speed)
    # Ideally vector difference, but scalar diff is a good proxy
    #df['wind_shear'] = (df['wind_speed_850hPa'] - df['avg_wind_speed']).abs()

    #data['Shear_State'] = pd.cut(
    #    df['wind_shear'],
    #    bins=[-np.inf, 10, 20, np.inf],
    #    labels=['Low', 'Moderate', 'High'] # >20km/h diff usually kills thermals
    #)

    data['Wind_850_State'] = pd.cut(
        df['wind_speed_850hPa'],
        bins=[-np.inf, 15, 30, np.inf],
        labels=STATE_NAMES['Wind_850_State']
    )

    # Handle optional columns for forecast vs training data
    if 'is_workingday' in df.columns:
        data['Social_Window'] = df['is_workingday'].map({1: 'Low', 0: 'High'})
    else:
        # Default to High for forecasts
        data['Social_Window'] = 'High'

    if 'best_score' in df.columns:
        data['Pilot_Skill_Present'] = pd.cut(
            df['best_score'],
            bins=[-np.inf, 17, 87, np.inf],
            labels=STATE_NAMES['Pilot_Skill_Present']
        )
    else:
        # For forecast data, assume intermediate skill level
        data['Pilot_Skill_Present'] = 'Intermediate'

    # ---RF MODEL CONFIDENCE in flyability ---
    # This is actually the majority fraction from the RF
    if 'rf_confidence' in df.columns:
        data['RF_Flyability_Confidence'] = pd.cut(
            df.apply(lambda row: row['rf_confidence'] if row['rf_prediction'] else 1.0 - row['rf_confidence'], axis=1),
            bins=[-np.inf, 0.33, 0.67, np.inf],
            labels=STATE_NAMES['RF_Flyability_Confidence']
        )

    # --- TARGETS (What we want to predict) ---
    # Binary Flyability
    if 'flight_count' in df.columns:
        data['Is_Flyable'] = df['flight_count'].apply(lambda x: 'Yes' if x > 0 else 'No')
    else:
        # For forecast, this will be predicted
        data['Is_Flyable'] = 'No'  # Placeholder
    
    # --- XC POTENTIAL (The True Target) ---
    # Assuming 'max_daily_score' is the best flight of the day in points/km
    if 'max_daily_score' in df.columns:
        data['XC_Result'] = pd.cut(
            df['max_daily_score'].fillna(0), # 0 if nobody flew
            bins=[-np.inf, 5, 15, 50, np.inf], 
            labels=STATE_NAMES['XC_Result']
        )
    else:
        # For forecast, this will be predicted
        data['XC_Result'] = 'A-Sled'  # Placeholder


    if 'avg_flight_duration' in df.columns:
        data['Avg_Flight_Duration'] = pd.cut(
            df['avg_flight_duration'].fillna(0), # 0 if nobody flew
            bins=[-np.inf, 9, 45, 120, np.inf], 
            labels=STATE_NAMES['Avg_Flight_Duration']
        )
    else:
        data['Avg_Flight_Duration'] = 'A-Abgleiter'  # Placeholder

    return data.dropna() # BNs hate NaNs


def add_intermediate_states(df_discrete):
    """
    Add intermediate state columns for Bayesian Network inference.
    These states are derived from discretized weather states.
    """
    def label_safety(row):
        if row['Wind_State'] in ['Very Strong', 'Extreme'] or row['Turbulence_State'] == 'Dangerous':
            return 'Unsafe'
        return 'Safe'
    
    def label_dynamic_lift(row):
        # previous:
        # df_discrete['Alignment_State'].apply(lambda x: 'On' if x in ['Perfect', 'Okay'] else 'Off')
        if row['Alignment_State'] == 'Cross':
            return 'Off'
        if row['Alignment_State'] == 'Okay' and row['Wind_State'] in ['Strong', 'Very Strong']:
            return 'On'
        if row['Alignment_State'] == 'Perfect' and row['Wind_State'] in ['Ideal', 'Strong', 'Very Strong']:
            return 'On'
        
        return 'Off'
    
    df_discrete['Launch_Safety'] = df_discrete.apply(label_safety, axis=1)
    df_discrete['Site_Mechanics'] = df_discrete.apply(label_dynamic_lift, axis=1)
    df_discrete['Lift_Potential'] = df_discrete.apply(lambda x: 'Good' if x['Thermal_Quality'] in ['OK', 'Great'] else 'Bad', axis=1)
    
    return df_discrete


def get_formatted_pseudo_counts(model, global_counts):
    """
    Transforms flat global counts into the (node_card, product_of_parents_card) 
    shape required by pgmpy.
    
    For each node, creates an array of shape [num_states(node), ∏ num_states(parents)]
    where the prior counts are distributed based on global frequencies.
    """
    formatted_pseudo_counts = {}

    equivalent_sample_size = ESS  # Use the global ESS constant
    
    for node in model.nodes():
        # Get the states for this node (in semantic order from STATE_NAMES)
        node_states = STATE_NAMES[node]
        node_card = len(node_states)
        
        # Get parent information
        parents = list(model.get_parents(node))
        
        if not parents:
            # No parents: shape is (node_card, 1)
            # Fill with global counts for each state
            counts_array = np.ones((node_card, 1))
            for i, state in enumerate(node_states):
                # Use global count if available, otherwise use 1
                counts_array[i, 0] = global_counts.get(node, {}).get(state, 1)
            formatted_pseudo_counts[node] = counts_array
        else:
            # Has parents: shape is (node_card, product of parent cardinalities)
            parent_cards = [len(STATE_NAMES[p]) for p in parents]
            parent_card_prod = np.prod(parent_cards)
            
            # Initialize with global prior for this node
            counts_array = np.ones((node_card, parent_card_prod))
            
            # Distribute global counts across all parent combinations
            for i, state in enumerate(node_states):
                # 1. Get the raw global count
                global_count = global_counts.get(node, {}).get(state, 1)
                
                # 2. Convert to a PROBABILITY (Frequency)
                # Total global observations for this node
                total_node_obs = sum(global_counts.get(node, {}).values())
                global_prob = global_count / total_node_obs if total_node_obs > 0 else 1/node_card
                
                # 3. Scale by ESS (e.g., 150) and distribute
                # This ensures the TOTAL prior weight for this node is exactly 150
                counts_array[i, :] = (global_prob * equivalent_sample_size) / parent_card_prod
            
            formatted_pseudo_counts[node] = counts_array
        
        logger.debug(f"{node}: shape {counts_array.shape}, parents={parents}")
    
    return formatted_pseudo_counts


async def build_and_train_network(df_discrete, skip_fit:bool=False, maximum_likelihood:bool=False, personalized:bool=False, site_name: str = None):

    model = BayesianNetwork(NETWORK_GRAPH)

    # Only add nodes that appear in the network graph (connected nodes)
    nodes_in_graph = set()
    for parent, child in NETWORK_GRAPH:
        nodes_in_graph.add(parent)
        nodes_in_graph.add(child)
    
    model.add_nodes_from(nodes_in_graph)
    model.state_names = { node: states for node, states in STATE_NAMES.items() if node in nodes_in_graph }

    if skip_fit:
        # to get the global prior counts only
        return model
    
    logger.info("Training Bayesian Network...")

    if personalized:
        # Try to load prior counts for this site
        prior_counts = load_site_prior_counts(site_name)
    else:
        prior_counts = await get_global_prior_counts(recalculate=False)

    formatted_pseudo_counts = get_formatted_pseudo_counts(model, prior_counts)

    if maximum_likelihood:
        estimator = MaximumLikelihoodEstimator(model, df_discrete)
        cpds = estimator.get_parameters()
    else:
        estimator = BayesianEstimator(model, df_discrete, state_names=model.state_names)
        cpds = estimator.get_parameters(
            prior_type='dirichlet', 
            pseudo_counts=formatted_pseudo_counts, 
            equivalent_sample_size=100  if personalized else ESS
        )
    
    model.add_cpds(*cpds)    

    logger.info(f"Model consistent: {model.check_model()}")

    return model


def predict_from_raw_weather(model, raw_weather_df):
    """
    Predict flight outcomes from raw (undiscretized) weather data
    
    Args:
        model: Trained Bayesian Network model
        raw_weather_df: DataFrame with raw weather features
        
    Returns:
        DataFrame with predictions and probabilities
    """
    # 1. Discretize the input data
    df_discrete = discretize_data(raw_weather_df)
    
    # 2. Add intermediate state columns for inference
    df_discrete = add_intermediate_states(df_discrete)
    
    # 3. Run inference for each row
    infer = VariableElimination(model)
    predictions = []
    
    for idx, row in df_discrete.iterrows():
        try:
            # Build evidence dictionary from the row
            evidence = {
                'Wind_State': row['Wind_State'],
                'Turbulence_State': row['Turbulence_State'],
                'Alignment_State': row['Alignment_State'],
                'Thermal_Quality': row['Thermal_Quality'],
                'Ceiling_State': row['Ceiling_State'],
                'Social_Window': row['Social_Window'], # High for forecasts
                'Pilot_Skill_Present': row['Pilot_Skill_Present'], # Intermediate for forecasts
                'RF_Flyability_Confidence': row['RF_Flyability_Confidence']
            }
            
            # Query the model for each variable separately
            flyable_result = infer.query(variables=['Is_Flyable'], evidence=evidence)
            xc_result = infer.query(variables=['XC_Result'], evidence=evidence)
            
            # Extract probabilities from DiscreteFactor objects
            # The values are ordered alphabetically by state name
            # TODO review if these are still correct with the lock_categories function
            flyable_states = flyable_result.state_names['Is_Flyable']
            flyable_values = flyable_result.values
            flyable_prob = flyable_values[flyable_states.index('Yes')] if 'Yes' in flyable_states else 0
            
            xc_states = xc_result.state_names['XC_Result']
            xc_values = xc_result.values
            
            predictions.append({
                'date': raw_weather_df.iloc[idx].get('date', idx),
                'is_flyable_prob': flyable_prob,
                'xc_sled_prob': xc_values[xc_states.index('A-Sled')] if 'A-Sled' in xc_states else 0,
                'xc_local_prob': xc_values[xc_states.index('B-Local')] if 'B-Local' in xc_states else 0,
                'xc_xc_prob': xc_values[xc_states.index('C-XC')] if 'C-XC' in xc_states else 0,
                'xc_hammer_prob': xc_values[xc_states.index('D-Hammer')] if 'D-Hammer' in xc_states else 0,
                'predicted_flyable': 'Yes' if flyable_prob > 0.5 else 'No'
            })
        except Exception as e:
            logger.warning(f"Prediction failed for row {idx}: {e}")
            predictions.append({
                'date': raw_weather_df.iloc[idx].get('date', idx),
                'is_flyable_prob': 0,
                'predicted_flyable': 'No'
            })
    
    return pd.DataFrame(predictions)

def lock_categories(df):
    """
    Forces the dataframe to include all possible states, 
    even if they have 0 occurrences.
    """
    for node, states in STATE_NAMES.items():
        if node in df.columns:
            # Check if we should treat it as an ordered progression
            # Logic: Multi-step bins (Wind, Duration, XC) are Ordered. 
            # Binary gates (Flyable, Mechanics) are Unordered.
            is_ordered = True if len(states) > 2 else False
            df[node] = pd.Categorical(df[node], categories=states, ordered=is_ordered   )
    return df


async def prepare_discretized_data(site_name: str, FKPilotID: str = None):
    # Prepare data
    df = await prepare_training_data(site_name,use_workingdays=True, FKPilotID=FKPilotID)
    
    # Include rf model predictions if available
    rf_model = await asyncio.to_thread(load_site_model, site_name, type='classifier')
    if rf_model is not None:
        X = df[rf_model['features']]
        df['rf_prediction'] = rf_model['model'].predict(X)
        df['rf_confidence'] = rf_model['model'].predict_proba(X).max(axis=1)
        logger.info(f"Included RF model predictions for Bayesian training.")

    if len(df) < 50:
        logger.error(f"Insufficient data: only {len(df)} days available")
        return None

    # 2. Discretize
    df_bn = discretize_data(df)

    # 3. Create Intermediate "Truth" Columns
    # Since we are training, we need to tell the model what "Launch_Safety" actually WAS historically.
    # We create these synthetic ground-truth columns based on logic.
    df_bn = add_intermediate_states(df_bn)

    # 4. Lock categories to ensure all possible states exist
    df_bn = lock_categories(df_bn)

    return df_bn


async def get_global_prior_counts(recalculate: bool = False):
    """
    Get global prior counts for all sites.
    
    Args:
        recalculate: If True, recalculate from data and save to file. 
                     If False (default), load from file if it exists.
    
    Returns:
        Dictionary of prior counts for each node
    """
    import json
    from pathlib import Path
    
    cache_file = Path('global_prior_counts.json')
    
    # Try to load from cache if not recalculating
    if not recalculate and cache_file.exists():
        logger.info(f"Loading global prior counts from {cache_file}")
        with open(cache_file, 'r') as f:
            global_prior_counts = json.load(f)
        logger.info("Global prior counts loaded from cache.")
        return global_prior_counts
    
    # Calculate from data
    logger.info("Calculating global prior counts from data...")
    dfs = [ await prepare_discretized_data(site_name) 
            for site_name in ['Rammelsberg NW', 'Königszinne', 'Börry', 'Porta', 'Brunsberg']]
    global_df = pd.concat(dfs, ignore_index=True)

    logger.info(f"\n=== Global Prior Counts  ===")
    for col in global_df.columns:
        logger.debug(f"{col}: {global_df[col].value_counts().to_dict()}")

    # if this is reworked then global_df does not need to be passed here?!
    model = await build_and_train_network(global_df, skip_fit=True)

    global_prior_counts = {node: global_df[node].value_counts().to_dict() # if node != 'Alignment_State' else {'Cross':100, 'Okay':100, 'Perfect':100}
                             for node in model.nodes()}
    logger.info(f"Global prior counts computed.")
    logger.info(global_prior_counts)
    
    # Save to cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(global_prior_counts, f, indent=2)
    logger.info(f"Global prior counts saved to {cache_file}")

    return global_prior_counts


async def flight_predictor(site_name: str, 
                           save_model: bool = False, 
                           maximum_likelihood: bool = False):

    df_bn = await prepare_discretized_data(site_name)

    logger.debug("\n=== Data Distribution ===")
    for col in df_bn.columns:
        logger.info(f"{col}: {df_bn[col].value_counts().to_dict()}")

    # 4. Train
    model = await build_and_train_network(df_bn, maximum_likelihood=maximum_likelihood)
    
    # Save site-specific prior counts
    save_site_prior_counts(site_name, {node: df_bn[node].value_counts().to_dict() for node in model.nodes()})

    logging.debug((model.get_cpds('Avg_Flight_Duration')))
    
    # Save model if requested
    if save_model:
        features = [
            'avg_wind_speed', 'max_wind_gust', 'avg_wind_alignment',
            'max_lapse_rate', 'max_boundary_layer_height', 'wind_speed_850hPa',
             'rf_confidence','rf_prediction'
            'is_workingday', 'best_score','avg_flight_duration'
        ]
        save_bayesian_model(model, site_name, features)

    # TEST RELOAD
    loaded_model_data = None # load_bayesian_model(site_name)
    if loaded_model_data:
        infer = VariableElimination(loaded_model_data['model'])
        logger.info(f"Loaded model with features: {loaded_model_data['features']}")
    else:
        # 5. Query the Model (Inference)
        infer = VariableElimination(model)

    logger.info("\n=== SCENARIO 1: The 'Windy but Aligned' Day ===")
    logger.info("Wind: Strong, Alignment: Perfect, Lapse: Stable")
    evidence={
            'Wind_State': 'Strong', 
            'Alignment_State': 'Perfect', 
            'Thermal_Quality': 'Stable',
            #'Wind_850_State': 'Strong',
            'Is_Flyable': 'Yes',
            'Pilot_Skill_Present': 'Intermediate'
    }    
    q1 = infer.query(variables=['XC_Result'], evidence=evidence)
    logger.info(q1)
    q1 = infer.query(variables=['Avg_Flight_Duration'], evidence=evidence)
    logger.info(q1)    


    pilot_skill = 'Intermediate'
    logger.info("\n=== SCENARIO 2: The 'Perfect Thermal' Day ===")
    logger.info(f"Wind: Perfect, Lapse: Great, Ceiling: High, Pilot: {pilot_skill}")
    evidence={
            'Wind_State': 'Ideal', 
            'Thermal_Quality': 'Great', 
            'Ceiling_State': 'High',
            'Alignment_State': 'Perfect',
            #'Wind_850_State': 'Light',
            'Is_Flyable': 'Yes' # We assume we launched
            , 'Pilot_Skill_Present': pilot_skill
        }
    q2 = infer.query(variables=['XC_Result'], evidence=evidence)
    logger.info(q2)
    q2 = infer.query(variables=['Avg_Flight_Duration'], evidence=evidence)
    logger.info(q2)


async def personal_predictor(FKPilotID: str):
    results = {}
    for site_name in ['Rammelsberg NW', 'Königszinne', 'Börry', 'Porta', 'Brunsberg']:

        df_bn = await prepare_discretized_data(site_name, FKPilotID)
        logger.info(f"Training personalized model for site {site_name} with {len(df_bn[df_bn['Is_Flyable'].isin(['Yes'])])} days of data.")
        model = await build_and_train_network(df_bn, maximum_likelihood=False, personalized=True, site_name=site_name)


        infer = VariableElimination(model)
        evidence={
            'Wind_State': 'Ideal', 
            'Thermal_Quality': 'Great', 
            'Ceiling_State': 'High',
            #'Wind_850_State': 'Light',
            'Is_Flyable': 'Yes' # We assume we launched
            , 'Pilot_Skill_Present':  'Intermediate'
        }
        q2 = infer.query(variables=['Avg_Flight_Duration'], evidence=evidence)
        probs = {state: prob for state, prob in zip(STATE_NAMES['Avg_Flight_Duration'], q2.values)}
        results[site_name] = probs

    logger.info(pd.DataFrame(results))

if __name__ == '__main__':
    import asyncio
    
    # Set up global counts
    #asyncio.run(get_global_prior_counts(recalculate=True))

    # Train and save model
    save=False
    asyncio.run(flight_predictor('Rammelsberg NW', save_model=save))
    #asyncio.run(flight_predictor('Königszinne', save_model=save))
    #asyncio.run(flight_predictor('Börry', save_model=save))
    #asyncio.run(flight_predictor('Porta', save_model=save))
    #asyncio.run(flight_predictor('Brunsberg', save_model=save))

    p = {
        'A': 12957,
        'O': 12953
    }
    
    #asyncio.run(personal_predictor(FKPilotID=p['A']))