# Background for site-specific paragliding forecasting in the north German flatlands

## Motivation

There exist some direct paragliding flyability applications like [paraglidable.com](https://paraglidable.com) and [Burnair](https://burnair.cloud), however, they both seem to be very "dramatic" with respect to the conditions. Especially paraglidable has a deep neural network trained on dominantly alpine flying data. This approach does not generalize well to our region because:

1. **Alpine vs. flatland conditions differ significantly**: Mountain flying involves ridge lift, orographic effects, and terrain-induced turbulence patterns that don't exist in flat terrain
2. **Training data bias**: Models trained on alpine data overweight factors like terrain slope, valley wind patterns, and altitude effects that are irrelevant in flatlands

In the flatlands of northern Germany (Niedersachsen) we have some special considerations:
* **Wind tolerance**: Stronger winds don't necessarily lead to turbulent and dangerous flying conditions due to lack of terrain obstacles
* **Data sparsity**: Relatively sparse flight data makes deep learning approaches impractical (overfitting risk)
* **Social factors**: No flights on a working day does not mean the site was not flyable - weekday vs weekend patterns must be accounted for
* **Thermal characteristics**: Flatland thermals are weaker and more dependent on surface characteristics (fields, forests, urban areas) rather than terrain elevation

## Data sources

### Weather

* **Source**: [OpenMeteo](https://open-meteo.com/) historic and forecast API
* **Coverage**: 2018-01-01 to present
* **Pressure level data**: Temperature and wind at 850 hPa (approximately 1,500m altitude) for pre-2022 days
* **Aggregation**: Hourly data aggregated for each day from 10:00 to 18:00 local time (typical soaring window)
* **Key variables**:
  - Wind speed and direction (surface and 850 hPa)
  - Temperature and lapse rate (thermal strength indicator)
  - Precipitation and cloud cover
  - Sunshine duration (thermal trigger)

### DHV XC

* **Source**: [DHV XC](https://www.dhv-xc.de/) - German Hang Gliding Association cross-country database
* **API**: Convenient public JSON API (allows easy site identification)
* **Coverage**: 2018-01-01 to present
* **Data richness**: Includes flight duration, distance, XC score, pilot information, launch coordinates, and more
* **Advantage**: Most complete and reliable dataset for the region

### Xcontest

* **Source**: [XContest](https://www.xcontest.org/) - International paragliding and hang gliding contest platform
* **Access method**: HTML parsing with authentication (no public API)
* **Coverage**: 2018-01-01 to present
* **Data gaps**: Some fields missing in base data (exact start coordinates to distinguish nearby launches like Rammelsberg NW and SW, flight duration)
* **Usage**: Supplementary data source for days when DHV XC flights are not available


## Models

### Random Forest

* **Approach**: Supervised classification for binary flyability prediction (flyable/not flyable)
* **Advantage**: Site-specific models capture local characteristics (terrain, orientation, typical weather patterns)
* **Training strategy**: 
  - Includes only flying days and weekends to remove social factors
  - Balanced dataset of flyable and unflyable dates
  - Uses weather conditions as features, flight occurrence as label
* **Feature engineering** (7 key features):
    * `avg_wind_speed` - Mean ground wind speed during soaring window
    * `avg_wind_alignment` - Wind direction alignment with site orientation (see below)
    * `max_wind_gust` - Safety indicator for turbulent conditions
    * `min_wind_speed` - Minimum wind to start and maintain ridge lift
    * `total_sunshine` - Thermal strength proxy
    * `total_precipitation` - Soaring inhibitor
    * `max_lapse_rate` - Atmospheric stability indicator (thermal potential)
* **Hyperparameters**: 200 trees with max depth of 5
* **Performance**: Up to 84% accuracy in 5-fold cross-validation
* **Feature importance insights**:
  - `avg_wind_alignment = cos(avg_wind_direction - site_direction)` is dominant (>50% for many sites)
  - Wind alignment captures the critical factor for ridge soaring sites
  - `total_precipitation` is consistently least important (~2% for Börry)
* **Interpretation**: Model effectively learns that wind direction relative to site orientation is the primary flyability factor in flatland ridge soaring


### Bayesian Network

* **Approach**: Probabilistic graphical model for multi-output prediction and causal reasoning
* **Advantages over Random Forest**:
  - Explicitly models correlations between variables (interpretable structure)
  - Learns multiple output variables simultaneously (flyability, expected XC score, expected flight duration)
  - Handles missing data gracefully through probabilistic inference
  - Can incorporate domain knowledge through network structure -> main benefit for sparse data with respect to deep learning
* **Social factor modeling**: Accounts for weekday/weekend effects and pilot skill levels explicitly
* **Three-tier hierarchical approach**:
  1. **Regional prior model**: Uses general weather patterns across all sites in northern Germany
     - Learns baseline relationships between weather and flying
     - Provides robust estimates even with limited site-specific data
  2. **Site-specific model**: Incorporates local characteristics (orientation, terrain, typical conditions)
     - Updates regional priors with site-specific evidence
     - Produces generic forecasts for any pilot at the site
  3. **Personalized forecast**: Conditions on pilot-specific information when available
     - Takes into account pilot skill level (affects flight duration and XC score predictions)
     - Adjusts for pilot preferences and typical flying behavior
* **Inference**: Uses Variable Elimination algorithm for computing conditional probability distributions
* **Output**: Provides probability distributions over discrete outcomes rather than point predictions (e.g., P(Flyable|weather), P(XC_Score_Category|weather, site, pilot))
* **Discretization**: Continuous weather variables are binned into interpretable categories (e.g., wind speed: Low/Medium/High) to enable tractable probabilistic inference
* **Disadvantage**: Physical parameters need careful discretization (handling continuous random variables is more complicated)

**TODO**: Add Bayesian Network structure graph showing nodes and edges