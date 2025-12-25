# tools/weather/constants.py

MIN_DATE = '2018-01-01'

URL_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"  # PRE 2022-01-01 - does not include pressure level data
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"  # FROM 2022-01-01, includes pressure level data


TIME_ZONE = "Europe/Berlin"
START_HOUR = 10
END_HOUR = 18

HOURLY_PARAMS_ARCHIVE = [
        "temperature_2m",
        "dew_point_2m",
        "precipitation",
        "weather_code",
        "cloud_cover_low",
        "surface_pressure", 
        "wind_speed_10m", 
        "wind_direction_10m", 
        "wind_gusts_10m", 
        "sunshine_duration",
        "boundary_layer_height",
        "direct_radiation",
        "diffuse_radiation"
]

HOURLY_PARAMS = ["temperature_2m", # in archive
                 "dew_point_2m", # in archive
                 "precipitation", # in archive
                 "weather_code", # in archive
                 "cloud_cover_low",   # Detects Stratus/Fog ,  in archive              
                 "surface_pressure",  # in archive
                 "wind_speed_10m",  # in archive
                 "wind_direction_10m",  # in archive
                 "wind_gusts_10m",  # in archive
                 "sunshine_duration",  # in archive
                 "boundary_layer_height",  # in archive
                 "direct_radiation",  # in archive
                 "diffuse_radiation", #  Detects Cirrus/Haze ,  in archive 
                 "cape", 
                 "lifted_index",                  
                 "temperature_950hPa",
                 "temperature_850hPa",
                 "wind_speed_950hPa",
                 "wind_direction_950hPa",             
                 "wind_speed_850hPa", 
                 "wind_direction_850hPa", 
                 "geopotential_height_850hPa"]