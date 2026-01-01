import cdsapi
import xarray as xr
import zipfile
import io

dataset = "reanalysis-era5-pressure-levels"

def download_era5_temperature_850hPa(year: str, output_file: str):
    """Download ERA5 reanalysis temperature data at 850 hPa for a given year.
    
    Args:
        year: Year as a string (e.g., "2018")
        output_file: Path to save the downloaded ZIP file
    """
    client = cdsapi.Client()

    request = {
        "product_type": ["reanalysis"],
        "variable": ["temperature"],
        "year": [
            year
        ],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": [
            "10:00", "11:00", "12:00",
            "13:00", "14:00", "15:00",
            "16:00", "17:00", "18:00"
        ],
        "pressure_level": ["850"],
        "data_format": "netcdf",
        "download_format": "zip",
        "area": [53, 8.5, 51, 10.5]
    }

    client.retrieve(dataset, request).download(output_file)


def extract_temperature_from_era5(year: str) -> dict:    
    """Extract temperature data from ERA5 NetCDF file."""
    zip_file = f"era5_temperature_{year}_850hPa.zip"
    with zipfile.ZipFile(zip_file, 'r') as z:
        # Get the name of the first file in the zip (the .nc file)
        file_name = z.namelist()[0]
        
        # Read the binary data into memory
        with z.open(file_name) as f:
            file_content = f.read()
            
        # Use io.BytesIO to make the binary data look like a file to xarray
        # Note: This requires the 'netcdf4' or 'h5netcdf' engine
        ds = xr.open_dataset(io.BytesIO(file_content), engine='h5netcdf')   

    # Convert to DataFrame
    # df = ds.to_dataframe()   
    #print(df.head())
    #print(df.columns)
    
    return ds

if __name__ == "__main__":
    
    year = "2021"
    
    #ownload_era5_temperature_850hPa(year, output_file)

