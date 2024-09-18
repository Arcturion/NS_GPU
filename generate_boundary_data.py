import xarray as xr
import gcsfs
import numpy as np

def load_era5_data(start_date, end_date):
    gcs = gcsfs.GCSFileSystem(token='anon')
    path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
    era5 = xr.open_zarr(gcs.get_mapper(path), chunks={'time': 'auto'})
    return era5.sel(time=slice(start_date, end_date))

def extract_variables(era5):
    variables_single_level = [
        '100m_u_component_of_wind', '100m_v_component_of_wind',
        '2m_temperature', '2m_dewpoint_temperature', 
        'surface_pressure'
    ]
    variables_multilevel = [
        'u_component_of_wind', 'v_component_of_wind',
        'temperature', 'specific_humidity', 'geopotential'
    ]
    initial_condition_single = era5[variables_single_level]
    initial_condition_multi = era5[variables_multilevel]
    return xr.merge([initial_condition_single, initial_condition_multi])

def crop_to_pacific(initial_conditions):
    return initial_conditions.sel(latitude=slice(40, 0), longitude=slice(120, 180))

def prepare_initial_state(initial_conditions_pacific):
    initial_state = initial_conditions_pacific.isel(time=0)
    return {
        # Multi-level variables
        'u': initial_state['u_component_of_wind'].values,
        'v': initial_state['v_component_of_wind'].values,
        'temperature': initial_state['temperature'].values,
        'specific_humidity': initial_state['specific_humidity'].values,
        'geopotential': initial_state['geopotential'].values,
        # Single-level variables
        'u_100m': initial_state['100m_u_component_of_wind'].values,
        'v_100m': initial_state['100m_v_component_of_wind'].values,
        'temperature_2m': initial_state['2m_temperature'].values,
        'dewpoint_2m': initial_state['2m_dewpoint_temperature'].values,
        'surface_pressure': initial_state['surface_pressure'].values
    }, initial_state.latitude.values, initial_state.longitude.values, initial_state.level.values

def calculate_relative_vorticity(u, v, lat, lon):
    dx = np.gradient(lon) * np.pi / 180 * 6371000 * np.cos(lat * np.pi / 180)
    dy = np.gradient(lat) * np.pi / 180 * 6371000
    dudy, dudx = np.gradient(u)
    dvdy, dvdx = np.gradient(v)
    return dvdx / dx[:, np.newaxis] - dudy / dy[:, np.newaxis]

def calculate_coriolis_parameter(lat):
    omega = 7.2921e-5  # Earth's angular velocity in rad/s
    return 2 * omega * np.sin(lat * np.pi / 180)

def calculate_vorticity(initial_conditions, lat, lon):
    def calc_relative_vorticity(u, v):
        # Create 2D meshgrid from 1D lat and lon
        lon_2d, lat_2d = np.meshgrid(lon, lat)
        
        dx = np.gradient(lon_2d, axis=1) * np.pi / 180 * 6371000 * np.cos(lat_2d * np.pi / 180)
        dy = np.gradient(lat_2d, axis=0) * np.pi / 180 * 6371000
        dudy, dudx = np.gradient(u)
        dvdy, dvdx = np.gradient(v)
        return dvdx / dx - dudy / dy

    coriolis_parameter = calculate_coriolis_parameter(lat)

    # Calculate vorticity for multi-level wind
    relative_vorticity_3d = np.array([calc_relative_vorticity(initial_conditions['u'][level], 
                                                              initial_conditions['v'][level]) 
                                      for level in range(initial_conditions['u'].shape[0])])
    absolute_vorticity_3d = relative_vorticity_3d + coriolis_parameter[np.newaxis, :, np.newaxis]

    # Calculate vorticity for 100m wind
    relative_vorticity_100m = calc_relative_vorticity(initial_conditions['u_100m'], 
                                                      initial_conditions['v_100m'])
    absolute_vorticity_100m = relative_vorticity_100m + coriolis_parameter[:, np.newaxis]

    return {
        'relative_vorticity_3d': relative_vorticity_3d,
        'absolute_vorticity_3d': absolute_vorticity_3d,
        'relative_vorticity_100m': relative_vorticity_100m,
        'absolute_vorticity_100m': absolute_vorticity_100m,
        'coriolis_parameter': coriolis_parameter
    }

def save_initial_conditions(filename, initial_conditions, lat, lon, levels):
    np.savez(filename, **initial_conditions, lat=lat, lon=lon, levels=levels)

def main():
    era5 = load_era5_data('2023-01-01', '2023-01-02')
    initial_conditions = extract_variables(era5)
    initial_conditions_pacific = crop_to_pacific(initial_conditions)
    initial_conditions, lat, lon, levels = prepare_initial_state(initial_conditions_pacific)
    
    vorticity_components = calculate_vorticity(initial_conditions, lat, lon)
    
    initial_conditions.update(vorticity_components)
    
    save_initial_conditions('pacific_initial_conditions.npz', initial_conditions, lat, lon, levels)

if __name__ == "__main__":
    main()
