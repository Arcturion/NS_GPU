import numpy as np
from scipy.interpolate import interp1d

def load_full_data(filename):
    """Load the full domain data saved by boundary.py"""
    return np.load(filename)

def extract_boundaries(data):
    """Extract boundary data from full domain data"""
    if data.ndim == 3:  # time, lat, lon
        return {
            'north': data[:, 0, :],
            'south': data[:, -1, :],
            'east': data[:, :, -1],
            'west': data[:, :, 0]
        }
    elif data.ndim == 4:  # time, level, lat, lon
        return {
            'north': data[:, :, 0, :],
            'south': data[:, :, -1, :],
            'east': data[:, :, :, -1],
            'west': data[:, :, :, 0]
        }
    else:
        raise ValueError(f"Dimensions must be 3 or 4, but got: {data.ndim}")

def interpolate_time(data, old_times, new_times):
    """Interpolate data in time using linear interpolation"""
    interp_func = interp1d(old_times, data, axis=0, kind='linear', fill_value='extrapolate')
    return interp_func(new_times)

def interpolate_space(data, old_coords, new_coords):
    """Interpolate data in space using linear interpolation"""
    if data.ndim == 1:  # 1D array (e.g., for north or south boundary)
        interp_func = interp1d(old_coords, data, kind='linear', fill_value='extrapolate')
        return interp_func(new_coords)
    elif data.ndim == 2:  # 2D array (e.g., for east or west boundary)
        return np.array([interpolate_space(data[:, i], old_coords[0], new_coords[0]) for i in range(data.shape[1])])
    elif data.ndim == 3:  # 3D array (e.g., for time-varying boundary)
        return np.array([interpolate_space(data[i], old_coords, new_coords) for i in range(data.shape[0])])
    else:
        raise ValueError(f"Unexpected number of dimensions: {data.ndim}")

def process_boundary_data(input_file, output_file, new_timestep, spatial_scale):
    """Process the full domain data, interpolate, and save only the boundary data"""
    full_data = load_full_data(input_file)
    
    # Prepare new time and space coordinates
    old_times = full_data['times']
    new_times = np.arange(old_times[0], old_times[-1], new_timestep)
    
    old_lat, old_lon = full_data['lat'], full_data['lon']
    new_lat = np.linspace(old_lat[0], old_lat[-1], len(old_lat) * spatial_scale)
    new_lon = np.linspace(old_lon[0], old_lon[-1], len(old_lon) * spatial_scale)
    
    boundary_conditions = {}
    for var in full_data.files:
        if var not in ['lat', 'lon', 'levels', 'times']:
            # Extract boundaries
            boundaries = extract_boundaries(full_data[var])
            
            # Interpolate in time and space
            interpolated_boundaries = {}
            for direction, boundary_data in boundaries.items():
                time_interpolated = interpolate_time(boundary_data, old_times, new_times)
                if direction in ['north', 'south']:
                    space_interpolated = interpolate_space(time_interpolated, old_lon, new_lon)
                else:  # east or west
                    space_interpolated = interpolate_space(time_interpolated, old_lat, new_lat)
                interpolated_boundaries[direction] = space_interpolated
            
            boundary_conditions[var] = interpolated_boundaries
    
    # Save the processed and interpolated boundary data
    np.savez(output_file, 
             **boundary_conditions,
             lat=new_lat,
             lon=new_lon,
             levels=full_data['levels'],
             times=new_times)

    print(f"Processed and interpolated boundary data saved to {output_file}")

if __name__ == "__main__":
    input_file = 'pacific_boundary_conditions.npz'  # Output from boundary.py
    output_file = 'processed_pacific_boundary_conditions.npz'  # Ready for model use
    new_timestep = 120  # 2 minutes in seconds
    spatial_scale = 3  # 3x finer resolution
    process_boundary_data(input_file, output_file, new_timestep, spatial_scale)
