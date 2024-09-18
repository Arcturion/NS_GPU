import numpy as np

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
        raise ValueError(f"dimensi harus 3 atau 4, tapi yang ini kok begini bray: {data.ndim}")

def process_boundary_data(input_file, output_file):
    """Process the full domain data and save only the boundary data"""
    full_data = load_full_data(input_file)
    
    boundary_conditions = {}
    for var in full_data.files:
        if var not in ['lat', 'lon', 'levels', 'times']:
            boundary_conditions[var] = extract_boundaries(full_data[var])
    
    # Save the processed boundary data
    np.savez(output_file, 
             **boundary_conditions,
             lat=full_data['lat'],
             lon=full_data['lon'],
             levels=full_data['levels'],
             times=full_data['times'])

    print(f"Processed boundary data ude disimpan di {output_file}")

if __name__ == "__main__":
    input_file = 'pacific_boundary_conditions.npz'  # Output from boundary.py
    output_file = 'processed_pacific_boundary_conditions.npz'  # Ready for model use
    process_boundary_data(input_file, output_file)
