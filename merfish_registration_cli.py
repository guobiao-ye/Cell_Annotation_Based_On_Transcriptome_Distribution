import os
import time
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

def parse_boundary_series(series):
    """Parses a Pandas Series containing comma-separated coordinate strings into a Series of numerical lists."""
    import warnings
    warnings.filterwarnings('ignore', 'elementwise comparison failed')
    
    parsed_series = series.copy()
    non_na_mask = series.notna()
    parsed_series.loc[non_na_mask] = series[non_na_mask].apply(
        lambda x: [float(p) for p in x.split(', ')] if isinstance(x, str) else np.nan
    )
    return parsed_series

def create_polygon(row, z_slice):
    """Creates a Shapely Polygon object from row data."""
    x_coords = row[f'boundaryX_z{z_slice}_parsed']
    y_coords = row[f'boundaryY_z{z_slice}_parsed']
    if isinstance(x_coords, list) and isinstance(y_coords, list) and len(x_coords) > 2:
        return Polygon(zip(x_coords, y_coords))
    return None

# *** New helper function: Calculate global centroid ***
def calculate_global_centroid(row, available_z_slices):
    """
    Calculates the global centroid for a single cell (row) across all available Z-slices.
    """
    all_x_coords = []
    all_y_coords = []
    
    for z in available_z_slices:
        x_coords = row.get(f'boundaryX_z{z}_parsed')
        y_coords = row.get(f'boundaryY_z{z}_parsed')
        
        if isinstance(x_coords, list):
            all_x_coords.extend(x_coords)
        if isinstance(y_coords, list):
            all_y_coords.extend(y_coords)
            
    if not all_x_coords:  # If a cell has no boundary points
        return pd.Series([np.nan, np.nan], index=['center_x_global', 'center_y_global'])
        
    # Returns a Pandas Series containing the global center X and Y
    return pd.Series([np.mean(all_x_coords), np.mean(all_y_coords)], index=['center_x_global', 'center_y_global'])


def main():
    """
    Main function to process, register, and normalize MERFISH data using the CPU.
    Normalization is performed using the aggregated center of all Z-planes.
    """
    # ... (argparse section remains unchanged) ...
    parser = argparse.ArgumentParser(description="Register and normalize MERFISH gene spot and cell boundary data.")
    parser.add_argument('--spots', required=True, help='Path to the input gene spots (spots) CSV file.')
    parser.add_argument('--boundaries', required=True, help='Path to the input cell boundaries (boundaries) CSV file.')
    parser.add_argument('--output', required=True, help='Path to the output normalized results CSV file.')
    args = parser.parse_args()

    spots_file = args.spots
    boundaries_file = args.boundaries
    output_file = args.output

    print(f"--- Starting MERFISH data processing (CPU-only version) ---")
    print(f"  - Gene spots file: {os.path.basename(spots_file)}")
    print(f"  - Cell boundaries file: {os.path.basename(boundaries_file)}")
    print(f"  - Output file: {os.path.basename(output_file)}")
    start_time = time.time()
    
    # ... (data loading section remains unchanged) ...
    print(f"Step 1/5: Loading cell boundaries file...")
    boundaries_df_pd = pd.read_csv(boundaries_file, index_col=0)
    boundaries_df_pd.index.name = 'cell_id'
    
    print(f"Step 2/5: Loading gene spots file...")
    spots_df_pd = pd.read_csv(spots_file)
    print(f"Successfully loaded {len(spots_df_pd)} gene spots.")
    
    # --- 3. Preprocess cell boundary data and calculate global centers ---
    print("Step 3/5: Parsing cell boundary coordinates and calculating global centers...")
    
    z_cols = [col for col in boundaries_df_pd.columns if 'boundaryX_z' in col]
    if not z_cols:
        print("Error: No 'boundaryX_z*' columns found in the boundaries file."); return
    available_z_slices = sorted([int(c.split('z')[-1]) for c in z_cols])
    print(f"  - Detected available Z-slices: {available_z_slices}")

    for z in available_z_slices:
        boundaries_df_pd[f'boundaryX_z{z}_parsed'] = parse_boundary_series(boundaries_df_pd[f'boundaryX_z{z}'])
        boundaries_df_pd[f'boundaryY_z{z}_parsed'] = parse_boundary_series(boundaries_df_pd[f'boundaryY_z{z}'])

    # *** Key change: Calculate global centroid ***
    print("  - Calculating the global center for each cell using aggregated points from all Z-slices...")
    # Use apply and our new helper function to calculate the global center for each row
    global_centers = boundaries_df_pd.apply(
        calculate_global_centroid, 
        args=(available_z_slices,), 
        axis=1
    )
    # Merge the calculated centers back into the main DataFrame
    boundaries_df_pd = pd.concat([boundaries_df_pd, global_centers], axis=1)

    # --- 4. Iterate through Z-slices for spatial registration ---
    print("Step 4/5: Performing spatial registration on the CPU...")
    all_assigned_spots = []

    for z_slice in available_z_slices:
        print(f"  - Processing Z-slice: {z_slice}")
        
        spots_z = spots_df_pd[spots_df_pd['global_z'] == float(z_slice)].copy()
        if len(spots_z) == 0:
            print(f"    No gene spots found at Z={z_slice}, skipping."); continue
        
        spots_gdf = gpd.GeoDataFrame(spots_z, geometry=gpd.points_from_xy(spots_z.global_x, spots_z.global_y))
        
        valid_boundaries = boundaries_df_pd.dropna(subset=[f'boundaryX_z{z_slice}_parsed', 'center_x_global']).copy()
        if len(valid_boundaries) == 0:
            print(f"    No cell boundaries found at Z={z_slice}, skipping."); continue
            
        valid_boundaries['geometry'] = valid_boundaries.apply(create_polygon, args=(z_slice,), axis=1)
        valid_boundaries = valid_boundaries.dropna(subset=['geometry'])
        
        if valid_boundaries.empty:
            print(f"    No valid polygons at Z={z_slice}, skipping."); continue
            
        boundaries_gdf = gpd.GeoDataFrame(valid_boundaries, geometry='geometry')
        # During the spatial join, the 'center_x_global' and 'center_y_global' columns will be automatically carried over
        assigned_gdf = gpd.sjoin(spots_gdf, boundaries_gdf, how="inner", predicate="within")
        
        print(f"    At Z={z_slice}, found {len(assigned_gdf)} gene spots within cells.")
        all_assigned_spots.append(assigned_gdf)

    if not all_assigned_spots:
        print("Warning: No gene spots were found within cells across all Z-slices. The output will be empty.")
        pd.DataFrame(columns=['cell_id', 'target_gene', 'normalized_x', 'normalized_y', 'z']).to_csv(output_file, index=False)
        print(f"An empty output file has been generated: '{output_file}'"); return

    # --- 5. Merge, normalize, and save the results ---
    print("Step 5/5: Merging, normalizing, and saving the results...")

    final_df_pd = pd.concat(all_assigned_spots, ignore_index=True)
    
    # Normalize using the global centers
    final_df_pd['normalized_x'] = final_df_pd['global_x'] - final_df_pd['center_x_global']
    final_df_pd['normalized_y'] = final_df_pd['global_y'] - final_df_pd['center_y_global']
    
    output_df_pd = final_df_pd[[
        'cell_id', 'target_gene', 'normalized_x', 'normalized_y', 'global_z'
    ]].rename(columns={'global_z': 'z'})
    
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    output_df_pd.to_csv(output_file, index=False)
    
    end_time = time.time()
    print(f"\n--- Processing complete ---")
    print(f"A total of {len(output_df_pd)} gene spots were successfully assigned to cells and normalized.")
    print(f"The results have been saved to '{output_file}'.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()