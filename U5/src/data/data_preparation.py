import pandas as pd
import numpy as np
import os

# ── Data Paths ──
# New path points to the root directory dataset
CRIME_DATA_PATH = os.getenv("CRIME_DATA_PATH", "data/Crimes_in_india_2001-2013.csv")
OUTPUT_DATA_PATH = "cleaned_crime_features.csv"

# Top 5 crime types based on Unit IV analysis (Matching CSV Column Names)
TOP_CRIMES = [
    "THEFT", 
    "HURT/GREVIOUS HURT", 
    "OTHER THEFT", 
    "AUTO THEFT", 
    "BURGLARY"
]

def preprocess_crime_data(input_csv, output_csv):
    """
    Reads the wide-format NCRB dataset, filters for top crime types, 
    and aggregates them into a DISTRICT-level feature matrix.
    """
    if not os.path.exists(input_csv):
        # Fallback to local data/ if root is not accessible in container
        input_csv = "data/Crimes_in_india_2001-2013.csv"
        if not os.path.exists(input_csv):
            print(f"Error: {input_csv} not found.")
            return False

    print(f"Loading expanded wide-format data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # ── FILTER OUT SUMMARY ROWS ──
    # NCRB data contains 'TOTAL' rows for states which skew the clustering
    exclude_list = ['TOTAL', 'DELHI UT TOTAL', 'DISTRICT TOTAL', 'GRAND TOTAL']
    df = df[~df['DISTRICT'].str.upper().isin(exclude_list)]
    df = df[~df['DISTRICT'].str.contains('TOTAL', case=False, na=False)]
    
    # ── AGGREGATION ──
    print(f"Aggregating {len(TOP_CRIMES)} core features by District...")
    
    # Select ID columns and the top 5 crime columns
    cols_to_keep = ['STATE/UT', 'DISTRICT'] + TOP_CRIMES
    df_filtered = df[cols_to_keep]
    
    # Group by State + District and sum across years
    features_df = df_filtered.groupby(['STATE/UT', 'DISTRICT']).sum().reset_index()
    
    print(f"Saving district feature matrix to {output_csv} (Districts: {len(features_df)})")
    features_df.to_csv(output_csv, index=False)
    return True

if __name__ == "__main__":
    preprocess_crime_data(CRIME_DATA_PATH, OUTPUT_DATA_PATH)
