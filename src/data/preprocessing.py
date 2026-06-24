# ==============================================================================
# Data Preprocessing Module (Clean, Segment, Split, and Scale)
# ==============================================================================

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans State of Charge (SOC) values by clipping to a minimum of 0
    and imputes zero capacities by using the mean capacity of the same Dataset.

    Args:
        df (pd.DataFrame): Raw input battery dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = df.copy()
    
    # 1. Clean SOC
    df['SOC'] = df['SOC'].clip(lower=0.0)
    
    # 2. Impute zero capacity
    zero_capacity_mask = df['capacity'] == 0
    mean_capacity = df[~zero_capacity_mask].groupby('Dataset')['capacity'].mean()
    dataset_means = df['Dataset'].map(mean_capacity)
    df.loc[zero_capacity_mask, 'capacity'] = dataset_means[zero_capacity_mask]
    
    return df

def segment_and_downsample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segments raw battery telemetry by car and charge_segment, sorts by Timestamp,
    downsamples (taking every 3rd row, e.g. 30s intervals), creates a unique cycle_id,
    and fixes inverted Min/Max Cell Temperature values.

    Args:
        df (pd.DataFrame): Cleaned input battery dataframe.

    Returns:
        pd.DataFrame: Segmented and downsampled dataframe.
    """
    grouped = df.groupby(['car', 'charge_segment'])
    cycle_segments = []
    
    for (car_id, segment_id), group in grouped:
        # Sort chronologically
        group_sorted = group.sort_values('Timestamp')
        # Downsample: take every 3rd point (30s intervals from 10s)
        downsampled = group_sorted.iloc[::3].copy()
        downsampled['cycle_id'] = f"{car_id}_{segment_id}"
        cycle_segments.append(downsampled)
        
    df_segmented = pd.concat(cycle_segments, ignore_index=True)
    
    # Flip temperature values in-place if Min_Cell_Temp > Max_Cell_Temp
    swap_mask = df_segmented["Min_Cell_Temperature"] > df_segmented["Max_Cell_Temperature"]
    if swap_mask.any():
        df_segmented.loc[swap_mask, ["Max_Cell_Temperature", "Min_Cell_Temperature"]] = (
            df_segmented.loc[swap_mask, ["Min_Cell_Temperature", "Max_Cell_Temperature"]].values
        )
        
    return df_segmented

def split_and_scale_data(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42,
    scaler_save_path: str = "artifacts/models/scaler.pkl"
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Splits the dataframe into training and testing sets based on unique cycle_ids,
    fits a MinMaxScaler ONLY on the training split, and transforms both splits to
    resolve data leakage.

    Args:
        df (pd.DataFrame): Segmented and cleaned dataframe.
        test_size (float): Proportion of cycle IDs to use for testing. Defaults to 0.2.
        random_state (int): Seed for split reproducibility. Defaults to 42.
        scaler_save_path (str): File path to save the fitted MinMaxScaler.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]: Normalized df_train, df_test, and the scaler.
    """
    columns_to_normalize = [
        'Average_Cell_Voltage', 'Charging_Current', 'Max_Cell_Voltage', 'Min_Cell_Voltage',
        'Max_Cell_Temperature', 'Min_Cell_Temperature', 'SOC', 'Timestamp',
        'mileage', 'capacity'
    ]
    
    # 1. Split unique cycle IDs to prevent sequence-level data leakage
    unique_cycle_ids = df['cycle_id'].unique()
    train_ids, test_ids = train_test_split(
        unique_cycle_ids, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Filter rows based on splits
    df_train = df[df['cycle_id'].isin(train_ids)].copy()
    df_test = df[df['cycle_id'].isin(test_ids)].copy()
    
    # 2. Fit MinMaxScaler ONLY on the training split
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # Fit and transform training
    df_train[columns_to_normalize] = scaler.fit_transform(df_train[columns_to_normalize])
    
    # Transform test
    df_test[columns_to_normalize] = scaler.transform(df_test[columns_to_normalize])
    
    # Save the scaler
    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    
    print(f"[PREPROCESSING] Scaler saved to {scaler_save_path}")
    print(f"[PREPROCESSING] Train size: {len(df_train)} rows ({len(train_ids)} cycles)")
    print(f"[PREPROCESSING] Test size: {len(df_test)} rows ({len(test_ids)} cycles)")
    
    return df_train, df_test, scaler
