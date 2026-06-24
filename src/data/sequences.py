import os
import numpy as np
import pandas as pd
from typing import List, Tuple

# Default feature list. Can be overridden via configuration.
DEFAULT_FEATURES = [
    'Average_Cell_Voltage', 'Charging_Current', 'Max_Cell_Voltage', 'Min_Cell_Voltage',
    'Max_Cell_Temperature', 'Min_Cell_Temperature', 'SOC', 'Timestamp',
    'mileage', 'capacity'
]

def segment_and_downsample(df: pd.DataFrame, downsample_rate: int = 3) -> pd.DataFrame:
    """
    Segments raw battery telemetry by car and charge_segment, sorts by Timestamp,
    downsamples, creates a unique cycle_id, and fixes inverted Min/Max Cell Temperature values.

    Args:
        df (pd.DataFrame): Cleaned input battery dataframe.
        downsample_rate (int): Rate for downsampling (e.g., 3 means taking every 3rd row).

    Returns:
        pd.DataFrame: Segmented and downsampled dataframe.
    """
    grouped = df.groupby(['car', 'charge_segment'])
    cycle_segments = []
    
    for (car_id, segment_id), group in grouped:
        group_sorted = group.sort_values('Timestamp')
        # Downsample
        downsampled = group_sorted.iloc[::downsample_rate].copy()
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

def create_sequences(
    df: pd.DataFrame, 
    features: List[str] = DEFAULT_FEATURES, 
    sequence_length: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a dataframe into 3D padded sequences and extracts conditioning variables.
    
    Args:
        df (pd.DataFrame): Input dataframe containing 'cycle_id' and features.
        features (List[str]): List of column names to include in the sequence features.
        sequence_length (int): Fixed sequence length for padding/truncating.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - 3D padded sequences array (samples, sequence_length, num_features)
            - 2D conditioning arrays (capacity) (samples, 1)
    """
    grouped = df.groupby('cycle_id')
    sequences = []
    conditioning_list = []
    
    # Group keys are sorted automatically by pandas groupby, ensuring reproducible order
    for _, group in grouped:
        # Sort chronologically by Timestamp to maintain order
        group_sorted = group.sort_values('Timestamp')
        seq = group_sorted[features].to_numpy()
        sequences.append(seq)
        
        # Smoothed capacity conditioning array (taking the last capacity value per cycle)
        conditioning_list.append(group_sorted['capacity'].iloc[-1])
        
    if not sequences:
        return (np.zeros((0, sequence_length, len(features)), dtype=np.float32), 
                np.zeros((0, 1), dtype=np.float32))
        
    num_features = sequences[0].shape[1]
    padded_sequences = np.zeros((len(sequences), sequence_length, num_features), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        if seq_len >= sequence_length:
            padded_sequences[i] = seq[:sequence_length]
        else:
            pad = np.tile(seq[-1], (sequence_length - seq_len, 1))
            padded_sequences[i] = np.vstack([seq, pad])
            
    conditioning_values = np.array(conditioning_list, dtype=np.float32).reshape(-1, 1)
    
    return padded_sequences, conditioning_values

def save_sequences(
    train_seqs: np.ndarray, test_seqs: np.ndarray,
    train_conds: np.ndarray, test_conds: np.ndarray,
    processed_dir: str = "data/processed/"
) -> None:
    """
    Saves the 3D padded sequences and conditioning arrays to .npy files.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    np.save(os.path.join(processed_dir, "train_sequences.npy"), train_seqs)
    np.save(os.path.join(processed_dir, "test_sequences.npy"), test_seqs)
    np.save(os.path.join(processed_dir, "train_conditioning.npy"), train_conds)
    np.save(os.path.join(processed_dir, "test_conditioning.npy"), test_conds)
    
    print(f"[SEQUENCES] Saved padded sequence arrays to {processed_dir}")
