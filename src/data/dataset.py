# ==============================================================================
# Dataset Formatting and Execution Pipeline Module
# ==============================================================================

import os
import numpy as np
import pandas as pd
from typing import Tuple
from src.data.preprocessing import clean_data, segment_and_downsample, split_and_scale_data

def create_sequences(df: pd.DataFrame, sequence_length: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a normalized/scaled battery telemetry dataframe into 3D padded sequences
    and extracts conditioning variables (the capacity of the cycle).

    Args:
        df (pd.DataFrame): Input dataframe. Must contain 'cycle_id' and the 10 core features.
        sequence_length (int): Fixed size of output sequences. Defaults to 64.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Padded sequences (shape: [N, sequence_length, 10])
                                       and conditioning variables (shape: [N, 1]).
    """
    input_features = [
        'Average_Cell_Voltage', 'Charging_Current', 'Max_Cell_Voltage', 'Min_Cell_Voltage',
        'Max_Cell_Temperature', 'Min_Cell_Temperature', 'SOC', 'Timestamp',
        'mileage', 'capacity'
    ]
    
    grouped = df.groupby('cycle_id')
    sequences = []
    conditioning_list = []
    
    # Group keys are sorted automatically by pandas groupby, ensuring reproducible order
    for _, group in grouped:
        # Sort chronologically by Timestamp to maintain order
        group_sorted = group.sort_values('Timestamp')
        seq = group_sorted[input_features].to_numpy()
        sequences.append(seq)
        # Conditioning variable is the final capacity of each cycle
        conditioning_list.append(group_sorted['capacity'].iloc[-1])
        
    if not sequences:
        return (np.zeros((0, sequence_length, len(input_features)), dtype=np.float32), 
                np.zeros((0, 1), dtype=np.float32))
        
    num_features = sequences[0].shape[1]
    padded_sequences = np.zeros((len(sequences), sequence_length, num_features), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        if seq_len >= sequence_length:
            # Truncate
            padded_sequences[i] = seq[:sequence_length]
        else:
            # Pad by repeating the last timestep's values
            pad = np.tile(seq[-1], (sequence_length - seq_len, 1))
            padded_sequences[i] = np.vstack([seq, pad])
            
    conditioning_values = np.array(conditioning_list, dtype=np.float32).reshape(-1, 1)
    
    return padded_sequences, conditioning_values

def process_and_save_data(
    raw_data_path: str,
    processed_dir: str,
    scaler_save_path: str = "artifacts/models/scaler.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
    sequence_length: int = 64
) -> None:
    """
    Executes the complete data processing pipeline:
      1. Loads raw battery CSV telemetry data.
      2. Cleans raw data (SOC clipping and capacity imputation).
      3. Segments by car/charge_segment, downsamples, and swaps temp values.
      4. Splits train/test on unique cycle IDs and scales variables.
      5. Saves the train and test dataframes as CSVs.
      6. Converts the datasets into padded 3D sequences and conditioning vectors.
      7. Writes all outputs (CSVs and NumPy arrays) to the processed directory.

    Args:
        raw_data_path (str): Path to raw CSV data.
        processed_dir (str): Directory where outputs will be saved.
        scaler_save_path (str): File path to save the fitted MinMaxScaler.
        test_size (float): Proportion of cycle IDs to use for testing. Defaults to 0.2.
        random_state (int): Seed for split reproducibility. Defaults to 42.
        sequence_length (int): Fixed size of output sequences. Defaults to 64.
    """
    print(f"[DATASET] Loading raw data from: {raw_data_path}")
    df_raw = pd.read_csv(raw_data_path)
    
    print("[DATASET] Cleaning data...")
    df_cleaned = clean_data(df_raw)
    
    print("[DATASET] Segmenting and downsampling...")
    df_segmented = segment_and_downsample(df_cleaned)
    
    print("[DATASET] Splitting train/test by cycle IDs and scaling...")
    df_train, df_test, scaler = split_and_scale_data(
        df_segmented,
        test_size=test_size,
        random_state=random_state,
        scaler_save_path=scaler_save_path
    )
    
    # Ensure processed directory exists
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save processed CSVs
    train_csv_path = os.path.join(processed_dir, "train_data.csv")
    test_csv_path = os.path.join(processed_dir, "test_data.csv")
    df_train.to_csv(train_csv_path, index=False)
    df_test.to_csv(test_csv_path, index=False)
    print(f"[DATASET] Saved {train_csv_path} and {test_csv_path}")
    
    # Format and save sequences & conditioning variables
    print("[DATASET] Generating 3D sequences...")
    train_seqs, train_conds = create_sequences(df_train, sequence_length=sequence_length)
    test_seqs, test_conds = create_sequences(df_test, sequence_length=sequence_length)
    
    train_seq_path = os.path.join(processed_dir, "train_sequences.npy")
    test_seq_path = os.path.join(processed_dir, "test_sequences.npy")
    train_cond_path = os.path.join(processed_dir, "train_conditioning.npy")
    test_cond_path = os.path.join(processed_dir, "test_conditioning.npy")
    
    np.save(train_seq_path, train_seqs)
    np.save(test_seq_path, test_seqs)
    np.save(train_cond_path, train_conds)
    np.save(test_cond_path, test_conds)
    
    print(f"[DATASET] Saved sequences and conditioning vectors to {processed_dir}")
    print(f"          Train Sequences: {train_seqs.shape}, Train Conditioning: {train_conds.shape}")
    print(f"          Test Sequences: {test_seqs.shape}, Test Conditioning: {test_conds.shape}")
