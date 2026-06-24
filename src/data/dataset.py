# ==============================================================================
# Dataset Formatting and Execution Pipeline Module
# ==============================================================================

import os
import pandas as pd
from src.data.preprocessing import clean_data, split_and_scale_data
from src.data.sequences import segment_and_downsample, create_sequences, save_sequences

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
    
    save_sequences(
        train_seqs, test_seqs, 
        train_conds, test_conds, 
        processed_dir=processed_dir
    )
    print(f"          Train Sequences: {train_seqs.shape}, Train Conditioning: {train_conds.shape}")
    print(f"          Test Sequences: {test_seqs.shape}, Test Conditioning: {test_conds.shape}")
