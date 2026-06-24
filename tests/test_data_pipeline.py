# ==============================================================================
# Unit Tests for Data Preprocessing and Sequence Generation Pipeline
# ==============================================================================

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import joblib
from src.data.preprocessing import clean_data, segment_and_downsample, split_and_scale_data
from src.data.dataset import create_sequences, process_and_save_data

@pytest.fixture
def sample_raw_df():
    # Create sample telemetry data for 2 cars, 2 charge segments each, 12 rows per segment (total 48 rows)
    data = []
    datasets = ["Dataset_A", "Dataset_B"]
    for car_id in [1, 2]:
        for segment_id in [101, 102]:
            for step in range(12):
                data.append({
                    "car": car_id,
                    "charge_segment": segment_id,
                    "Timestamp": step * 10,
                    "Average_Cell_Voltage": 3.7 + 0.01 * step,
                    "Charging_Current": 50.0 - 0.5 * step,
                    "Max_Cell_Voltage": 3.8 + 0.01 * step,
                    "Min_Cell_Voltage": 3.6 + 0.01 * step,
                    # Intentionally invert Min/Max temperature on step 0 to test swap logic
                    "Max_Cell_Temperature": 25.0 if step > 0 else 20.0,
                    "Min_Cell_Temperature": 21.0 if step > 0 else 25.0,
                    "SOC": 10.0 + 2.0 * step if step > 0 else -1.0,  # Negative SOC to test clipping
                    "mileage": 1000 + step * 2,
                    "capacity": 50.0 if (step > 0 or car_id == 2) else 0.0,  # Zero capacity to test imputation
                    "Dataset": datasets[car_id - 1]
                })
    return pd.DataFrame(data)

def test_clean_data(sample_raw_df):
    df_cleaned = clean_data(sample_raw_df)
    
    # Assert negative SOC was clipped to 0
    assert (df_cleaned['SOC'] >= 0.0).all()
    
    # Assert zero capacity was imputed using Dataset_A's mean capacity (which is 50.0)
    assert (df_cleaned['capacity'] > 0.0).all()
    assert (df_cleaned['capacity'] == 50.0).all()

def test_segment_and_downsample(sample_raw_df):
    df_cleaned = clean_data(sample_raw_df)
    df_segmented = segment_and_downsample(df_cleaned)
    
    # Downsampling takes every 3rd row: 12 steps downsamples to 4 rows per segment.
    # Total rows: 2 cars * 2 segments * 4 steps = 16 rows.
    assert len(df_segmented) == 16
    
    # Check cycle_id exists and matches format
    assert 'cycle_id' in df_segmented.columns
    assert set(df_segmented['cycle_id'].unique()) == {"1_101", "1_102", "2_101", "2_102"}
    
    # Check temperature swap logic. We had Max_T=20, Min_T=25 at step 0, which should swap to Max_T=25, Min_T=20.
    # Min temperature should always be <= Max temperature.
    assert (df_segmented["Min_Cell_Temperature"] <= df_segmented["Max_Cell_Temperature"]).all()

def test_split_and_scale_data(sample_raw_df):
    df_cleaned = clean_data(sample_raw_df)
    df_segmented = segment_and_downsample(df_cleaned)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        scaler_path = os.path.join(tmpdir, "scaler.pkl")
        
        # Test split with test_size=0.5 (2 train cycles, 2 test cycles)
        df_train, df_test, scaler = split_and_scale_data(
            df_segmented,
            test_size=0.5,
            random_state=42,
            scaler_save_path=scaler_path
        )
        
        # Verify no data leakage in splits: unique cycle IDs should be disjoint
        train_cycles = set(df_train['cycle_id'].unique())
        test_cycles = set(df_test['cycle_id'].unique())
        
        assert len(train_cycles) == 2
        assert len(test_cycles) == 2
        assert train_cycles.isdisjoint(test_cycles)
        
        # Verify scaler was fit ONLY on train data
        columns_to_normalize = [
            'Average_Cell_Voltage', 'Charging_Current', 'Max_Cell_Voltage', 'Min_Cell_Voltage',
            'Max_Cell_Temperature', 'Min_Cell_Temperature', 'SOC', 'Timestamp',
            'mileage', 'capacity'
        ]
        
        # Verify min/max values in df_train are exactly within [-1, 1] (tolerating floating point limits)
        for col in columns_to_normalize:
            assert df_train[col].min() >= -1.0 - 1e-7
            assert df_train[col].max() <= 1.0 + 1e-7
            
        # Verify scaler file exists and can be loaded
        assert os.path.exists(scaler_path)
        loaded_scaler = joblib.load(scaler_path)
        assert isinstance(loaded_scaler, type(scaler))

def test_create_sequences(sample_raw_df):
    df_cleaned = clean_data(sample_raw_df)
    df_segmented = segment_and_downsample(df_cleaned)
    
    df_train, df_test, _ = split_and_scale_data(
        df_segmented,
        test_size=0.5,
        random_state=42,
        scaler_save_path=os.path.devnull
    )
    
    # Padded sequence size = 6. Each cycle has 4 downsampled steps.
    seqs, conds = create_sequences(df_train, sequence_length=6)
    
    assert seqs.shape == (2, 6, 10)
    assert conds.shape == (2, 1)
    
    # Check padding: last step (index 3) values should be repeated for step 4 and step 5
    for cycle_idx in range(2):
        last_step_val = seqs[cycle_idx, 3, :]
        pad_step_val_1 = seqs[cycle_idx, 4, :]
        pad_step_val_2 = seqs[cycle_idx, 5, :]
        np.testing.assert_array_equal(last_step_val, pad_step_val_1)
        np.testing.assert_array_equal(last_step_val, pad_step_val_2)

def test_process_and_save_data(sample_raw_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_csv_path = os.path.join(tmpdir, "raw_data.csv")
        sample_raw_df.to_csv(raw_csv_path, index=False)
        
        processed_dir = os.path.join(tmpdir, "processed")
        scaler_path = os.path.join(tmpdir, "scaler.pkl")
        
        process_and_save_data(
            raw_data_path=raw_csv_path,
            processed_dir=processed_dir,
            scaler_save_path=scaler_path,
            test_size=0.5,
            random_state=42,
            sequence_length=6
        )
        
        # Verify output files exist
        assert os.path.exists(os.path.join(processed_dir, "train_data.csv"))
        assert os.path.exists(os.path.join(processed_dir, "test_data.csv"))
        assert os.path.exists(os.path.join(processed_dir, "train_sequences.npy"))
        assert os.path.exists(os.path.join(processed_dir, "test_sequences.npy"))
        assert os.path.exists(os.path.join(processed_dir, "train_conditioning.npy"))
        assert os.path.exists(os.path.join(processed_dir, "test_conditioning.npy"))
        assert os.path.exists(scaler_path)
        
        # Load and verify shape of arrays
        train_seq = np.load(os.path.join(processed_dir, "train_sequences.npy"))
        train_cond = np.load(os.path.join(processed_dir, "train_conditioning.npy"))
        assert train_seq.shape == (2, 6, 10)
        assert train_cond.shape == (2, 1)
