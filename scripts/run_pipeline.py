# ==============================================================================
# End-to-End Pipeline Execution Entry Point
# ==============================================================================

import hydra
from omegaconf import DictConfig
from src.data.dataset import process_and_save_data
from src.utils.seeding import set_seed

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # 1. Set global seed for reproducibility
    set_seed(cfg.seed)
    
    # 2. Retrieve paths and parameters from configuration
    raw_data_path = cfg.paths.raw_data_file
    processed_dir = "data/processed"
    scaler_save_path = "artifacts/models/scaler.pkl"
    
    # 3. Execute data preprocessing pipeline
    print(f"[PIPELINE] Starting data processing pipeline using seed={cfg.seed}...")
    process_and_save_data(
        raw_data_path=raw_data_path,
        processed_dir=processed_dir,
        scaler_save_path=scaler_save_path,
        test_size=0.2,  # 80/20 train/test split on unique cycle IDs
        random_state=cfg.seed,
        sequence_length=cfg.model.sequence_length
    )
    print("[PIPELINE] Data processing complete. Padded sequences and scaling variables are stored.")

if __name__ == "__main__":
    main()
