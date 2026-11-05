import os
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_samples(extracted_path=None):
    """
    Helper to set up sample videos for the prototype.
    If extracted_path is provided, it copies a few samples from VidLLVIP.
    """
    samples_dir = Path("data/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    if extracted_path:
        extracted_path = Path(extracted_path)
        ir_root = extracted_path / "dataset" / "ir"
        vi_root = extracted_path / "dataset" / "vi"
        
        if not ir_root.exists() or not vi_root.exists():
            logger.error(f"Could not find dataset/ir or dataset/vi in {extracted_path}")
            return
            
        # Copy the first 2 pairs
        ir_files = sorted(list(ir_root.glob("*.mp4")))[:2]
        for ir_file in ir_files:
            vi_file = vi_root / ir_file.name
            if vi_file.exists():
                shutil.copy(ir_file, samples_dir / "thermal_sample.mp4") # Simple naming for config
                shutil.copy(vi_file, samples_dir / "optical_sample.mp4")
                logger.info(f"Copied sample pair: {ir_file.name}")
                break # Just need one pair for MVP
    else:
        logger.info("No extracted path provided. Please download the VidLLVIP dataset from Hugging Face:")
        logger.info("1. Go to https://huggingface.co/datasets/jianfeng0369/VidLLVIP")
        logger.info("2. Download 'dataset.zip'")
        logger.info("3. Extract it to a folder")
        logger.info("4. Run this script with: python datasets/download_samples.py --path /path/to/extracted_folder")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to extracted VidLLVIP dataset")
    args = parser.parse_args()
    setup_samples(args.path)
