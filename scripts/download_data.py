"""
Download and setup data for LTSF-Linear project.
Downloads VIC.csv from Google Drive and organizes data directory.
"""

import os
import sys
from pathlib import Path
import logging
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_data_directories():
    """Create data directory structure."""
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    # Create directories
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created data directories:")
    logger.info(f"  - {raw_dir}")
    logger.info(f"  - {processed_dir}")
    
    return raw_dir, processed_dir


def download_vic_data(output_dir: Path, file_id: str = "18J_Z8b-qMMj9wm5eGyQ-1nPS16PfRePK"):
    """
    Download VIC.csv from Google Drive.
    
    Args:
        output_dir: Directory to save downloaded file
        file_id: Google Drive file ID
    """
    try:
        import gdown
    except ImportError:
        logger.error("gdown not installed. Installing...")
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown
    
    output_path = output_dir / "VIC.csv"
    
    # Check if already exists
    if output_path.exists():
        logger.warning(f"File already exists: {output_path}")
        response = input("Overwrite? (y/n): ").lower()
        if response != 'y':
            logger.info("Skipping download")
            return output_path
    
    # Download from Google Drive
    url = f"https://drive.google.com/uc?id={file_id}"
    logger.info(f"Downloading VIC.csv from Google Drive...")
    logger.info(f"URL: {url}")
    
    try:
        gdown.download(url, str(output_path), quiet=False)
        logger.info(f"✓ Downloaded successfully: {output_path}")
        
        # Verify file
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"✓ File size: {output_path.stat().st_size / 1024:.2f} KB")
        else:
            raise Exception("Downloaded file is empty or not found")
            
    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        raise
    
    return output_path


def verify_data(file_path: Path):
    """
    Verify downloaded data integrity.
    
    Args:
        file_path: Path to CSV file
    """
    import pandas as pd
    
    logger.info("Verifying data...")
    
    try:
        df = pd.read_csv(file_path)
        
        # Check basic structure
        logger.info(f"✓ Shape: {df.shape}")
        logger.info(f"✓ Columns: {list(df.columns)}")
        
        # Check required columns
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
        else:
            logger.info(f"✓ All required columns present")
        
        # Check date range
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            logger.info(f"✓ Date range: {df['time'].min()} to {df['time'].max()}")
            logger.info(f"✓ Total records: {len(df)}")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"Missing values detected:\n{missing[missing > 0]}")
        else:
            logger.info(f"✓ No missing values")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Verification failed: {e}")
        return False

def main():
    """Main download workflow."""
    parser = argparse.ArgumentParser(description="Download VIC stock data")
    parser.add_argument(
        '--file-id',
        type=str,
        default="18J_Z8b-qMMj9wm5eGyQ-1nPS16PfRePK",
        help="Google Drive file ID"
    )
    parser.add_argument(
        '--skip-verify',
        action='store_true',
        help="Skip data verification"
    )
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("LTSF-Linear Data Download Script")
    logger.info("=" * 60)
    
    # Step 1: Setup directories
    logger.info("\n[Step 1/4] Setting up directories...")
    raw_dir, processed_dir = setup_data_directories()
    
    # Step 2: Download data
    logger.info("\n[Step 2/4] Downloading VIC.csv...")
    try:
        data_path = download_vic_data(raw_dir, file_id=args.file_id)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)
    
    # Step 3: Verify data
    if not args.skip_verify:
        logger.info("\n[Step 3/4] Verifying data...")
        if not verify_data(data_path):
            logger.error("Data verification failed!")
            sys.exit(1)
    else:
        logger.info("\n[Step 3/4] Skipping verification")
    
    # Success
    logger.info("\n" + "=" * 60)
    logger.info("✓ Data download complete!")
    logger.info(f"✓ Data location: {data_path}")
    logger.info("=" * 60)
    
    logger.info("\nNext steps:")
    logger.info("  1. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
    logger.info("  2. Run pipeline: python scripts/prepare_data.py")
    logger.info("  3. Train models: python scripts/train.py")


if __name__ == "__main__":
    main()