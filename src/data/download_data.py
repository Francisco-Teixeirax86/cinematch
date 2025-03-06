""""
Script to download the MovieLens Dataset for the CineMatch recommendation engine
"""
import os
import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

#Constants
DATASET_URL_DEV = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATASET_URL_PROD = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
DATA_DIR = Path(__file__).parents[2] / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ZIP_PATH = RAW_DATA_DIR / "ml-latest-small.zip"

def download_dataset():
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)

    if os.path.exists(ZIP_PATH):
        logger.info(f"{ZIP_PATH} already exists, skipping download")
        return

    try:
        logger.info(f"Downloading from {DATASET_URL_DEV}")
        urllib.request.urlretrieve(DATASET_URL_DEV, ZIP_PATH)
        logger.info(f"Download complete {ZIP_PATH}")
    except Exception as ex:
        logger.error(f"Failed to download {ZIP_PATH}: {ex}")
        sys.exit(1)

def extract_dataset():
    if not os.path.exists(ZIP_PATH):
        logger.info(f"Dataset not found at {ZIP_PATH}")
        sys.exit(1)

    try :
        logger.info(f"Extracting {ZIP_PATH}")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        logger.info(f"Extracting complete {ZIP_PATH}")
    except Exception as ex:
        logger.error(f"Failed to extract {ZIP_PATH}: {ex}")
        sys.exit(1)

def main():
    """Main function to download an extracted the MovieLens Dataset"""
    logger.info("Downloading MovieLens Dataset...")
    download_dataset()
    extract_dataset()
    logger.info("Finished Downloading MovieLens Dataset.")

if __name__ == "__main__":
    main()