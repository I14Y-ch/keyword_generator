#!/usr/bin/env python3
"""
Setup script to initialize the application for deployment.
Downloads required spaCy models.
"""
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def download_spacy_model(model_name='de_core_news_md'):
    """Download spaCy language model for German NLP."""
    logger.info(f"Downloading spaCy model: {model_name}...")
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'spacy', 'download', model_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info(f"✓ Successfully downloaded {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to download {model_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error downloading {model_name}: {e}")
        return False

def main():
    """Run setup tasks."""
    logger.info("Starting application setup...")
    
    # Download spaCy German model
    success = download_spacy_model('de_core_news_md')
    
    if success:
        logger.info("✓ Setup completed successfully!")
        return 0
    else:
        logger.error("✗ Setup encountered errors. Please check the logs above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
