#!/bin/bash
# Setup script for the Keyword Generator application

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Downloading spacy language model..."
python -m spacy download en_core_web_sm

echo "Setup complete!"
