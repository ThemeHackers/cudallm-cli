#!/bin/bash
set -e

echo "=================================================="
echo "Starting automated CUDA installation for cudallm-cli"
echo "=================================================="


if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        echo "Activating existing virtual environment..."
        source .venv/bin/activate
    else
        echo "Creating new virtual environment (.venv)..."
        python3 -m venv .venv
        source .venv/bin/activate
    fi
fi


python3 -m pip install --upgrade pip



echo "Installing dependencies from requirements.txt with CUDA support..."
pip install -r requirements.txt --no-cache-dir

echo "Installing cudallm-cli in editable mode..."
pip install -e .


echo "Initializing environment..."
cudallm init

echo "=================================================="
echo "Installation completed! Run 'cudallm doctor' to verify."
echo "=================================================="
