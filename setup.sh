#!/bin/bash

# Exit on error
set -e

echo "Setting up MobilePoser environment for M2 Mac..."

# Remove existing environment if it exists
conda deactivate 2>/dev/null || true
conda remove --name mobileposer --all -y 2>/dev/null || true

# Create new environment
echo "Creating new conda environment..."
conda create -n mobileposer python=3.9 -c conda-forge -y

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mobileposer

# Install conda packages
echo "Installing conda packages..."
conda install -c conda-forge numpy==1.24.3 -y  # Using a specific version that's compatible
conda install -c conda-forge pandas scipy matplotlib -y

# Install PyTorch for M2
echo "Installing PyTorch for M2 Mac..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Install other requirements
echo "Installing other requirements..."
pip install lightning==2.0.0  # Using a specific version for compatibility
pip install open3d
pip install vctoolkit

# Clone and install modified chumpy
echo "Installing modified chumpy..."
if [ -d "chumpy" ]; then
    rm -rf chumpy
fi
git clone https://github.com/hassony2/chumpy.git
cd chumpy
# Apply patch to fix numpy deprecation warnings
sed -i '' 's/from numpy import bool, int, float, complex, object, str, nan, inf/from numpy import bool_, integer, floating, complexfloating, object_, str_, nan, inf/' chumpy/__init__.py
pip install -e .
cd ..

# Install local package
echo "Installing local package..."
pip install -e .

echo "Setup complete! You can now run: python mobileposer/example.py --model checkpoints/7/poser/base_model.pth --dataset dip --seq-num 5"