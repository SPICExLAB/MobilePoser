import os
import sys
import torch
import numpy as np

def check_environment():
    print("Python version:", sys.version)
    print("\nPyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("MPS (Metal) available:", torch.backends.mps.is_available())
    
    print("\nNumPy version:", np.__version__)
    
    # Check model path
    model_path = "checkpoints/7/poser/base_model.pth"
    print("\nModel path exists:", os.path.exists(model_path))
    if os.path.exists(model_path):
        print("Model file size:", os.path.getsize(model_path) / (1024*1024), "MB")
    
    # Try loading the model
    try:
        print("\nAttempting to load model...")
        state_dict = torch.load(model_path, map_location='cpu')
        print("Model loaded successfully")
        print("Model keys:", list(state_dict.keys())[:5], "...")
    except Exception as e:
        print("Error loading model:", str(e))

if __name__ == "__main__":
    check_environment()