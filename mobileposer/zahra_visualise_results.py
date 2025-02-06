import torch
from pathlib import Path
import numpy as np
from mobileposer.viewers import SMPLViewer
from mobileposer.config import *

class PredictionViewer(SMPLViewer):
    """Extension of SMPLViewer for prediction-only visualization."""
    def view_predictions(self, pose_p, tran_p):
        """View only predictions without ground truth."""
        # Create dummy data matching prediction shape for ground truth
        n_frames = pose_p.shape[0]
        pose_t = torch.zeros_like(pose_p)
        tran_t = torch.zeros_like(tran_p)
        
        super().view(pose_p, tran_p, pose_t, tran_t, with_tran=True)

def visualize_predictions(pred_path):
    """Visualize the predictions using SMPLViewer."""
    # Load predictions
    print("Loading predictions from:", pred_path)
    predictions = torch.load(pred_path, map_location='cpu')
    
    # Print available keys and their types
    print("\nPrediction data contains:")
    for key, value in predictions.items():
        print(f"{key}: {type(value)}")
        if isinstance(value, torch.Tensor):
            print(f"  Shape: {value.shape}")
    
    # Extract the predicted poses and translations
    pose = predictions.get('pose')
    translation = predictions.get('translation')
    
    if pose is None:
        raise ValueError("No pose data found in predictions")
    if translation is None:
        raise ValueError("No translation data found in predictions")
    
    print("\nPose shape:", pose.shape)
    print("Translation shape:", translation.shape)
    
    # Check SMPL model file
    smpl_path = paths.smpl_file
    if not smpl_path.exists():
        raise FileNotFoundError(f"SMPL model file not found at {smpl_path}")
    print("\nSMPL model file found at:", smpl_path)
    
    # Initialize viewer
    viewer = PredictionViewer()
    
    # View the predictions
    viewer.view_predictions(pose, translation)

if __name__ == "__main__":
    # Path to your predictions file
    pred_path = Path("data/processed_datasets/predictions.pt")
    
    if not pred_path.exists():
        print(f"Error: Predictions file not found at {pred_path}")
        print("Make sure you have generated predictions.pt first.")
        exit(1)
    
    try:
        visualize_predictions(pred_path)
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("\nPlease download the SMPL model file and place it at the correct location.")
        print("1. Register and download from https://smpl.is.tue.mpg.de/")
        print("2. Place basicmodel_m.pkl in the smpl/ directory")
    except Exception as e:
        print(f"\nError visualizing predictions: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting steps:")
        print("1. Make sure all dependencies are installed (pip install -r requirements.txt)")
        print("2. Verify predictions.pt was generated successfully")