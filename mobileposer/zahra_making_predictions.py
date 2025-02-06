import torch
import numpy as np
from pathlib import Path
from mobileposer.models import *
from mobileposer.config import *
from mobileposer.utils.model_utils import load_model

class IMUDataProcessor:
    def __init__(self, model_path):
        """Initialize the IMU data processor with a trained model."""
        self.device = torch.device('cpu')  # Force CPU for Mac
        self.model = load_model(model_path).to(self.device).eval()
        
    def process_frame(self, frame_data):
        """Process a single frame of IMU data."""
        # For 5 sensor locations, each with 12 values (3 acc + 9 rotation)
        processed_data = torch.zeros(model_config.n_imu)  # 60 values total
        
        for sensor_idx in range(5):
            start_idx = sensor_idx * 12
            sensor_start = sensor_idx * 12  # 12 values per sensor in the output
            
            # Extract and process acceleration (3 values)
            acc = frame_data[start_idx:start_idx + 3] / amass.acc_scale
            processed_data[sensor_start:sensor_start + 3] = acc
            
            # Extract and process orientation (9 values)
            ori = frame_data[start_idx + 3:start_idx + 12]
            processed_data[sensor_start + 3:sensor_start + 12] = ori
            
        return processed_data
        
    def load_imu_data(self, left_path, right_path):
        """Load and combine IMU data from left and right sensor files."""
        # Load the tensor data
        left_data = torch.load(left_path)['imu_data']
        right_data = torch.load(right_path)['imu_data']
        
        n_frames = left_data.shape[0]
        print(f"Processing {n_frames} frames of IMU data")
        
        # Initialize output tensor (n_frames x 60)
        imu_input = torch.zeros((n_frames, model_config.n_imu))
        
        # Process each frame
        for i in range(n_frames):
            # Get base frame data from left sensor
            frame_data = self.process_frame(left_data[i])
            
            # Override right pocket sensor data (4th sensor position)
            right_pocket_start = 3 * 12  # 4th sensor position (0-based index)
            right_pocket_data = right_data[i, right_pocket_start:right_pocket_start + 12]
            
            # Apply right pocket data
            frame_data[right_pocket_start:right_pocket_start + 12] = right_pocket_data
            
            imu_input[i] = frame_data
            
        print(f"Final input shape: {imu_input.shape}")
        return imu_input
    
    def predict_pose(self, imu_data):
        """Generate pose predictions from processed IMU data."""
        with torch.no_grad():
            # Forward pass through the model
            pose, joints, tran, contact = self.model.forward_offline(
                imu_data.unsqueeze(0).to(self.device), 
                [imu_data.shape[0]]
            )
        return pose, tran, joints, contact

def main():
    # Setup paths
    data_dir = Path("data/processed_datasets")  # Adjust as needed
    model_path = paths.weights_file  # From config.py
    
    # Initialize processor
    processor = IMUDataProcessor(model_path)
    
    try:
        # Load and process IMU data
        imu_data = processor.load_imu_data(
            data_dir / "mobileposer_data_left.pt",
            data_dir / "mobileposer_data_right.pt"
        )
        
        # Generate predictions
        pose, tran, joints, contact = processor.predict_pose(imu_data)
        
        # Save predictions
        output = {
            'pose': pose.cpu(),
            'translation': tran.cpu(),
            'joints': joints.cpu(),
            'foot_contact': contact.cpu()
        }
        torch.save(output, data_dir / "predictions.pt")
        print("\nSuccessfully generated and saved predictions!")
        
    except Exception as e:
        print(f"\nError processing data: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPlease check your data structure and try again.")

if __name__ == "__main__":
    main()