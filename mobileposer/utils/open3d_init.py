import os
import platform

def init_open3d_for_apple_silicon():
    """Initialize Open3D with settings optimized for Apple Silicon."""
    # Force CPU rendering
    os.environ['OPEN3D_CPU_RENDERING'] = '1'
    
    # Disable Metal/MPS
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Set OpenGL parameters
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    
    # Force single-threaded operation
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # Import Open3D after setting environment variables
    try:
        import open3d as o3d
        
        # Configure Open3D for CPU rendering
        if platform.processor() == 'arm':
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
            
        return True
    except Exception as e:
        print(f"Error initializing Open3D: {e}")
        return False