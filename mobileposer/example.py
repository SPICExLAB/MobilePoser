import os
import sys
import torch
import torch.nn as nn
import lightning as L
from torch.nn import functional as F
import numpy as np
from argparse import ArgumentParser
import traceback

from mobileposer.config import *
from mobileposer.utils.model_utils import *
from mobileposer.viewer import Viewer
from mobileposer.utils.open3d_init import init_open3d_for_apple_silicon

def debug_print(msg):
    print(f"DEBUG: {msg}", flush=True)

if __name__ == "__main__":
    try:
        debug_print("Starting script")
        args = ArgumentParser()
        args.add_argument('--model', type=str, default=paths.weights_file)
        args.add_argument('--dataset', type=str, default='dip')
        args.add_argument('--combo', type=str, default='lw_rp')
        args.add_argument('--with-tran', action='store_true')
        args.add_argument('--seq-num', type=int, default=1)
        args = args.parse_args()

        debug_print("Arguments parsed")

        # Force CPU usage and set memory settings
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        device = torch.device('cpu')
        torch.set_default_dtype(torch.float32)
        
        debug_print(f"Using device: {device}")

        # check for valid combo
        combos = amass.combos.keys()
        if args.combo not in combos:
            raise ValueError(f"Invalid combo: {args.combo}. Must be one of {combos}")

        debug_print("Combo validated")

        # Memory info before viewer creation
        debug_print(f"Memory allocated: {torch.cuda.memory_allocated() if torch.cuda.is_available() else 'N/A'}")
        
        try:
            if not init_open3d_for_apple_silicon():
                print("Warning: Open3D initialization failed, visualization may not work properly")
            debug_print("Creating viewer")
            v = Viewer(dataset=args.dataset, seq_num=args.seq_num, combo=args.combo)
            debug_print("Viewer created")
            
            debug_print("Starting visualization")
            v.view(with_tran=args.with_tran)
            debug_print("Visualization complete")
            
        except Exception as e:
            debug_print(f"Error in viewer: {str(e)}")
            debug_print(traceback.format_exc())
            raise

    except Exception as e:
        debug_print(f"Fatal error: {str(e)}")
        debug_print(traceback.format_exc())
        sys.exit(1)