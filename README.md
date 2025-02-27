# MobilePoser: Real-Time Full-Body Pose Estimation and 3D Human Translation from IMUs in Mobile Consumer Devices
Author's implementation of the paper [MobilePoser: Real-Time Full-Body Pose Estimation and 3D Human Translation from IMUs in Mobile Consumer Devices](https://dl.acm.org/doi/pdf/10.1145/3654777.3676461). This work was published at UIST'24.

<br>
<div align="center">
<img src="teaser.gif" alt="teaser.gif" width="100%">
</div>
<br>

## Installation 
We recommend configuring the project inside an Anaconda environment. We have tested everything using [Anaconda](https://docs.anaconda.com/anaconda/install/) version 23.9.0 and Python 3.9. The first step is to create a virtual environment, as shown below (named `mobileposer`).
```
conda create -n mobileposer python=3.9
```
You should then activate the environment as shown below. All following operations must be completed within the virtual environment.
```
conda activate mobileposer
```
Then, install the required packages.
```
pip install -r requirements.txt
```
You will then need to install the local mobileposer package for development via the command below. You must run this from the root directory (e.g., where setup.py is).
```
pip install -e .
```

## Process Datasets

### Download Training Data
1. Register and download the AMASS dataset from [here](https://amass.is.tue.mpg.de/). We use 'SMPLH+G' for each dataset. 
2. Register and download the DIP-IMU dataset from [here](https://dip.is.tuebingen.mpg.de/). Download the raw (unormalized) data.
3. Request access to the TotalCapture dataset [here](https://cvssp.org/data/totalcapture/). Download Vicon Groundtruth in the raw folder, and IMU data in the IMU folder. 
4. Download the IMUPoser dataset from [here](https://github.com/FIGLAB/IMUPoser).

Once downloaded, your directory might appear as follows:
```bash
data
└── raw
    ├── AMASS
    │   ├── ACCAD
    │   ├── BioMotionLab_NTroje
    │   ├── BMLhandball
    │   ├── ...
    │   └── Transitions_mocap
    ├── DIP_IMU
    │   ├── s_01
    │   ├── s_02
    │   ├── s_03
    │   ├── ...
    │   └── s_10
    ├── IMUPoser
    │   ├── P1
    │   ├── P2
    │   ├── P3
    │   ├── ...
    │   └── P10
    └── TotalCapture/
            ├── IMU/
            │   ├── s1_acting1.pkl
            │   ├── ...
            └── raw/
                ├── S1/
                │   ├── acting1/
                │   │   ├── gt_skel_gbl_ori.txt
                │   │   ├── gt_skel_gbl_pos.txt
                │   ├── ...
```

### Setup Training Data 
In `config.py`: 
- Set `paths.processed_datasets` to the directory containing the pre-processed datasets.
- Set `paths.raw_amass` to the directory containing the AMASS dataset.
- Set `paths.raw_dip` to the directory containing the DIP dataset.
- Set `paths.raw_imuposer` to the directory containing the IMUPoser dataset.
  
The script `process.py` drives the dataset pre-processing. This script takes the following parameters:
1. `--dataset`: Dataset to pre-process (`amass`, `dip`, `imuposer`). Defaults to `amass`.

As an example, the following command will pre-process the DIP dataset. 
```
$ python process.py --dataset dip
```

## Training Models 
The script `train.py` drives the training process. This script takes the following parameters:
1. `--module`: Train an individual module (`poser`, `joints`, `foot_contact`, `velocity`). Default to training all modules. 
2. `--init-from`: Initialize training from an existing checkpoint. Defaults to training from scratch. 
3. `--finetune`: Specify dataset for finetuning module (e.g., `dip`). 
4. `--fast-dev-run`: A boolean flag that caps the execution to a single epoch. This flag is useful for debugging.

As an example, we can execute the following command to train all modules: 
```
$ python train.py
```

### Finetuning Model
To facilitate finetuning MobilePoser, we provide a script `finetune.sh`. To run this script, use the following syntax: 
```
$ ./finetune.sh <dataset-name> <checkpoint-directory>
```

### Prepare Model
The script `combine_weights.py` combines the weights of individual modules into a single weight file that can be loaded into `MobilePoserNet`. 
To run this script, use the following syntax: 
```
$ python combine_weights.py --finetune <dataset-name> --checkpoint <checkpoint-directory>
```
Omit the `--finetune` argument if you did not finetune. The resulting weight file will be stored under the same directory as the `checkpoint-directory>`


### Download pre-trained network weights
We provide a pre-trained model for the set of configurations listed in `config.py`. 
1. Download weights from [here](https://uchicago.box.com/s/ey3y49srpo79propzvmjx0t8u3ael6cl). 
2. In `config.py`, set the `paths.weights_file` to the model path.

### Run Evaluation
The script `evaluate.py` drives model evaluation. This script takes the following arguments. 
1. `--model`: Path to the trained model.
2. `--dataset`: Dataset to execute testing on (e.g., `dip`, `imuposer`, `totalcapture`).
   
As an example, we can execute the following concrete command:
```
$ python evaluate.py --model checkpoints/weights.pth --dataset dip
```

### Visualizing Results 
To visualize the prediction results of the trained model, we provide a script `example.py`. This script takes the following arguments. 
1. `--model`: Path to the trained model.
2. `--dataset`: Dataset to execute prediction for visualization. Defaults to `dip`.
3. `--seq-num`: Sequence nuber of dataset to execute prediction. Defaults to 1.
4. `--with-tran`: A boolean flag to enable visualizing translation estimation. Defaults to False. 
5. `--combo`: Device-location combination. Defaults to 'lw_rp' (left-wrist right-pocket).
   
Additionally, you can set the GT environment variable to customize visualization modes:
- GT=1: Visualizes both predictions and ground-truth.
- GT=2: Visualizes only the ground-truth data.

As an example, we can execute the following concrete command:
```
$ GT=1 python example.py --model checkpoints/weights.pth --dataset dip --seq-num 5 --with-tran
```

Note, we recommend using your local machine to visualize the results. 

## Citation 
```
@inproceedings{xu2024mobileposer,
  title={MobilePoser: Real-Time Full-Body Pose Estimation and 3D Human Translation from IMUs in Mobile Consumer Devices},
  author={Xu, Vasco and Gao, Chenfeng and Hoffmann, Henry and Ahuja, Karan},
  booktitle={Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology},
  pages={1--11},
  year={2024}
}
```

## Contact
For questions, please contact nu.spicelab@gmail.com.

## Acknowledgements 
We would like to thank the following projects for great prior work that inspired us: [TransPose](https://github.com/Xinyu-Yi/TransPose), [PIP](https://xinyu-yi.github.io/PIP/), [IMUPoser](https://github.com/FIGLAB/IMUPoser). 

## License 
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. For commercial use, a separate commercial license is required. Please contact kahuja@northwestern.edu at Northwestern University for licensing inquiries.
