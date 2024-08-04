# <p align="center">A cost-effective and robust mapping method for diverse crop types using weakly supervised semantic segmentation with sparse point samples</p>

This is an official implementation of the "A cost-effective and robust mapping method for diverse crop types using weakly supervised semantic segmentation with sparse point samples".

The overall structure of StrucLabX-Net for weakly supervised semantic segmentation of satellite image time series. 
![image](https://github.com/user-attachments/assets/5a7f4bdc-7810-470e-897b-333240ca6a24)

The overall architecture of U-TempoNet with an example of the crop classification as a downstream task
![image](https://github.com/user-attachments/assets/595f8409-626c-4819-9d33-a0469cc8e6f2)




## Requirement 
``````
PyTorch
Numpy
OpenCV-Python
tqdm
``````

## Data Format
``````
inputs
└── <train>
    ├── img
    │ ├── 00001.tif
    │ ├── 00002.tif
    │ ├── 00003.tif
    │ └── ...
    ├── lbl
    │ ├── 00001.tif
    │ ├── 00002.tif
    │ ├── 00003.tif
    │ └── ...
└── <val>
    ├── img
    │ ├── 00001.tif
    │ ├── 00002.tif
    │ ├── 00003.tif
    │ └── ...
    ├── lbl
    │ ├── 00001.tif
    │ ├── 00002.tif
    │ ├── 00003.tif
    │ └── ...
``````

## Usage
``````
import torch
from utils import train
from utils.BalancedDataParallel import BalancedDataParallel
from model import model as utemponet

batch_size = 16
lr = 1e-3
MAX_EPOCH = 150
NUM_WORKERS = 4
GPU0_BSZ = 8
ACC_GRAD = 1

IN_CHANNELS = 4
NUM_CLASSES = 8
NUM_LAYERS = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = utemponet.UTempoNet(
                IN_CHANNELS,
                NUM_CLASSES,
                NUM_LAYERS,
                )

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
#   dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = BalancedDataParallel(GPU0_BSZ // ACC_GRAD,model,device_ids=[0,1],output_device=0)
    
train_folder = r'**/train'
val_folder = r'**/val'

model_name = 'model_name'

model = model.to(device)

train_kwargs = dict({'net':model,
                    'devices':device,
                    'batchsize':batch_size,
                    'lr':lr,
                    'num_classes':NUM_CLASSES,
                    'max_epoch':MAX_EPOCH,
                    'train_folder':train_folder,
                    'val_folder':val_folder,
                    'num_workers':NUM_WORKERS,
                    'data_aug': True,
                    'model_name':model_name,
                    'resume':False,
                    'hyper_params':{'th_a':0.99,
                                    'th_b':0.15}
                    })
train.train_model(**train_kwargs)
``````
