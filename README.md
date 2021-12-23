# CNN-FPGA-Accelerator

Introduction
---------------------------------------
Our CNN-FPGA-Accelerator is an open-source implementation of Convolutional neural network(CNN) accelerator on FPGA.   
We use the architecture of VGG model except for fully connected layers.   
The project goal is minimizing execution time by minimizing DRAM access and optimizing tiling factors for each layer.
Additionally, we add pooling layers and ReLU activation.

![architecture](https://user-images.githubusercontent.com/31407544/147211496-c22d235e-55a1-45c1-b15d-cd455ddbe9f8.jpg)

Build Setup (+ Requirements?)
---------------------------------------
aws-fpga-repo-dir  git clone?   
```
% source setup.sh
% cd [folder path]
% make cleanall
% make run TARGET=sw_emu DEVICE=$AWS_PLATFORM all
% make run TARGET=hw_emu DEVICE=$AWS_PlATFORM all
```

Results
---------------------------------------
Execution time for each layer
- Layer1: Tr=  Tm=   Execution time:
- Layer2:
- Layer3:
- Layer4:
- Layer5: 
