# CNN-FPGA-Accelerator

# Introduction

Our CNN-FPGA-Accelerator is an open-source implementation of Convolutional neural network(CNN) accelerator on FPGA.   
We use the architecture of VGG model except for fully connected layers.   
The project goal is minimizing execution time by minimizing DRAM access and optimizing tiling factors for each layer.   
Additionally, we add pooling layers and ReLU activation.


![architecture](https://user-images.githubusercontent.com/31407544/147211496-c22d235e-55a1-45c1-b15d-cd455ddbe9f8.jpg)

<br />
<br />
<br />

# Description about Folders

## 1. Single layer configuration
This folder is for running just one single convolutional layer. 

Before you run each layer, **you should modify variable values in cnn.h file.**   
Please refer to the image below.
 ![factors](https://user-images.githubusercontent.com/31407544/147213681-247cdef7-7372-4c20-9828-a12ae3d00c4d.jpg)



## 2. 5layer_configuration
This folder is a connected version of all VGG convolution layers.

<br />
<br />
<br />

# Build Setup

### Device Setting

```
% git clone https://github.com/aws/aws-fpga.git $AWS_FPGA_REPO_DIR
% cd $AWS_FPGA_REPO_DIR
% source vitis_setup.sh
% source /opt/Xilinx/Vitis_HLS/2021.1/settings64.sh
% cd $AWS_FPGA_REPO_DIR/Vitis/examples/xilinx
% mkdir cnn_onboard
```

Then, download one of our folders which one you want to run into **cnn_onboard folder.**   


### Emulation and Run on FPGA
```
% cd cnn_onboard
% make cleanall
% make run TARGET=sw_emu DEVICE=$AWS_PLATFORM all
% make run TARGET=hw_emu DEVICE=$AWS_PLATFORM all
% make run TARGET=hw DEVICE=$AWS_PLATFORM all
```
<br />
<br />
<br />

# Results


![result](https://user-images.githubusercontent.com/31407544/147215283-55ffffaa-3388-41ef-819f-5ac907961913.jpg)
