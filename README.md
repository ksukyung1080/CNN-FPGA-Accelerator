# CNN-FPGA-Accelerator

Introduction
---------------------------------------
Our CNN-FPGA-Accelerator is an open-source implementation of Convolutional neural network(CNN) accelerator on FPGA.   
We use the architecture of VGG model except for fully connected layers.   
The project goal is minimizing execution time by minimizing DRAM access and optimizing tiling factors for each layer.   
Additionally, we add pooling layers and ReLU activation.


![architecture](https://user-images.githubusercontent.com/31407544/147211496-c22d235e-55a1-45c1-b15d-cd455ddbe9f8.jpg)


Build Setup
---------------------------------------
### Device Setting

```
% git clone https://github.com/aws/aws-fpga.git $AWS_FPGA_REPO_DIR
% cd $AWS_FPGA_REPO_DIR
% source vitis_setup.sh
% source /opt/Xilinx/Vitis_HLS/2021.1/settings64.sh
```

### NOTICE
Before you run each layer, **you should modify variable values in cnn.h file.**   
Please refer to the image below.
 ![factors](https://user-images.githubusercontent.com/31407544/147213681-247cdef7-7372-4c20-9828-a12ae3d00c4d.jpg)

### Emulation and Run on FPGA
```
% cd [folder path]
% make cleanall
% make run TARGET=sw_emu DEVICE=$AWS_PLATFORM all
% make run TARGET=hw_emu DEVICE=$AWS_PlATFORM all
% make run TARGET=hw DEVICE=$AWS_PLATFORM all
```

Results
---------------------------------------

![result](https://user-images.githubusercontent.com/31407544/147215283-55ffffaa-3388-41ef-819f-5ac907961913.jpg)
