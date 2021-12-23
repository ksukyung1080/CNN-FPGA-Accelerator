/**
* Copyright (C) 2020 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#include "xcl2.hpp"
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <omp.h>

#include "cnn.h"

void LoadData(
	std::vector<short, aligned_allocator<short> > & inp,
	std::vector<short, aligned_allocator<short> > & ker1,
	std::vector<short, aligned_allocator<short> > & ker2,
	std::vector<short, aligned_allocator<short> > & ker3,
	std::vector<short, aligned_allocator<short> > & ker4,
	std::vector<short, aligned_allocator<short> > & ker5,
	std::vector<short, aligned_allocator<short> > & ker6,
	std::vector<short, aligned_allocator<short> > & ker7,
	std::vector<short, aligned_allocator<short> > & ker8
	) {
	

	// initialize input
	for(int row = 0; row < IR+IK-1; row++) {
		for(int col = 0; col < IC+IK-1; col++) {
			for(int chi = 0; chi < IN; chi++) {
				if (row >= (IK-1)/2 && row < IR+(IK-1)/2 && col >= (IK-1)/2 && col < IC+(IK-1)/2) {
					inp[chi*(IR+IK-1)*(IC+IK-1) + row*(IC+IK-1) + col] = rand() % 3 - 1;
//					printf("chi : %d, col : %d, row : %d, input : %d\n", chi, col, row, inp[chi*(R+K-1)*(C+K-1) + row*(C+K-1) + col] );
				}
				else {
					inp[chi*(IR+IK-1)*(IC+IK-1) + row*(IC+IK-1) + col] = 0;
				}}}}

	// initialize kernel
		for(int cho = 0; cho < IM; cho++) {
		for(int chi = 0; chi < IN; chi++) {
          		 	for (int ki = 0; ki < IK; ki++) {
           				for (int kj = 0; kj < IK; kj++) {    
					ker1[cho*(IN*IK*IK) + chi*(IK*IK) + ki*IK + kj] = rand() % 8 - 1;
	}}}}
		for(int cho = 0; cho < IM*2; cho++) {
		for(int chi = 0; chi < IN; chi++) {
          		 	for (int ki = 0; ki < IK; ki++) {
           				for (int kj = 0; kj < IK; kj++) {    
					ker2[cho*(IN*IK*IK) + chi*(IK*IK) + ki*IK + kj] = rand() % 8 - 1;
	}}}}
		for(int cho = 0; cho < IM*2; cho++) {
		for(int chi = 0; chi < IN*2; chi++) {
          		 	for (int ki = 0; ki < IK; ki++) {
           				for (int kj = 0; kj < IK; kj++) {    
					ker3[cho*(IN*IK*IK) + chi*(IK*IK) + ki*IK + kj] = rand() % 8 - 1;
	}}}}
		for(int cho = 0; cho < IM*4; cho++) {
		for(int chi = 0; chi < IN*2; chi++) {
          		 	for (int ki = 0; ki < IK; ki++) {
           				for (int kj = 0; kj < IK; kj++) {    
					ker4[cho*(IN*IK*IK) + chi*(IK*IK) + ki*IK + kj] = rand() % 8 - 1;
	}}}}
		for(int cho = 0; cho < IM*4; cho++) {
		for(int chi = 0; chi < IN*4; chi++) {
          		 	for (int ki = 0; ki < IK; ki++) {
           				for (int kj = 0; kj < IK; kj++) {    
					ker5[cho*(IN*IK*IK) + chi*(IK*IK) + ki*IK + kj] = rand() % 8 - 1;
	}}}}
		for(int cho = 0; cho < IM*8; cho++) {
		for(int chi = 0; chi < IN*4; chi++) {
          		 	for (int ki = 0; ki < IK; ki++) {
           				for (int kj = 0; kj < IK; kj++) {    
					ker6[cho*(IN*IK*IK) + chi*(IK*IK) + ki*IK + kj] = rand() % 8 - 1;
	}}}}
		for(int cho = 0; cho < IM*8; cho++) {
		for(int chi = 0; chi < IN*8; chi++) {
          		 	for (int ki = 0; ki < IK; ki++) {
           				for (int kj = 0; kj < IK; kj++) {    
					ker7[cho*(IN*IK*IK) + chi*(IK*IK) + ki*IK + kj] = rand() % 8 - 1;
	}}}}
		for(int cho = 0; cho < IM*8; cho++) {
		for(int chi = 0; chi < IN*8; chi++) {
          		 	for (int ki = 0; ki < IK; ki++) {
           				for (int kj = 0; kj < IK; kj++) {    
					ker8[cho*(IN*IK*IK) + chi*(IK*IK) + ki*IK + kj] = rand() % 8 - 1;
	}}}}
	

}
//--------------------------------------------------------------------------------------------------------------------

short IsError(short a, short b) {
	return fabs((a - b) / (a + b)) > 1e-3f && fabs(a - b) > 0.05f;
}

//--------------------------------------------------------------------------------------------------------------------
/*
void cnn_sw(
	std::vector<short, aligned_allocator<short> > inp,
	std::vector<short, aligned_allocator<short> > ker,
	std::vector<short, aligned_allocator<short> > & out
	){


	// Convolution
#pragma omp parallel for
	for(int row = 0; row < R; row++) {
		for(int col = 0; col < C; col++) {
			for(int cho = 0; cho < M; cho++) {
				for(int chi = 0; chi < N; chi++) {
                				for (int ki = 0; ki < K; ki++) {
                					for (int kj = 0; kj < K; kj++) {
							out[row*M*C + col*M + cho] += ker[ki*M*N*K + kj*M*N + cho*N + chi] * inp[(row+ki)*N*(C+K-1) + (col+kj)*N + chi];

						}
					}
				}
//			printf("%d\n",out[row*M*C + col*M + cho]);
			}
		}
	}

}
*/
//--------------------------------------------------------------------------------------------------------------------
/*
void cnn_sw_pooling(
	std::vector<short, aligned_allocator<short> > inlayer,
	std::vector<short, aligned_allocator<short> > & outlayer	
	){
	int ss = 0;
	static short out[R*C*M/2/2];

#pragma omp parallel for
	for(int i = 0; i < M; i++) {
		for(int j = 0; j < R/2; j++) {
			for(int k = 0; k < C/2; k++) {
				short maxx = Max(inlayer[i*R*C+2*k+2*R*j], inlayer[i*R*C+2*K+2*R*j+1], inlayer[i*R*C+2*k+2*R*j+C], inlayer[i*R*C+2*K+2*R*j+C+1]);	
				out[ss] = maxx;
				ss++;
			}
		}	
	}
	for(int i = 0; i < R*C/2/2; i++){
		for(int j = 0; j < M; j++){
			outlayer[M*i+j] = out[i+j*R*C/2/2];
		}
	}
} 
*/
//--------------------------------------------------------------------------------------------------------------------


int main(int argc, char** argv) {
	if (argc != 2) {
		std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
		return EXIT_FAILURE;
	}
	
	std::vector<short, aligned_allocator<short> > inp(IN * (IR + IK - 1) * (IC + IK - 1));
	std::vector<short, aligned_allocator<short> > ker1(IM * IN * IK * IK);
	std::vector<short, aligned_allocator<short> > ker2(IM *  2 * IN * IK * IK);
	std::vector<short, aligned_allocator<short> > ker3(IM *  2 * IN *  2 * IK * IK);
	std::vector<short, aligned_allocator<short> > ker4(IM *  4 * IN *  2 * IK * IK);
	std::vector<short, aligned_allocator<short> > ker5(IM *  4 * IN *  4 * IK * IK);
	std::vector<short, aligned_allocator<short> > ker6(IM *  8 * IN *  4 * IK * IK);
	std::vector<short, aligned_allocator<short> > ker7(IM *  8 * IN *  8 * IK * IK);
	std::vector<short, aligned_allocator<short> > ker8(IM *  8 * IN *  8 * IK * IK);
//	std::vector<short, aligned_allocator<short> > out_sw(M*R*C);
//	std::vector<short, aligned_allocator<short> > out_sw_pooling(M*R/2*C/2);
	std::vector<short, aligned_allocator<short> > out_hw(IM*8*IR/16*IC/16);


	std::cout << "Loading input data...\n";
	LoadData(inp, ker1, ker2, ker3, ker4, ker5, ker6, ker7, ker8);
	//LoadData(inp, ker1);
	std::cout << "Done.\n";

	

#pragma omp parallel
{
	int tid = omp_get_thread_num();
	if( tid == 0 ){
		int nthreads = omp_get_num_threads();
		std::cout << "Running CPU CNN with " << nthreads << " threads...\n";
	}
}
#pragma omp parallel for
	// initialize output
	for(int row = 0; row < IR/16; row++) {
		for(int col = 0; col < IC/16; col++) {
			for(int cho = 0; cho < IM*8; cho++) {
//				out_sw[cho*(R*C) + row*C + col] = 0;
				out_hw[cho*(IR/16*IC/16) + row*IC/16 + col] = 0;
//				out_sw_pooling[cho*(R/2*C/2) + row*C/2 + col] = 0;
	}}}
	// initialize output
/*	for(int row = 0; row < R; row++) {
		for(int col = 0; col < C; col++) {
			for(int cho = 0; cho < M; cho++) {
				out_sw[cho*(R*C) + row*C + col] = 0;
	}}}


	auto start = std::chrono::steady_clock::now();

	cnn_sw(inp, ker, out_sw);
	cnn_sw_pooling(out_sw, out_sw_pooling);
	auto end = std::chrono::steady_clock::now();
	std::cout << "Done.\n";

	double exec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	double gflops = double(N) * M * R * C * K * K * 2 / (exec_time);
	std::cout << "Time: " << exec_time*1e-9 << ", GFLOPS: " << gflops << std::endl;
*/
	std::string binaryFile = argv[1];
	cl_int err;
	cl::Context context;
	cl::Kernel kernel;
	cl::CommandQueue q;


	// OPENCL HOST CODE AREA START
	// get_xil_devices() is a utility API which will find the xilinx
	// platforms and will return list of devices connected to Xilinx platform
	auto devices = xcl::get_xil_devices();
	// read_binary_file() is a utility API which will load the binaryFile
	// and will return the pointer to file buffer.
	auto fileBuf = xcl::read_binary_file(binaryFile);
	cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
	bool valid_device = false;
	for (unsigned int i = 0; i < devices.size(); i++) {
		auto device = devices[i];
		// Creating Context and Command Queue for selected Device
		OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
		OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
		std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
		cl::Program program(context, {device}, bins, nullptr, &err);
		if (err != CL_SUCCESS) {
			std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
		} else {
			std::cout << "Device[" << i << "]: program successful!\n";
			OCL_CHECK(err, kernel = cl::Kernel(program, "cnn", &err));
			valid_device = true;
			break; // we break because we found a valid device
		}
	}
	if (!valid_device) {
		std::cout << "Failed to program any device found, exit!\n";
		exit(EXIT_FAILURE);
	}

	// Allocate Buffer in Global Memory
	OCL_CHECK(err, cl::Buffer buffer_input(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * IN*(IR+IK-1)*(IC+IK-1), inp.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * IN * IM * IK * IK, ker1.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * IN * IM *  2 * IK * IK, ker2.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * IN *  2 * IM *  2 * IK * IK, ker3.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight4(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * IN *  2 * IM *  4 * IK * IK, ker4.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight5(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * IN *  4 * IM *  4 * IK * IK, ker5.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight6(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * IN *  4 * IM *  8 * IK * IK, ker6.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight7(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * IN *  8 * IM *  8 * IK * IK, ker7.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight8(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * IN *  8 * IM *  8 * IK * IK, ker8.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * IM*8*IR/16*IC/16, out_hw.data(), &err));
	OCL_CHECK(err, err = kernel.setArg(0, buffer_input));
	OCL_CHECK(err, err = kernel.setArg(1, buffer_weight1));
	OCL_CHECK(err, err = kernel.setArg(2, buffer_weight2));
	OCL_CHECK(err, err = kernel.setArg(3, buffer_weight3));
	OCL_CHECK(err, err = kernel.setArg(4, buffer_weight4));
	OCL_CHECK(err, err = kernel.setArg(5, buffer_weight5));
	OCL_CHECK(err, err = kernel.setArg(6, buffer_weight6));
	OCL_CHECK(err, err = kernel.setArg(7, buffer_weight7));
	OCL_CHECK(err, err = kernel.setArg(8, buffer_weight8));
	OCL_CHECK(err, err = kernel.setArg(9, buffer_output));
	
	// Copy input data to device global memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input, buffer_weight1, buffer_weight2, buffer_weight3, buffer_weight4, buffer_weight5, buffer_weight6, buffer_weight7, buffer_weight8, buffer_output}, 0 /* 0 means from host*/));
	q.finish();
	
	
	
	std::cout << "Running FPGA CNN...\n";
	auto start = std::chrono::steady_clock::now();

	// Launch the Kernel
	OCL_CHECK(err, err = q.enqueueTask(kernel));
	q.finish();

	auto end = std::chrono::steady_clock::now();
	std::cout << "Done.\n";

	double x = 1;
	double exec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	double gflops = double(x) * 10635706368 / (exec_time);
	std::cout << "Time: " << exec_time*1e-9 << ", GFLOPS: " << gflops << std::endl;

	// Copy Result from Device Global Memory to Host Local Memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
	q.finish();
	// OPENCL HOST CODE AREA END
/*	int sss = 0;
	std::cout <<"output value\n";
	for(int v = 0; v < IM*2*IC/4*IR/4; v++) {
		printf("output value %d : %d\n",sss, out_hw[v]);
		sss++;
	}

*/
	
/*	// Verification
	int err_cnt = 0;
	for(int row = 0; row < R; row++) {
		for(int col = 0; col < C; col++) {
			for(int cho = 0; cho < M; cho++) {	
				printf("cho:%d row:%d col:%d sw:%d \n", cho, row, col, out_sw_pooling[cho*(R/2*C/2) + row*C/2 + col]);

			}
		}
	}
	
	for(int row = 0; row < R/2; row++) {
		for(int col = 0; col < C/2; col++) {
			for(int cho = 0; cho < M; cho++) {
				if(out_sw_pooling[cho*(R/2*C/2) + row*C/2 + col] != out_hw[cho*(R/2*C/2) + row*C/2 + col]) {
					err_cnt++;
				}
				printf("cho:%d row:%d col:%d sw:%d hw:%d\n", cho, row, col, out_sw_pooling[cho*(R/2*C/2) + row*C/2 + col], out_hw[cho*(R/2*C/2) + row*C/2 + col]);
				
			}
		}
	}
	if(err_cnt != 0){
		printf("FAILED! Error count : %d\n", err_cnt);
	}
	else{
		printf("PASSED!\n");
	}	

*/

	return EXIT_SUCCESS;
}
