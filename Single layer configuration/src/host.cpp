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
	std::vector<short, aligned_allocator<short> > & ker
	){

	// initialize input
	for(int row = 0; row < R+K-1; row++) {
		for(int col = 0; col < C+K-1; col++) {
			for(int chi = 0; chi < N; chi++) {
				if (row >= (K-1)/2 && row < R+(K-1)/2 && col >= (K-1)/2 && col < C+(K-1)/2) {
					inp[chi*(R+K-1)*(C+K-1) + row*(C+K-1) + col] = rand() % 3 - 1;
				}
				else {
					inp[chi*(R+K-1)*(C+K-1) + row*(C+K-1) + col] = 0;
			}}}}
	// initialize kernel
	for(int cho = 0; cho < M; cho++) {
		for(int chi = 0; chi < N; chi++) {
          		 	for (int ki = 0; ki < K; ki++) {
           				for (int kj = 0; kj < K; kj++) {    
					ker[cho*(N*K*K) + chi*(K*K) + ki*K + kj] = rand() % 3 - 1;
			}}}}
	

}
//--------------------------------------------------------------------------------------------------------------------

short IsError(short a, short b) {
	return fabs((a - b) / (a + b)) > 1e-3f && fabs(a - b) > 0.05f;
}

//--------------------------------------------------------------------------------------------------------------------

int main(int argc, char** argv) {
	if (argc != 2) {
		std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
		return EXIT_FAILURE;
	}
	
	std::vector<short, aligned_allocator<short> > inp(N*(R+K-1)*(C+K-1));
	std::vector<short, aligned_allocator<short> > ker(M*N*K*K);
	std::vector<short, aligned_allocator<short> > out_hw(M*(R/2+(K-1))*(C/2+(K-1)));


	std::cout << "Loading input data...\n";
	LoadData(inp, ker);
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
	for(int row = 0; row < R/2+K-1; row++) {
		for(int col = 0; col < C/2+K-1; col++) {
			for(int cho = 0; cho < M; cho++) {
				out_hw[cho*((R/2+(K-1))*(C/2+(K-1))) + row*(C/2+(K-1)) + col] = 0;
	}}}

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
	OCL_CHECK(err, cl::Buffer buffer_input(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N*(R+K-1)*(C+K-1), inp.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_weight(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(short) * N*M*K*K, ker.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(short) * R/2*C/2*M, out_hw.data(), &err));
	OCL_CHECK(err, err = kernel.setArg(0, buffer_input));
	OCL_CHECK(err, err = kernel.setArg(1, buffer_weight));
	OCL_CHECK(err, err = kernel.setArg(2, buffer_output));

	// Copy input data to device global memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input, buffer_weight, buffer_output}, 0 /* 0 means from host*/));
	q.finish();
	
	std::cout << "Running FPGA CNN...\n";
	auto start = std::chrono::steady_clock::now();

	// Launch the Kernel
	OCL_CHECK(err, err = q.enqueueTask(kernel));
	q.finish();

	auto end = std::chrono::steady_clock::now();
	std::cout << "Done.\n";

	double exec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	double gflops = double(N) * M * R * C * K * K * 2 / (exec_time);
	std::cout << "Time: " << exec_time*1e-9 << ", GFLOPS: " << gflops << std::endl;

	// Copy Result from Device Global Memory to Host Local Memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
	q.finish();
	// OPENCL HOST CODE AREA END

	return EXIT_SUCCESS;
}
