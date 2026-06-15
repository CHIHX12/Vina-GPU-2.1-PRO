#pragma once
// One-GPU OpenCL session setup, extracted from main_procedure_cl.cpp: platform/device/context/queue,
// build-or-load both kernel programs, create the kernels, query device limits. (Single-ligand path;
// the dual path keeps its own cached singleton.)
#include "commonMacros.h"   // cl_* types
#include <string>

struct OclSession {
    cl_platform_id*  platforms;
    cl_device_id*    devices;
    cl_context       context;
    cl_command_queue queue;
    cl_program       programs[2];
    cl_kernel        kernels[2];
    cl_int           gpu_platform_id;
    size_t           max_wg_size;
    size_t           max_wi_size[3];
};

OclSession ocl_setup(int gpu_id, const std::string& opencl_binary_path);
