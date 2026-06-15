#include "ocl_setup.h"
#include "wrapcl.h"
#include <mutex>
#include <fstream>
#include <cstdio>

using namespace std;

OclSession ocl_setup(int gpu_id, const std::string& opencl_binary_path) {
	cl_int err;
	cl_platform_id* platforms;
	cl_device_id* devices;
	cl_context context;
	cl_command_queue queue;
	cl_int gpu_platform_id = 0;
	SetupPlatform(&platforms, &gpu_platform_id);
	SetupDevice(platforms, &devices, gpu_platform_id);
	printf("Using GPU device index: %d\n", gpu_id);
	// Fix: context must be created with the exact target device, not always devices[0]
	SetupContext(platforms, devices + gpu_id, &context, 1, gpu_platform_id);
	SetupQueue(&queue, context, devices, gpu_id);

	cl_program programs[2];

	//printf("\nSearch depth is set to %d", par.mc.search_depth);

#ifdef BUILD_KERNEL_FROM_SOURCE
{
	// Fix: serialize kernel compilation across GPU threads — only one thread compiles
	// and saves the .bin files; others wait and then load the cached result.
	static std::mutex kernel_compile_mutex;
	std::lock_guard<std::mutex> compile_lock(kernel_compile_mutex);

	// Compiled .bin files live in opencl_binary_path (writable host dir, default ".")
	const std::string bin_out_path = opencl_binary_path.empty() ? "." : opencl_binary_path;
	const std::string bin1 = bin_out_path + "/Kernel1_Opt.bin";
	const std::string bin2 = bin_out_path + "/Kernel2_Opt.bin";

	// Check if cached .bin files already exist for this machine's GPU
	auto file_exists = [](const std::string& p) {
		std::ifstream f(p); return f.good();
	};

	if (file_exists(bin1) && file_exists(bin2)) {
		printf("\nLoading cached GPU kernels from %s\n", bin_out_path.c_str()); fflush(stdout);
	} else {
		// First run on this GPU: compile from source and cache the .bin files
		// VINA_GPU_HOME is set in the SIF's ENV; falls back to "." for local builds
		const char* vina_home_env = std::getenv("VINA_GPU_HOME");
		const std::string kernel_src_path = vina_home_env ? std::string(vina_home_env) : ".";
		const std::string include_path = kernel_src_path + "/OpenCL/inc";
		const std::string addtion = "";

		printf("\n\nCompiling GPU kernels for this machine (one-time, caching to %s)...\n",
		       bin_out_path.c_str()); fflush(stdout);

		// --- Kernel 1 ---
		printf("  Building kernel 1 from source..."); fflush(stdout);
		char* program1_file_n[NUM_OF_FILES_KERNEL_1];
		size_t program1_size_n[NUM_OF_FILES_KERNEL_1];
		std::string file1_paths[NUM_OF_FILES_KERNEL_1] = {
			kernel_src_path + "/OpenCL/src/kernels/code_head.cl",
			kernel_src_path + "/OpenCL/src/kernels/kernel1.cl"
		};
		read_n_file(program1_file_n, program1_size_n, file1_paths, NUM_OF_FILES_KERNEL_1);
		std::string final_file;
		size_t final_size = NUM_OF_FILES_KERNEL_1 - 1;
		for (int i = 0; i < NUM_OF_FILES_KERNEL_1; i++) {
			final_file = (i == 0) ? program1_file_n[0] : final_file + '\n' + (std::string)program1_file_n[i];
			final_size += program1_size_n[i];
		}
		const char* final_files1_char = final_file.data();
		programs[0] = clCreateProgramWithSource(context, 1, (const char**)&final_files1_char, &final_size, &err); checkErr(err);
		SetupBuildProgramWithSource(programs[0], NULL, devices + gpu_id, include_path, addtion);
		SaveProgramToBinary(programs[0], bin1.c_str());
		printf(" done\n"); fflush(stdout);

		// --- Kernel 2 ---
		printf("  Building kernel 2 from source..."); fflush(stdout);
		char* program2_file_n[NUM_OF_FILES_KERNEL_2];
		size_t program2_size_n[NUM_OF_FILES_KERNEL_2];
		std::string file2_paths[NUM_OF_FILES_KERNEL_2] = {
			kernel_src_path + "/OpenCL/src/kernels/code_head.cl",
			kernel_src_path + "/OpenCL/src/kernels/mutate_conf.cl",
			kernel_src_path + "/OpenCL/src/kernels/matrix.cl",
			kernel_src_path + "/OpenCL/src/kernels/quasi_newton.cl",
			kernel_src_path + "/OpenCL/src/kernels/kernel2.cl"
		};
		read_n_file(program2_file_n, program2_size_n, file2_paths, NUM_OF_FILES_KERNEL_2);
		final_size = NUM_OF_FILES_KERNEL_2 - 1;
		for (int i = 0; i < NUM_OF_FILES_KERNEL_2; i++) {
			final_file = (i == 0) ? program2_file_n[0] : final_file + '\n' + (std::string)program2_file_n[i];
			final_size += program2_size_n[i];
		}
		const char* final_files2_char = final_file.data();
		programs[1] = clCreateProgramWithSource(context, 1, (const char**)&final_files2_char, &final_size, &err); checkErr(err);
		SetupBuildProgramWithSource(programs[1], NULL, devices + gpu_id, include_path, addtion);
		SaveProgramToBinary(programs[1], bin2.c_str());
		printf(" done\n\n"); fflush(stdout);
	}
}
#endif
	// Fix: pass devices+gpu_id so the program is built for the target GPU, not always devices[0]
	programs[0] = SetupBuildProgramWithBinary(context, devices + gpu_id, (opencl_binary_path + std::string("/Kernel1_Opt.bin")).c_str());

	programs[1] = SetupBuildProgramWithBinary(context, devices + gpu_id, (opencl_binary_path + std::string("/Kernel2_Opt.bin")).c_str());

	err = clUnloadPlatformCompiler(platforms[gpu_platform_id]); checkErr(err);
	//Set kernel arguments
	cl_kernel kernels[2];
	char kernel_name[][50] = { "kernel1","kernel2"};
	SetupKernel(kernels, programs, 2, kernel_name);

	size_t max_wg_size; // max work item within one work group
	size_t max_wi_size[3]; // max work item within each dimension(global)
	// Fix: query the target GPU (devices[gpu_id]), not always devices[0]
	err = clGetDeviceInfo(devices[gpu_id], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL); checkErr(err);
	err = clGetDeviceInfo(devices[gpu_id], CL_DEVICE_MAX_WORK_ITEM_SIZES, 3 * sizeof(size_t), &max_wi_size, NULL); checkErr(err);

	OclSession s;
	s.platforms = platforms; s.devices = devices; s.context = context; s.queue = queue;
	s.programs[0] = programs[0]; s.programs[1] = programs[1];
	s.kernels[0]  = kernels[0];  s.kernels[1]  = kernels[1];
	s.gpu_platform_id = gpu_platform_id; s.max_wg_size = max_wg_size;
	s.max_wi_size[0]=max_wi_size[0]; s.max_wi_size[1]=max_wi_size[1]; s.max_wi_size[2]=max_wi_size[2];
	return s;
}
