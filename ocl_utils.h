#ifndef OCL_UTILS_H
#define OCL_UTILS_H

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_DEVICES_PER_PLATFORM 100


typedef struct {
	cl_device_id device;
	size_t wg_mult;
	size_t max_wg_dim;
} device_work_group_info;

typedef struct {
	cl_platform_id platform;
	cl_device_id *devices;
	int num_dev;
} platform_dev;

platform_dev *retrieve_platforms(int *num_platforms);
void display_platforms(platform_dev *system);

#endif
