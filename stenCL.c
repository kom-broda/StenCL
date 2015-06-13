#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

typedef struct {
	cl_device_id device;
	size_t max_wg_dim;
} device_wg;

typedef struct {
	cl_platform_id platform;
	device_wg *device;
} platform_wg;

int main ()
{
	int i;
	
	//Load the kernel source code
	FILE *fp;
	char *source;
	size_t source_size;
	
	fp = fopen ("vector_add_kernel.cl", "r");
	
	if (!fp){
		fprintf (stderr, "Failed to load kernel.\n");
		exit (1);
	}
	
	source = (char *) malloc (MAX_SOURCE_SIZE);
	source_size = fread (source, 1, MAX_SOURCE_SIZE, fp);
	
	fclose (fp);
	
	//Get platforms and devices informations
	cl_platform_id *platforms = NULL;
	cl_device_id *devices = NULL;
	cl_uint *ret_num_devices;
	cl_uint ret_num_platforms;
	
	cl_int err = clGetPlatformIDs (2, platforms, &ret_num_platforms);
	
	devices = (cl_device_id *) malloc (ret_num_platforms);
	ret_num_devices = (cl_uint *) malloc (ret_num_platforms);
	
	for (i = 0; i < ret_num_platforms; i++){
		err = clGetDeviceIDs (platforms[i], CL_DEVICE_TYPE_DEFAULT, 5, &devices[i], &ret_num_devices[i]);
	}
	
	
	
	
	
	
	
	return 0;
}
