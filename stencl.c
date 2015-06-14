#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "ocl_utils.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

unsigned total_dev (platform_dev *plat, int num){
	unsigned count = 0;
	int i;
	for (i = 0; i < num; i++){
		count += plat[i].num_dev;
	}

	return count;
}

void main()
{
    platform_dev *system_;
    char info[200];
    int i, j, num_platforms;
    
    cl_int err;
    
    system_=retrieve_platforms(&num_platforms);
    
    for(i = 0; i < num_platforms; i++)
    {
        err = clGetPlatformInfo (system_[i].platform, CL_PLATFORM_NAME, 200, (void *)info, NULL);
        printf("Platform n.%d, name: %s\n",i+1,info);
        
        for(j = 0; j < system_[i].num_dev; j++)
        {
            err = clGetDeviceInfo (system_[i].devices[j], CL_DEVICE_NAME, 200, (void *)info, NULL);
            printf("Device n.%d, name: %s\n",j+1,info);
        }
    }
    
    // Load the kernel source code
    FILE *fp;
    char *source_str;
    size_t source_size;
 
    fp = fopen("vector_add_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
    
    //Retrieve work group dimension for each device
    device_work_group_info *dev_wg = (device_work_group_info *) malloc (sizeof (device_work_group_info) * total_dev (system_, num_platforms));
    int k = 0;
    cl_context context;
    cl_program program;
    cl_kernel kernel;
    
    for (i = 0; i < num_platforms; i++){
    	for (j = 0; j < system_[i].num_dev; j++){
    		dev_wg[k].device = system_[i].devices[j];
    		
    		// Create an OpenCL context
    		context = clCreateContext( NULL, 1, &dev_wg[k].device, NULL, NULL, &err);
    
    		// Create a program from the kernel source
    		program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &err);
 
    		// Build the program
    		err = clBuildProgram(program, 1, &dev_wg[k].device, NULL, NULL, NULL);
    		
    		// Create the OpenCL kernel
    		kernel = clCreateKernel(program, "vector_add", &err);    
			
			//Get work group dimension for the k-th device
    		err = clGetKernelWorkGroupInfo (kernel, dev_wg[k].device,	CL_KERNEL_WORK_GROUP_SIZE, sizeof (size_t),
    										(void *) &dev_wg[k].max_wg_dim, NULL);
    		
    		printf ("Max work group dimension: %d\n", dev_wg[k].max_wg_dim);								
    		k++;
    		
    	}
    }   
    
}
