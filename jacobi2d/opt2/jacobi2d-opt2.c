/**
*
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "../../include/stencl.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD .5

/* Problem size */
//Number of iterations per execution
#define TMAX 100
//Working set sizes on each dimension
#define NX 1024
#define NY 1024
//Workgroup size on each dimension
#define DIM_X 8
#define DIM_Y 8

#define MAX_SOURCE_SIZE (0x100000)

#define NUM_PLATFORMS 2

cl_device_id device_id;   
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel1;
cl_kernel clKernel2;
cl_command_queue clCommandQue;
cl_program clProgram;

cl_mem cl_set;
cl_mem cl_res;
float point = 0;
char *source_str;
size_t source_size;

size_t local_dim;
size_t global_dim;

void cl_mem_init(const size_t DIMX, const size_t DIMY, float set[DIMX][DIMY], float res[DIMX][DIMY]) {    
    /*
        OpenCL buffers initialization
        Example: fict_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(float) * TMAX, NULL, &errcode);
    */
	cl_set = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, NX*NY*sizeof(float), NULL, &errcode);
	cl_res = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, NX*NY*sizeof(float), NULL, &errcode);	
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

    /*
        OpenCL buffers writing
        Example: errcode = clEnqueueWriteBuffer(clCommandQue, fict_mem_obj, CL_TRUE, 0, sizeof(float) * TMAX, _fict_, 0, NULL, NULL);
    */
	errcode = clEnqueueWriteBuffer(clCommandQue, cl_set, CL_TRUE, 0, NX*NY*sizeof(float), set, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
}

void load_kernel() {
	// Create OpenCL kernels
	clKernel1 = clCreateKernel(clProgram, "_jacobi", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");
	
	clKernel2 = clCreateKernel(clProgram, "_copy", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel2\n");
}

void sequential_jacobi_2d(const size_t DIMX, const size_t DIMY, float set[DIMX][DIMY], float res[DIMX][DIMY]){
	int t, i, j;
	for (t = 0; t < TMAX; t++)
	{
		for (i = 0; i < NX; i++){
		    for (j = 0; j < NY; j++){
				if(i == 0 || j == 0|| i == NX - 1 || j == NY - 1){
	                res[i][j] = set[i][j];
	            }else{
	                res[i][j] = (set[i][j] +
                                 set[i][j-1] + set[i][j+1] +
                                 set[i-1][j]+ set[i+1][j]) * 0.2;                       
	            }
		    }
		}
		
		for (i = 0; i < NX; i++){
		    for (j = 0; j < NY; j++){
	               set[i][j] = res[i][j];
		    }
		}
	}
	
	return;
}

void cl_launch_kernel() {
	int i;
	size_t globalWorkSize[] = {NX , NY};
	size_t localWorkSize[] = {DIM_X , DIM_Y};
	
	for(i = 0; i < TMAX; i++){
		/* Set kernel arguments
	   	Example: errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&fict_mem_obj);
		*/
		errcode = clSetKernelArg(clKernel1, 0, sizeof(cl_mem), &cl_set);
		errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), &cl_res);
		errcode |= clSetKernelArg(clKernel1, 2, DIM_X*DIM_Y*sizeof(float), NULL);
		
		switch(errcode){
		    case CL_INVALID_KERNEL:
		        printf("Invalid kernel\n");
		        break;
	        case CL_INVALID_ARG_INDEX:
	            printf("Invalid arg index\n");
	            break;
            case CL_INVALID_ARG_VALUE:
	            printf("Invalid arg value\n");
	            break;
            case CL_INVALID_MEM_OBJECT:
	            printf("Invalid mem object\n");
	            break;
            case CL_INVALID_SAMPLER:
	            printf("Invalid sampler\n");
	            break;
            case CL_INVALID_ARG_SIZE:
	            printf("Invalid arg size\n");
	            break;
		}
		
		if(errcode != CL_SUCCESS) printf("Error in setting arguments to kernel1 - cycle %d\n",i);

		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);		
		if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
		
		clEnqueueBarrierWithWaitList(clCommandQue, 0, NULL, NULL);
		
		errcode = clSetKernelArg(clKernel2, 0, sizeof(cl_mem), &cl_set);
		errcode |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), &cl_res);
		if(errcode != CL_SUCCESS) printf("Error in setting arguments to kernel2 - cycle %d\n",i);

		errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		if(errcode != CL_SUCCESS) printf("Error in launching kernel2 - cycle %d\n",i);
		
		switch(errcode){
		    case CL_INVALID_KERNEL:
		        printf("Invalid kernel\n");
		        break;
	        case CL_INVALID_PROGRAM_EXECUTABLE:
		        printf("Invalid program exe\n");
		        break;
	        case CL_INVALID_COMMAND_QUEUE:
		        printf("Invalid COMM QUEUE\n");
		        break;
	        case CL_INVALID_KERNEL_ARGS:
		        printf("Invalid KER ARGS\n");
		        break;	        
		}
		
		clFinish(clCommandQue);	
	}
}

void cl_clean_up() {
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode |= clFinish(clCommandQue);
	errcode |= clReleaseKernel(clKernel1);
	errcode |= clReleaseKernel(clKernel2);
	errcode |= clReleaseProgram(clProgram);
	errcode |= clReleaseMemObject(cl_set);
	errcode |= clReleaseMemObject(cl_res);
	errcode |= clReleaseCommandQueue(clCommandQue);
	errcode |= clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}

int main(void) 
{	
	// arrays/matrices to be manipulated
	static float set[NX][NY];
	static float cpy[NX][NY];
	static float res[NX][NY];
	static float res_gpu[NX][NY];
	int i, j;
	
	float cl_ex_time;
	float seq_ex_time;
	struct timespec tstart = {0,0}, tend = {0,0};
	
	init_2d_array(NX, NY, set);
	memcpy_2d(NX, NY, set, cpy);	   
	
	printf("\n*********2D Jacobian stencil - Opt-2*********\n\n");
	errcode = cl_initialization(&device_id, &clGPUContext, &clCommandQue);
	if(errcode != CL_SUCCESS) exit(1);
	
	/*OpenCL execution*/
	source_str = read_cl_file("jacobi2d-opt2.cl", &source_size);		
	errcode = cl_load_prog(&clProgram, &clGPUContext, &device_id, source_str, source_size);
	if(errcode != CL_SUCCESS) exit(1);
	load_kernel();
	cl_mem_init(NX, NY, set, res);
	
	cl_ulong size;
	clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);
	
	clock_gettime(CLOCK_MONOTONIC, &tstart);
	
	cl_launch_kernel();
	errcode = clEnqueueReadBuffer(clCommandQue, cl_set, CL_TRUE, 0, NX*NY*sizeof(float), res_gpu, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");
	
	clock_gettime(CLOCK_MONOTONIC, &tend);
	cl_ex_time = (float) (((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - 
				((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
    
	/*Sequential execution*/
	clock_gettime(CLOCK_MONOTONIC, &tstart);
	sequential_jacobi_2d(NX, NY, cpy, res);
	clock_gettime(CLOCK_MONOTONIC, &tend);	  
	seq_ex_time = (float) (((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - 
		      	 ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));		
					
	printf("\nParallel time:\t\t%.5f s\n", cl_ex_time);
	printf("Sequential time:\t%.5f s\n", seq_ex_time);
	printf("Speedup:\t\t%.1fx\n", seq_ex_time/cl_ex_time);
	
	for(i = 0; i < NX; i++)
		for(j = 0; j < NY; j++)
			if (res_gpu[i][j]==0 && res[i][j] !=0){
				printf("%d %d\n", i, j);
			}
	
	compare_results_2d(NX, NY, res, res_gpu, PERCENT_DIFF_ERROR_THRESHOLD);		
	
	cl_clean_up();
		
   	return 0;
}
