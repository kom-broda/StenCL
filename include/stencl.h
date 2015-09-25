#ifndef __STENCL_H__
#define __STENCL_H__

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/* Load the kernel source code into the array source_str */
char * read_cl_file(char *kernel_name, size_t *source_size);

/* Initialize OpenCL context */
cl_int cl_initialization(cl_device_id *device_id, cl_context *clGPUContext, cl_command_queue *clCommandQue);

/* Load OpenCL program from kernel source */
cl_int cl_load_prog(cl_program *clProgram, cl_context *clGPUContext, cl_device_id *device_id, char *source_str, size_t source_size);

/* Initialize 1D array with random values */
void init_1d_array(const size_t DIM_X, float set[DIM_X]);

/* Initialize 2D array with random values */
void init_2d_array(const size_t DIM_X, const size_t DIM_Y, float set[DIM_X][DIM_Y]);

/* Initialize 3D array with random values */
void init_3d_array(const size_t DIM_X, const size_t DIM_Y, const size_t DIM_Z, float set[DIM_X][DIM_Y][DIM_Z]);

/* Copy the 2D array src into the 2D array dst */
void memcpy_2d(const size_t DIM_X, const size_t DIM_Y, float src[DIM_X][DIM_Y], float dst[DIM_X][DIM_Y]);

/* Copy the 3D array src into the 3D array dst */
void memcpy_3d(const size_t DIM_X, const size_t DIM_Y, const size_t DIM_Z, float src[DIM_X][DIM_Y][DIM_Z], float dst[DIM_X][DIM_Y][DIM_Z]);

/* Compare the 1D arrays a and b */
void compare_results_1d(const size_t DIM_X, float* a, float* b, float threshold);

/* Compare the 2D arrays a and b */
void compare_results_2d(const size_t DIM_X, const size_t DIM_Y, float a[DIM_X][DIM_Y], float b[DIM_X][DIM_Y], float threshold);

/* Compare the 3D arrays a and b */
void compare_results_3d(const size_t DIM_X, const size_t DIM_Y, const size_t DIM_Z, float a[DIM_X][DIM_Y][DIM_Z], float b[DIM_X][DIM_Y][DIM_Z], float threshold);

#endif /* __STENCL_H__ */
