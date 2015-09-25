#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)
#define MAX_NUM_PLATFORMS 2

cl_platform_id *platform_id;
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;

/* Load the kernel source code into the array source_str */
char* read_cl_file(char *kernel_name, size_t *source_size) {

	FILE *fp;
	char *source_str;
	
	fp = fopen(kernel_name, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	
	source_str = (char*) malloc(MAX_SOURCE_SIZE);
	*source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
	
	return source_str;
	
}

/* Initialize OpenCL context */
cl_int cl_initialization(cl_device_id *device_id, cl_context *clGPUContext, cl_command_queue *clCommandQue) {	

	int dim, i, choice;
	size_t *item;
	char str_temp[1024];
	
	// Get platform and device information
	
	//Allocate memory for 2 platforms
	platform_id = (cl_platform_id *) malloc(sizeof(cl_platform_id) * MAX_NUM_PLATFORMS);
	
	errcode = clGetPlatformIDs(MAX_NUM_PLATFORMS, platform_id, &num_platforms);
	if(errcode == CL_SUCCESS) printf("%d platform(s) detected\n",num_platforms);
	else printf("Error getting platform IDs\n");
	
	platform_id = realloc(platform_id, sizeof(cl_platform_id) * num_platforms);
	
	for(i = 0; i < num_platforms; i++){
		errcode = clGetPlatformInfo(platform_id[i],CL_PLATFORM_NAME, sizeof(str_temp), str_temp,NULL);
		if(errcode == CL_SUCCESS) printf("Platform n. %d name is %s\n",i + 1, str_temp);
		else printf("Error getting platform name\n");
		
		errcode = clGetPlatformInfo(platform_id[i], CL_PLATFORM_VERSION, sizeof(str_temp), str_temp,NULL);
		if(errcode == CL_SUCCESS) printf("Platform n. %d version is %s\n",i + 1, str_temp);
		else printf("Error getting platform version\n");
	}
	
	if(num_platforms > 1){
		do{
			printf("On which platform you want to run the kernel?\n");
			scanf("%d", &choice);
			if(choice > num_platforms || choice <= 0) printf("Invalid choice!\n");
		} while (choice > num_platforms || choice <= 0);	
		choice--;
	} else choice = 0;
	
	errcode = clGetDeviceIDs(platform_id[choice], CL_DEVICE_TYPE_ALL, 1, device_id, &num_devices);
	if(errcode == CL_SUCCESS) printf("Number of devices is %d\n", num_devices);
	else printf("Error getting device IDs\n");

	errcode = clGetDeviceInfo(*device_id,CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("Device name is %s\n",str_temp);
	else printf("Error getting device name\n");
	
	errcode = clGetDeviceInfo(*device_id,CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(int *), (void *)&dim,NULL);
	if(errcode == CL_SUCCESS) printf("Device max workgroup size is %d \n",dim);
	else printf("Error getting device max workgroup size\n");
	
	errcode = clGetDeviceInfo(*device_id,CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(int *), (void *)&dim,NULL);
	if(errcode == CL_SUCCESS) printf("Device max work item dimensions is %d \n",dim);
	else printf("Error getting device work item dimensions\n");
	
	item=(size_t *)malloc(dim*(sizeof(size_t)));
	
	errcode = clGetDeviceInfo(*device_id,CL_DEVICE_MAX_WORK_ITEM_SIZES, dim*sizeof(size_t), item,NULL);
	if(errcode == CL_SUCCESS) printf("Device max work item sizes is %d*%d*%d \n",(int)item[0], (int)item[1], (int)item[2]);
	else printf("Error getting device max workitem sizes\n");
	
	// Create an OpenCL context
	*clGPUContext = clCreateContext( NULL, 1, device_id, NULL, NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating context\n");
 
	//Create a command-queue
	*clCommandQue = clCreateCommandQueue(*clGPUContext, *device_id, 0, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");
	
	return errcode;
	
}

/* Load OpenCL program from kernel source */
cl_int cl_load_prog(cl_program *clProgram, cl_context *clGPUContext, cl_device_id *device_id, char *source_str, size_t source_size) {
	cl_int errcode;
	
	// Create a program from the kernel source
	*clProgram = clCreateProgramWithSource(*clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating program\n");
	// Build the program
	errcode = clBuildProgram(*clProgram, 1, device_id, NULL, NULL, NULL);
	switch(errcode){
	    case CL_INVALID_PROGRAM:
	        printf("Invalid PROGRAM\n");
	        break;
        case CL_INVALID_VALUE:
	        printf("Invalid VALUE\n");
	        break;
        case CL_INVALID_DEVICE:
	        printf("Invalid DEVICE\n");
	        break;
        case CL_INVALID_BUILD_OPTIONS:
	        printf("Invalid BUILD OPT\n");
	        break;
        case CL_INVALID_OPERATION:
	        printf("Invalid OPERATION\n");
	        break;
        case CL_BUILD_PROGRAM_FAILURE:
	        printf("Invalid BUILD FAILURE\n");
	        size_t log_size;
            clGetProgramBuildInfo(*clProgram, *device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

            char *log = (char *) malloc(log_size);

            // Get the log
            clGetProgramBuildInfo(*clProgram, *device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

            // Print the log
            printf("%s\n", log);
	        break;
        case CL_COMPILER_NOT_AVAILABLE:
	        printf("Invalid COMPILER\n");
	        break;
        default:
            ;
	}	

	return errcode;
}	

/* Initialize 1D array with random values */
void init_1d_array(const size_t DIM_X, float set[DIM_X]) {	
	int i;
	
	srand(time(NULL));
	for(i = 0; i < DIM_X; i++)
		set[i] = rand() % 1024;	
}

/* Initialize 2D array with random values */
void init_2d_array(const size_t DIM_X, const size_t DIM_Y, float set[DIM_X][DIM_Y]) {
	int i, j;
	
	srand(time(NULL));
	for(i = 0; i < DIM_X; i++)
		for(j = 0; j < DIM_Y; j++)
			set[i][j] = rand() % 1024;
	
	return;
}

/* Initialize 3D array with random values */
void init_3d_array(const size_t DIM_X, const size_t DIM_Y, const size_t DIM_Z, float set[DIM_X][DIM_Y][DIM_Z]) {
	int i, j, k;
	
	srand(time(NULL));
	for(i = 0; i < DIM_X; i++)
		for(j = 0; j < DIM_Y; j++)
			for(k = 0; k < DIM_Z; k++)
				set[i][j][k] = rand() % 1024;
				
	return;
}

/* Copy the 2D array src into the 2D array dst */
void memcpy_2d(const size_t DIM_X, const size_t DIM_Y, float src[DIM_X][DIM_Y], float dst[DIM_X][DIM_Y]){
	int i, j;
	
	for(i = 0; i < DIM_X; i++)
		for(j = 0; j < DIM_Y; j++)
			dst[i][j] = src[i][j];
}

/* Copy the 3D array src into the 3D array dst */
void memcpy_3d(const size_t DIM_X, const size_t DIM_Y, const size_t DIM_Z, float src[DIM_X][DIM_Y][DIM_Z], float dst[DIM_X][DIM_Y][DIM_Z]) {
	int i, j, k;
	
	for(i = 0; i < DIM_X; i++)
		for(j = 0; j < DIM_Y; j++)
			for(k = 0; k < DIM_Z; k++)
				dst[i][j][k] = src[i][j][k];
}

/* Compute the difference between a and b in percentage */
float percent_diff(float a, float b)
{
	float tmp;
	
	tmp = abs(a - b);
	tmp = tmp / ((a+b)/2);
	tmp = tmp * 100;
	
	return tmp;	
}

/* Compare the 1D arrays a and b */
void compare_results_1d(const size_t DIM_X, float* a, float* b, float threshold) {	
	int i, fail;
	fail = 0;
	
	for(i = 0; i < DIM_X; i++){
		if(percent_diff(a[i], b[i]) > threshold){
			fail++;
		}		
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold: %d over %lu\n", fail, (long unsigned int) DIM_X);
}

/* Compare the 2D arrays a and b */
void compare_results_2d(const size_t DIM_X, const size_t DIM_Y, float a[DIM_X][DIM_Y], float b[DIM_X][DIM_Y], float threshold) {	
	int i, j, fail;
	fail = 0;
	
	for(i = 0; i < DIM_X; i++){
		for(j = 0; j < DIM_Y; j++){
			if(percent_diff(a[i][j], b[i][j]) > threshold){
				fail++;
			}	
		}			
	}

	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold: %d over %lu\n", fail, (long unsigned int) DIM_X*DIM_Y);
}

/* Compare the 3D arrays a and b */
void compare_results_3d(const size_t DIM_X, const size_t DIM_Y, const size_t DIM_Z, float a[DIM_X][DIM_Y][DIM_Z], float b[DIM_X][DIM_Y][DIM_Z], float threshold) {	
	int i, j, k, fail;
	fail = 0;
	
	for(i = 0; i < DIM_X; i++){
		for(j = 0; j < DIM_Y; j++){
			for(k = 0; k < DIM_Z; k++){
				if(percent_diff(a[i][j][k], b[i][j][k]) > threshold){
					fail++;
				}		
			}
		}		
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold: %d over %lu\n", fail, (long unsigned int) DIM_X*DIM_Y*DIM_Z);
}
