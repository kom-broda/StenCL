#include "ocl_utils.h"
#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>


platform_dev *retrieve_platforms(int *num_platforms)
{
    cl_platform_id *platform_id = NULL;
    cl_int ret;
    int i, ret_num_devices;
    platform_dev *system_=NULL;
    
    ret = clGetPlatformIDs(10, platform_id, num_platforms);
    
    system_ = (platform_dev *)malloc(*num_platforms * sizeof(platform_dev));
    platform_id = (cl_platform_id *)malloc(*num_platforms * sizeof(cl_platform_id));
    
    ret = clGetPlatformIDs(*num_platforms, platform_id, num_platforms);
    
    for(i = 0; i < *num_platforms; i++)
    {
        system_[i].devices = (cl_device_id *)malloc(MAX_DEVICES_PER_PLATFORM*sizeof(cl_device_id));
        system_[i].platform = platform_id[i];
        ret = clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_ALL, MAX_DEVICES_PER_PLATFORM, 
                system_[i].devices, &system_[i].num_dev);
    }
    
    return system_;    
}
