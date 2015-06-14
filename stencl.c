#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "ocl_utils.h"

#include <CL/cl.h>



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
    //ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
}
