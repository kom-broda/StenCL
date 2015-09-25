#define NX 32768
#define DIM_X 16

/* Monodimensional Jacobian kernel*/
__kernel void _jacobi(__global float *set, __global float *res, __local float *item) {
    int g_x = get_global_id(0);
    int l_x = get_local_id(0);
    
    //Loading data in workgroup memory
    item[l_x] = set[g_x];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(g_x > 0 && g_x < NX - 1){
    	if(l_x > 0 && l_x < DIM_X - 1)
    		res[g_x] = (item[l_x-1] + item[l_x] + item[l_x+1]) * 0.33333333;
    	else
    		res[g_x] = (set[g_x-1] + set[g_x] + set[g_x+1]) * 0.33333333;
    }
}

/* Updating buffer kernel*/
__kernel void _copy(__global float *set, __global float *res) {
    int g_x = get_global_id(0);
    
    if(g_x > 0 && g_x < NX - 1)
    	set[g_x] = res[g_x];
}
