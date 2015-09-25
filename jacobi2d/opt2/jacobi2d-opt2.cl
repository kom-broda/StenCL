#define NX 1024
#define NY 1024
#define DIM_X 8
#define DIM_Y 8

/* Three dimensional Jacobi kernel */
__kernel void _jacobi(__global float * p_set, __global float * p_res, __local float *p_item) {
    __global float (*set)[NY] = (__global float (*)[NY])p_set;
    __global float (*res)[NY] = (__global float (*)[NY])p_res;
    __local float (*item)[DIM_Y] = (__local float (*)[DIM_Y])p_item;    
    int g_x = get_global_id(0);
    int g_y = get_global_id(1);
    int l_x = get_local_id(0);
    int l_y = get_local_id(1);
    
    //Loading data in workgroup memory
    item[l_x][l_y] = set[g_x][g_y];
    barrier(CLK_LOCAL_MEM_FENCE);	

    if(g_x > 0 && g_x < NX-1 && g_y > 0 && g_y < NY - 1){
    	if(l_x > 0 && l_y > 0 && l_x < DIM_X - 1 && l_y < DIM_Y - 1){
			res[g_x][g_y] = (item[l_x - 1][l_y] + item[l_x + 1][l_y] +
			                     item[l_x][l_y - 1] + item[l_x][l_y + 1] +
			                     item[l_x][l_y]) * 0.2;
		}else{
			res[g_x][g_y] = (set[g_x - 1][g_y] + set[g_x + 1][g_y] +
								set[g_x][g_y - 1] + set[g_x][g_y + 1] +
								set[g_x][g_y]) * 0.2;
		}
    }
}

/* Updating buffer kernel*/
__kernel void _copy(__global float *p_set, __global float *p_res) {    
    __global float (*set)[NY] = (__global float (*)[NY])p_set;
    __global float (*res)[NY] = (__global float (*)[NY])p_res;
    int g_x = get_global_id(0);
    int g_y = get_global_id(1);
    
    if(g_x > 0 && g_x < NX - 1 && g_y > 0 && g_y < NY - 1)
    	set[g_x][g_y] = res[g_x][g_y];
}
