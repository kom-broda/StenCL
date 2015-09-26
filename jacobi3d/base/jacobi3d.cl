#define NX 128
#define NY 128
#define NZ 128

/* Three dimensional Jacobi kernel */
__kernel void _jacobi(__global float * p_set, __global float * p_res) {
    __global float (*set)[NY][NZ] = (__global float (*)[NY][NZ])p_set;
    __global float (*res)[NY][NZ] = (__global float (*)[NY][NZ])p_res;
    int pos_x = get_global_id(0);
    int pos_y = get_global_id(1);
    int pos_z = get_global_id(2);
    
    if(pos_x > 0 && pos_x < NX-1 && pos_y > 0 && pos_y < NY - 1 && pos_z > 0 && pos_z < NZ -1){
    	res[pos_x][pos_y][pos_z] = (set[pos_x - 1][pos_y][pos_z] + set[pos_x + 1][pos_y][pos_z] +
    	                            set[pos_x][pos_y - 1][pos_z] + set[pos_x][pos_y + 1][pos_z] +
    	                            set[pos_x][pos_y][pos_z - 1] + set[pos_x][pos_y][pos_z + 1] +
    	                            set[pos_x][pos_y][pos_z]) * 0.1428571;
  	}
}

/* Updating buffer kernel*/
__kernel void _copy(__global float *p_set, __global float *p_res) {    
    __global float (*set)[NY][NZ] = (__global float (*)[NY][NX])p_set;
    __global float (*res)[NY][NZ] = (__global float (*)[NY][NZ])p_res;
    int pos_x = get_global_id(0);
    int pos_y = get_global_id(1);
    int pos_z = get_global_id(2);
    
    if(pos_x > 0 && pos_x < NX-1 && pos_y > 0 && pos_y < NY - 1 && pos_z > 0 && pos_z < NZ -1){
    	set[pos_x][pos_y][pos_z] = res[pos_x][pos_y][pos_z];
	}
}
