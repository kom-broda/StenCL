#define NX 32768

/* Monodimensional Jacobian kernel*/
__kernel void _jacobi(__global float *set, __global float *res) {
    int pos_x = get_global_id(0);
    
    if(pos_x > 0 && pos_x < NX - 1){
    	res[pos_x] = (set[pos_x-1] + set[pos_x] + set[pos_x+1]) * 0.33333333;
    }
}

/* Updating buffer kernel*/
__kernel void _copy(__global float *set, __global float *res) {
    int pos_x = get_global_id(0);
    
    if(pos_x > 0 && pos_x < NX - 1)
    	set[pos_x] = res[pos_x];
}
