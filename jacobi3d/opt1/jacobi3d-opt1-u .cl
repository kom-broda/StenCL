#define NX 64
#define NY 64
#define NZ 64

#define DEPTH 4

/* Three dimensional Jacobi kernel */
__kernel void _jacobi(__global float *p_set, __global float *p_res) {
    int pos_z = get_global_id(2);
    
    if(pos_z % DEPTH == 0){    
        __global float (*set)[NY][NZ] = (__global float (*)[NY][NZ])p_set;
        __global float (*res)[NY][NZ] = (__global float (*)[NY][NZ])p_res;        
        int pos_x = get_global_id(0);
        int pos_y = get_global_id(1);
        int i;
        
        float reg[] = {set[pos_x][pos_y][pos_z], set[pos_x][pos_y][pos_z + 1], 
                       set[pos_x][pos_y][pos_z + 2], set[pos_x][pos_y][pos_z + 3]};
        
		if(pos_x > 0 && pos_y > 0 && pos_x < NX - 1 && pos_y < NY - 1){
			if(pos_z != 0){		
				res[pos_x][pos_y][pos_z] = (set[pos_x - 1][pos_y][pos_z] + set[pos_x + 1][pos_y][pos_z] +
										   set[pos_x][pos_y][pos_z - 1] + reg[0] + reg[1] +
										   set[pos_x][pos_y - 1][pos_z] + set[pos_x][pos_y + 1][pos_z]) * 0.142857;
			}
			
			pos_z++;
		
			res[pos_x][pos_y][pos_z] = (set[pos_x - 1][pos_y][pos_z] + set[pos_x + 1][pos_y][pos_z] +
										   reg[0] + reg[1] + reg[2] +
										   set[pos_x][pos_y - 1][pos_z] + set[pos_x][pos_y + 1][pos_z]) * 0.142857;
			pos_z++;		
		
			res[pos_x][pos_y][pos_z] = (set[pos_x - 1][pos_y][pos_z] + set[pos_x + 1][pos_y][pos_z] +
										   reg[1] + reg[2] + reg[3] +
										   set[pos_x][pos_y - 1][pos_z] + set[pos_x][pos_y + 1][pos_z]) * 0.142857;
			pos_z++;
			if(pos_z != NZ - 1){
				res[pos_x][pos_y][pos_z] = (set[pos_x - 1][pos_y][pos_z] + set[pos_x + 1][pos_y][pos_z] +
										   	   reg[2] + reg[3] + set[pos_x][pos_y][pos_z + 1] +
										  	   set[pos_x][pos_y - 1][pos_z] + set[pos_x][pos_y + 1][pos_z]) * 0.142857;						    			}	
    	}
    }
}

/* Updating buffer kernel*/
__kernel void _copy(__global float *p_set, __global float *p_res) {
    __global float (*set)[NY][NZ] = (__global float (*)[NY][NZ])p_set;
    __global float (*res)[NY][NZ] = (__global float (*)[NY][NZ])p_res;    
    int pos_x = get_global_id(0);
    int pos_y = get_global_id(1);
    int pos_z = get_global_id(2);
    
	if(pos_x > 0 && pos_y > 0 && pos_z > 0 && pos_x < NX - 1 && pos_y < NY - 1 && pos_z < NZ - 1)
    	set[pos_x][pos_y][pos_z] = res[pos_x][pos_y][pos_z];
}
