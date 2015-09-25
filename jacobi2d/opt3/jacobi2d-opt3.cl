#define NX 1024
#define NY 1024

#define DEPTH 4

/* Three dimensional Jacobi kernel */
__kernel void _jacobi(__global float * p_set, __global float * p_res) {        
    int pos_y = get_global_id(1);
    
    if(pos_y % DEPTH == 0){
    	__global float (*set)[NY] = (__global float (*)[NY])p_set;
   		__global float (*res)[NY] = (__global float (*)[NY])p_res;
    	int pos_x = get_global_id(0);
    	
    	float reg[] = {set[pos_x][pos_y], set[pos_x][pos_y + 1],
    				   set[pos_x][pos_y + 2], set[pos_x][pos_y + 3]};
    	
    	if(pos_x > 0 && pos_x < NX - 1){
    		if(pos_y != 0)
    			res[pos_x][pos_y] = (set[pos_x - 1][pos_y] + set[pos_x + 1][pos_y] +
    	        	            	 set[pos_x][pos_y - 1] + reg[0] + reg[1]) * 0.2;    	        	            	 
    	    pos_y++;
    	    
    	    res[pos_x][pos_y] = (set[pos_x - 1][pos_y] + set[pos_x + 1][pos_y] +
    	        	             reg[0] + reg[1] + reg[2]) * 0.2;
    	    pos_y++;
    	    
    	    res[pos_x][pos_y] = (set[pos_x - 1][pos_y] + set[pos_x + 1][pos_y] +
    	        	             reg[1] + reg[2] + reg[3]) * 0.2;
    	    pos_y++;
    	    
    	    if(pos_y != NY - 1)
    	    	res[pos_x][pos_y] = (set[pos_x - 1][pos_y] + set[pos_x + 1][pos_y] +
    	        	            	 reg[2] + reg[3] + set[pos_x][pos_y + 1]) * 0.2;    	        	       	
    	        	         
    	}    	
	}
}

/* Updating buffer kernel*/
__kernel void _copy(__global float *p_set, __global float *p_res) {    
    __global float (*set)[NY] = (__global float (*)[NY])p_set;
    __global float (*res)[NY] = (__global float (*)[NY])p_res;
    int pos_x = get_global_id(0);
    int pos_y = get_global_id(1);
    
    if(pos_x > 0 && pos_y > 0 && pos_x < NX - 1 && pos_y < NY - 1)
    	set[pos_x][pos_y] = res[pos_x][pos_y];
}
