#define NX 64
#define NY 64
#define NZ 64

#define DEPTH 4

/* Three dimensional Jacobi kernel */
__kernel void _jacobi(__global float *p_set, __global float *p_res) {
    int z = get_global_id(2);
    int y = get_global_id(1);
    
    if(z % DEPTH == 0 && y % DEPTH == 0){    
        __global float (*set)[NY][NZ] = (__global float (*)[NY][NZ])p_set;
        __global float (*res)[NY][NZ] = (__global float (*)[NY][NZ])p_res;        
        int x = get_global_id(0);
        int i, j;
        
        float reg[DEPTH][DEPTH] = {{set[x][y][z], set[x][y][z+1], set[x][y][z+2], set[x][y][z+3]},
					 {set[x][y+1][z], set[x][y+1][z+1], set[x][y+1][z+2], set[x][y+1][z+3]},
					 {set[x][y+2][z], set[x][y+2][z+1], set[x][y+2][z+2], set[x][y+2][z+3]},
					 {set[x][y+3][z], set[x][y+3][z+1], set[x][y+3][z+2], set[x][y+3][z+3]}};
        
        if(x > 0 && x < NX - 1){
        	//y = 0 ; 0 <= z < 4
        	if(y != 0){
        		if(z != 0){
        			res[x][y][z] = (set[x-1][y][z] + set[x+1][y][z] +
        						set[x][y-1][z] + reg[1][0] +
        						set[x][y][z-1] + reg[0][1] +
        						reg[0][0]) * 0.142857;
        		}
        		z++;
        		
        		res[x][y][z] = (set[x-1][y][z] + set[x+1][y][z] +
        					set[x][y-1][z] + reg[1][1] +
        					reg[0][0] + reg[0][2] +
        					reg[0][1]) * 0.142857;
        		z++;
        		
        		res[x][y][z] = (set[x-1][y][z] + set[x+1][y][z] +
        					set[x][y-1][z] + reg[1][2] +
        					reg[0][1] + reg[0][3] +
        					reg[0][2]) * 0.142857;
        		z++;
        		
        		if(z != NZ - 1){
        			res[x][y][z] = (set[x-1][y][z] + set[x+1][y][z] +
        					set[x][y-1][z] + reg[1][3] +
        					reg[0][2] + set[x][y][z+1] +
        					reg[0][3]) * 0.142857;
        		}
        		
        		z = z - (DEPTH - 1);
        	}
        	        	
        	y++;
        	
        	//y = 1 ; 0 <= z < 4
        	if(z != 0){
        		res[x][y][z] = (set[x-1][y][z] + set[x+1][y][z] +
        					set[x][y-1][z] + reg[2][0] +
        					set[x][y][z-1] + reg[1][1] +
        					reg[1][0]) * 0.142857;
        	}
        	z++;
        	
        	res[x][y][z] = (set[x-1][y][z] + set[x+1][y][z] +
        				set[x][y-1][z] + reg[2][1] +
        				reg[1][0] + reg[1][2] +
        				reg[1][1]) * 0.142857;
        	z++;
        	
        	res[x][y][z] = (set[x-1][y][z] + set[x+1][y][z] +
        				set[x][y-1][z] + reg[2][2] +
        				reg[1][1] + reg[1][3] +
        				reg[1][2]) * 0.142857;
        	z++;
        		
        	if(z != NZ - 1){
        		res[x][y][z] = (set[x-1][y][z] + set[x+1][y][z] +
        					set[x][y-1][z] + reg[2][3] +
        					reg[1][2] + set[x][y][z+1] +
        					reg[1][3]) * 0.142857;
        	}
        		
        	z = z - (DEPTH - 1);
        	y++;
        	
        	//y = 2 ; 0 <= z < 4
        	if(z != 0){
        		res[x][y][z] = (set[x-1][y][z] + set[x+1][y][z] +
        					set[x][y-1][z] + reg[3][0] +
        					set[x][y][z-1] + reg[2][1] +
        					reg[2][0]) * 0.142857;
        	}
        	z++;
        		
        	res[x][y][z] = (set[x-1][y][z] + set[x+1][y][z] +
        				set[x][y-1][z] + reg[3][1] +
        				reg[2][0] + reg[2][2] +
        				reg[2][1]) * 0.142857;
        	z++;
        		
        	res[x][y][z] = (set[x-1][y][z] + set[x+1][y][z] +
        				set[x][y-1][z] + reg[3][2] +
        				reg[2][1] + reg[2][3] +
        				reg[2][2]) * 0.142857;
        	z++;
        		
        	if(z != NZ - 1){
        		res[x][y][z] = (set[x-1][y][z] + set[x+1][y][z] +
        					set[x][y-1][z] + reg[3][3] +
        					reg[2][2] + set[x][y][z+1] +
        					reg[2][3]) * 0.142857;
        	}
        		
        	z = z - (DEPTH - 1);        	        	
        	y++;
        	
        	//y = 3 ; 0 <= z < 4
        	if(y != NY - 1){
		    	if(z != 0){
		    		res[x][y][z] = (set[x-1][y][z] + set[x+1][y][z] +
		    					set[x][y-1][z] + set[x][y+1][z] +
		    					set[x][y][z-1] + reg[3][1] +
		    					reg[3][0]) * 0.142857;
		    	}
		    	z++;
		    		
		    	res[x][y][z] = (set[x-1][y][z] + set[x+1][y][z] +
		    				set[x][y-1][z] + set[x][y+1][z] +
		    				reg[3][0] + reg[3][2] +
		    				reg[3][1]) * 0.142857;
		    	z++;
		    		
		    	res[x][y][z] = (set[x-1][y][z] + set[x+1][y][z] +
		    				set[x][y-1][z] + set[x][y+1][z] +
		    				reg[3][1] + reg[3][3] +
		    				reg[3][2]) * 0.142857;
		    	z++;
		    		
		    	if(z != NZ - 1){
		    		res[x][y][z] = (set[x-1][y][z] + set[x+1][y][z] +
		    					set[x][y-1][z] + set[x][y+1][z] +
		    					reg[3][2] + set[x][y][z+1] +
		    					reg[3][3]) * 0.142857;
		    	}
        	}
        }
    }
    
    //TODO bien?
}

/* Updating buffer kernel*/
__kernel void _copy(__global float *p_set, __global float *p_res) {
    __global float (*set)[NY][NZ] = (__global float (*)[NY][NZ])p_set;
    __global float (*res)[NY][NZ] = (__global float (*)[NY][NZ])p_res;    
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    
	if(x > 0 && y > 0 && z > 0 && x < NX - 1 && y < NY - 1 && z < NZ - 1)
    	set[x][y][z] = res[x][y][z];
}
