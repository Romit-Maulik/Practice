/*
Compile using nvcc cuda_heat.cu
Author: Romit Maulik - romit.maulik@okstate.edu
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

const double PI = 3.1415926535;
const double lx = 2.0*PI, ly = 2.0*PI;
const int nx = 254, ny = 254;
const double ALPHA = 0.8, STAB_PARAM = 0.8;
const double dx = lx/nx, dy = ly/ny;

/*
Host Functions
*/
void initialize_array(double *u);
void write_array(double *u);
/*
Device Functions
*/
__global__ void update_solution(double *_u, double *_utemp, double *_const_mult);
__global__ void update_periodic_boundaries(double *_u);
__global__ void copy_arrays(double *_u, double *_utemp);



int main (int argc, char** argv)
{
	double *u = new double [(nx+2)*(ny+2)];

	//double *u = malloc(sizeof(double) * (nx+2) * (ny+2)); //Pointer to host memory
	double *_u, *_utemp;			//Pointer to device memory
	double *const_mult = new double[1];
	double *_const_mult;

	initialize_array(u);      //Initialize solution on host

	// allocate storage space on the GPU
	cudaMalloc((void **)&_u, (nx+2) * (ny+2) * sizeof(double));
    cudaMalloc((void **)&_utemp, (nx+2) * (ny+2) * sizeof(double));
    cudaMalloc((void **)&_const_mult,sizeof(double));

    //Copy data to device
	cudaMemcpy(_u,u,(nx+2)*(ny+2)*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(_utemp,u,(nx+2)*(ny+2)*sizeof(double),cudaMemcpyHostToDevice);

    // assign a 2D distribution of CUDA "threads" within each CUDA "block"    
    int ThreadsPerBlock=16;
    dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );

    // calculate number of blocks along X and Y in a 2D CUDA "grid"
    dim3 dimGrid( ceil(float(nx+2)/float(dimBlock.x)), ceil(float(ny+2)/float(dimBlock.y)), 1 );
	
	double t,dt;
	const double ft=1.0;
	
	dt = STAB_PARAM*dx*dx/(4.0*ALPHA);
	((*const_mult)) = ALPHA*dt/(dx*dx);

	//Copy constant to device
	cudaMemcpy(_const_mult,const_mult,sizeof(double),cudaMemcpyHostToDevice);

	clock_t start, end;
	double cpu_time_used;

	start = clock();

	//FTCS integration - CUDA
	//Boundary conditions
	update_periodic_boundaries<<<dimGrid, dimBlock>>>(_u);
	cudaThreadSynchronize();

	update_periodic_boundaries<<<dimGrid, dimBlock>>>(_utemp);
	cudaThreadSynchronize();

	t = 0.0;
	do{
		
		//Update solution
		update_solution<<<dimGrid, dimBlock>>>(_u,_utemp,_const_mult);   // update T1 using data stored in T2
		cudaThreadSynchronize();

		//Boundary conditions
		update_periodic_boundaries<<<dimGrid, dimBlock>>>(_u);
		cudaThreadSynchronize();

		//Copy arrays
		copy_arrays<<<dimGrid, dimBlock>>>(_u,_utemp);
		cudaThreadSynchronize();

		t = t + dt;
	}while(t<ft);

	// copy final array to the CPU from the GPU 
    cudaMemcpy(u,_u,(nx+2)*(ny+2)*sizeof(double),cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

	printf("CPU time used = %f\n", cpu_time_used);

    //Write temperature to disk
    write_array(u);

    // release memory on the host 
    delete u;

    // release memory on the device 
    cudaFree(_u);
    cudaFree(_utemp);


	return 0;
}

void initialize_array(double *u)
{
	double x,y;

	for (int i = 1; i < nx+1; i++)
	{
		for (int j = 1; j < ny+1; j++)
		{
			x = (double) (i-1)/nx * 2.0 * PI;
			y = (double) (j-1)/ny * 2.0 * PI;

			(*(u + (ny+2)*i + j)) = sin(x+y);
		}
	}
}


__global__ void update_periodic_boundaries(double *_u)
{
    // compute the "i" and "j" location of the node point
    // handled by this thread

    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    int j = blockIdx.y * blockDim.y + threadIdx.y ;

    // Left boundary
    if(j>0 && j<ny+1 && i == 0) 
    {
    	(*(_u + j)) = (*(_u + (ny+2)*nx + j));//Correct

    }

    // Right boundary
    if(j>0 && j<ny+1 && i == nx + 1) 
    {
    	(*(_u + (ny+2)*(nx+1) + j)) = (*(_u + (ny+2) + j));//Correct
    }

    // Bottom boundary
    if(i>=0 && i<nx+2 && j == 0)
    {
    	(*(_u + (ny+2)*i)) = (*(_u + (ny+2)*i + ny));	//Correct
    }

    // top boundary
    if(i>=0 && i<nx+2 && j == ny + 1)
    {
    	(*(_u + (ny+2)*i + ny + 1)) = (*(_u + (ny+2)*i + 1));	//Correct
    	
    }
   
}

__global__ void update_solution(double *_u, double *_utemp, double *_const_mult)
{
    // compute the "i" and "j" location of the node point
    // handled by this thread

    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    int j = blockIdx.y * blockDim.y + threadIdx.y ;

    // get the natural index values of node (i,j) and its neighboring nodes
	int p = (ny+2)*i + j; 
	int n = (ny+2)*i + j + 1;
	int s = (ny+2)*i + j - 1;
	int w = (ny+2)*(i-1) + j;
	int e = (ny+2)*(i+1) + j;

    // only update "interior" node points
    if(i>0 && i<nx+1 && j>0 && j<ny+1) 
    {
        (*(_u + p)) = (*(_utemp + p)) + (*_const_mult)*(-4.0*(*(_utemp + p)) + (*(_utemp + n)) + (*(_utemp + s)) + (*(_utemp + w)) + (*(_utemp + e)));
    }
}

__global__ void copy_arrays(double *_u, double *_utemp)
{

    // compute the "i" and "j" location of the node point
    // handled by this thread

    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    int j = blockIdx.y * blockDim.y + threadIdx.y ;

	(*(_utemp + (ny+2)*i + j)) = (*(_u + (ny+2)*i + j));
}

void write_array(double *u)
{
	FILE *fp;
	fp = fopen("Temperature.txt","wb");

	for(int i=0;i<nx+2;i++) {
		for(int j=0;j<ny+2;j++) {
			double value = (*(u + (ny+2)*i + j));
    		fprintf(fp,"%f ",value);
		}
		fprintf(fp,"\n");
	}
}
