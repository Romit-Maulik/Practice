/*
Pointer arithmetic to solve heat equation in periodic domain
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define PI 3.1415926535
#define ALPHA 0.8
#define STAB_PARAM 0.8

/*
Function declarations here
*/
void initialize_array(double *u, int nx, int ny);
void copy_array(double *u, double *u_copy, int nx, int ny);
void update_periodic_boundaries(double *u, int nx, int ny);
void update_solution(double *u,double *u_temp, int nx, int ny, double const_mult);
void write_array(double *u, int nx, int ny);
void probe_location(double *u, int i, int j, int ny);

int main (int argc, char** argv)
{

	/*
	Space related variables
	*/
	const int nx = 254, ny = 254;
	const double lx = 2.0*PI, ly = 2.0*PI;
	double dx, dy;

	dx = lx/nx;
	dy = ly/ny;

	/*
	Allocate memory for our solutions
	*/
	double *u = malloc(sizeof(*u) * (nx+2) * (ny+2)); //Pointer to first element of u array
	double *u_temp = malloc(sizeof(*u) * (nx+2) * (ny+2));; //Pointer to first element of u_temp array

	/*
	Initialize our solution
	*/
	initialize_array(u,nx,ny);
	update_periodic_boundaries(u,nx,ny);
	copy_array(u,u_temp,nx,ny);

	/*
	Setting up time integration specifics
	*/
	double t,dt,const_mult;
	const double ft=1.0;

	clock_t start, end;
	double cpu_time_used;

	start = clock();

	dt = STAB_PARAM*dx*dx/(4.0*ALPHA);
	const_mult = ALPHA*dt/(dx*dx);

	//FTCS integration
	t = 0.0;
	do{
		update_solution(u,u_temp,nx,ny,const_mult);
		//printf("time = %f\n",t);

		t = t + dt;
	}while(t<ft);

	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

	printf("CPU time used = %f\n", cpu_time_used);
	
	write_array(u,nx,ny);

	return 0;
}

void initialize_array(double *u, int nx, int ny)
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

void copy_array(double *u, double *u_copy, int nx, int ny)
{

	for (int i = 0; i < nx+2; i++)
	{
		for (int j = 0; j < ny+2; j++)
		{
			
			(*(u_copy + (ny+2)*i + j)) = (*(u + (ny+2)*i + j));
		}
	}
}


void update_periodic_boundaries(double *u, int nx, int ny)
{
	//Left boundary
	for (int j = 1; j < ny+1; j++)
	{
		(*(u + j)) = (*(u + (ny+2)*nx + j));//Correct
	}

	//Right boundary
	for (int j = 1; j < ny+1; j++)
	{
		(*(u + (ny+2)*(nx+1) + j)) = (*(u + (ny+2) + j));//Correct
	}

	//Bottom boundary
	for (int i = 0; i < nx+2; i++)
	{
		(*(u + (ny+2)*i)) = (*(u + (ny+2)*i + ny));	//Correct
	}

	//Top boundary
	for (int i = 0; i < nx+2; i++)
	{
		(*(u + (ny+2)*i + ny + 1)) = (*(u + (ny+2)*i + 1));	//Correct
	}
}

void update_solution(double *u, double *u_temp, int nx, int ny, double const_mult)
{
	for (int i = 1; i < nx+1; i++)
	{
		for (int j = 1; j < ny+1; j++)
		{
			int p = (ny+2)*i + j; 
			int n = (ny+2)*i + j + 1;
			int s = (ny+2)*i + j - 1;
			int w = (ny+2)*(i-1) + j;
			int e = (ny+2)*(i+1) + j;

			(*(u + p)) = (*(u_temp + p)) + const_mult*(-4.0*(*(u_temp + p)) + (*(u_temp + n)) + (*(u_temp + s)) + (*(u_temp + w)) + (*(u_temp + e)));
		}
	}

	//Update boundary conditions
	update_periodic_boundaries(u,nx,ny);

	//Copy u to utemp
	copy_array(u,u_temp,nx,ny);
}

void write_array(double *u, int nx, int ny)
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

void probe_location(double *u, int i, int j, int ny)
{
	int p = (ny+2)*i + j; 

	printf("%f \n", (*(u+p)));

}