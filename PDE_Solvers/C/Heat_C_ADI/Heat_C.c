/*
Pointer arithmetic to solve heat equation in periodic domain
Compile using gcc heat_c.c -lm
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
void ftcs_update_solution(double *u,double *u_temp, int nx, int ny, double const_mult);
void write_array(double *u, int nx, int ny);
void probe_location(double *u, int i, int j, int ny);
void tridiag_solve(double *a, double *b, double *c, double *d, double *x, int ne);
void cyclic_tridiag_solve(double *a, double *b, double *c, double *d, double *x, int ne, double a1, double b1);
void adi_update_solution(double *u, double *u_temp, int nx, int ny, double dt, double dx, double dy);

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

	dt = 5.0*STAB_PARAM*dx*dx/(4.0*ALPHA);
	const_mult = ALPHA*dt/(dx*dx);

	//FTCS integration
	t = 0.0;
	do{
		// ftcs_update_solution(u,u_temp,nx,ny,const_mult);
		// printf("time = %f\n",t);
		adi_update_solution(u,u_temp,nx,ny,dt,dx,dy);
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

void ftcs_update_solution(double *u, double *u_temp, int nx, int ny, double const_mult)
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

void adi_update_solution(double *u, double *u_temp, int nx, int ny, double dt, double dx, double dy)
{
	double c1 = -ALPHA*dt/(2.0*dx*dx);
	double c2 = 1.0+ALPHA*dt/(dx*dx);//Diagonals of intermediate matrix
	double c3 = -ALPHA*dt/(2.0*dx*dx);
	double pre_mult = ALPHA*dt/(2.0*dx*dx);


	double *a = malloc(sizeof(*a) * (ny));
	double *b = malloc(sizeof(*b) * (ny));
	double *c = malloc(sizeof(*c) * (ny));
	double *d = malloc(sizeof(*d) * (ny));
	double *x = malloc(sizeof(*x) * (ny));

	// These are fixed - no need to reallocate
	int iter = 0;
	for (int i = 1; i < nx+1; i++)
	{
		(*(b+iter)) = c2;
		(*(a+iter)) = c1;
		(*(c+iter)) = c1;
		iter++;
	}
	(*(a)) = 0.0;
	(*(c+iter)) = 0.0;

	// X direction
	for (int j = 1; j < ny + 1; j++)
	{
		int iter = 0;
		for (int i = 1; i < nx+1; i++)
		{
			double jm1 = (*(u + (ny+2)*i + j - 1));
			double jp1 = (*(u + (ny+2)*i + j + 1));
			double jp = (*(u + (ny+2)*i + j));

			(*(d+iter)) = pre_mult*(jp1+jm1-2.0*jp) + jp;
			iter++;
		}

		cyclic_tridiag_solve(a,b,c,d,x,ny,c1,c3);

		iter = 0;
		for (int i = 1; i < nx+1; i++)
		{
			(*(u_temp+(ny+2)*i + j)) = (*(x+iter));
			iter++;
		}

	}

	// Done with x direction - now for y direction
	// First copy arrays and stuff 
	update_periodic_boundaries(u_temp,nx,ny);

	// for (int i = 0; i <= nx; i++)
	// {
	// 	printf("%i : %f\n",i,(*(u_temp+(ny+2)+i)));
	// }

	// return;

	for (int i = 1; i < nx + 1; i++)
	{
		int iter = 0;
		for (int j = 1; j < ny+1; j++)
		{
			double im1 = (*(u_temp + (ny+2)*(i-1) + j));
			double ip1 = (*(u_temp + (ny+2)*(i+1) + j));
			double ip = (*(u_temp + (ny+2)*i + j));

			(*(d+iter)) = pre_mult*(ip1+im1-2.0*ip) + ip;
			iter++;
		}

		cyclic_tridiag_solve(a,b,c,d,x,nx,c1,c3);

		iter = 0;
		for (int j = 1; j < ny+1; j++)
		{
			(*(u+(ny+2)*i + j)) = (*(x + iter));
			iter++;
		}
	}

	update_periodic_boundaries(u,nx,ny);

	free(a);
	free(b);
	free(c);
	free(d);
	free(x);

}

void cyclic_tridiag_solve(double *a, double *b, double *c, double *d, double *x, int ne, double a1, double b1)
{

	double *bb = malloc(sizeof(*bb) * (ne));
	double gamma = -(*(b));

	(*(bb)) = (*(b))-gamma;
	(*(bb+ne-1)) = (*(b+ne-1))-a1*b1/gamma;

	for (int i = 1;i<ne-1;i++)
	{
		(*(bb+i)) = (*(b+i));
	}
	// Call tridiagonal solver on modified arrays
	tridiag_solve(a,bb,c,d,x,ne);

	double *uu = malloc(sizeof(*uu) * (ne));
	(*(uu)) = gamma;
	(*(uu+ne-1)) = a1;

	for (int i = 1;i<ne-1;i++)
	{
		(*(uu+i)) = 0.0;
	}

	double *zz = malloc(sizeof(*zz) * (ne));

	// Call tridiagonal solver on modified arrays
	tridiag_solve(a,bb,c,uu,zz,ne);	

	// Modify solution
	double fact = ((*(x))+b1*(*(x+ne-1))/gamma)/(1.0+(*(zz))+b1*(*(zz+ne-1))/gamma);

	for (int i=0;i<ne;i++)
	{
		(*(x+i)) = (*(x+i))-fact*(*(zz+i));
	}

	free(zz);
	free(uu);
	free(bb);
}


void tridiag_solve(double *a, double *b, double *c, double *d, double *x, int ne)
{
	// Clean up x
	for (int i = 0; i < ne; i++)
	{
		(*(x+i)) = 0.0;
	}

	/* Leaving incident arrays unmodified */
	double bet = (*(b));
	(*(x)) = (*(d))/bet;


	// Forward
	double *gam = malloc(sizeof(*gam) * (ne)); 
	for (int i = 1; i < ne; i++)
	{
		(*(gam+i)) = (*(c+i-1))/bet;
		bet = (*(b+i))-(*(a+i))*(*(gam+i));
		(*(x+i)) = ((*(d+i))-(*(a+i))*(*(x+i-1)))/bet;
	}

	// Backward
	for (int i = ne-2; i >= 0; i--)
	{
		(*(x+i)) = (*(x+i))-(*(gam+i+1))*(*(x+i+1));
	}

	free(gam);

	/* - Modifying in place
	double *ww = malloc(sizeof(*ww) * (ne));

	for (int i = 1; i < ne; i++)
	{
		(*(ww+i)) = (*(a+i))/(*(b+i-1));
		(*(b+i)) = (*(b+i)) - (*(ww+i))*(*(c+i-1));
		(*(d+i)) = (*(d+i)) - (*(ww+i))*(*(d+i-1));
	}

	(*(x+ne-1)) = (*(d+ne-1))/(*(b+ne-1));

	for (int i = ne-2; i >= 0; i--)
	{
		(*(x+i)) = ((*(d+i))-(*(c+i))*(*(x+i+1)))/((*(b+i)));
	}

	free(ww);
	*/
}
