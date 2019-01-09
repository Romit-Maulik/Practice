#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

void tridiag_solve(double *a, double *b, double *c, double *d, double *x, int ne);
void cyclic_tridiag_solve(double *a, double *b, double *c, double *d, double *x, int ne, double a1, double b1);


int main (int argc, char** argv)
{

	double *a = malloc(sizeof(*a) * 5); 
	double *b = malloc(sizeof(*b) * 5);
	double *c = malloc(sizeof(*c) * 5);
	double *d = malloc(sizeof(*d) * 5);
	double *x = malloc(sizeof(*x) * 5);

	for (int i = 0; i < 5; i++)
	{
		(*(b+i)) = 4.0;
		(*(c+i)) = 3.0;
		(*(a+i)) = 3.0;
		(*(d+i)) = 1.0;
	}

	(*(a)) = 0.0;
	(*(c+4)) = 0.0;

	cyclic_tridiag_solve(a,b,c,d,x,5,3,3);

	for (int i = 0; i < 5; i++)
	{
		printf("%f \n", (*(x+i)));
	}
	
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
