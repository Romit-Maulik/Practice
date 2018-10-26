#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

/*
Global variables accessible to all processing elements
*/
const int nx_global = 256;
const int ny_global = 256;
const double lx = 2.0*3.141519;
const double ly = 2.0*3.141519;
const double stab_param = 0.8;
const double alpha = 0.8;

/*
Message direction tags
*/
const int send_up_tag = 0;
const int send_down_tag = 1;
const int send_left_tag = 2;
const int send_right_tag = 3;

/*
Function declarations here
*/
void initialize_array(double *u, int nx, int ny, int my_id, int x_id, int y_id, int proc_dim);
void copy_array(double *u, double *u_copy, int nx, int ny);
void send_messages(double *u, int nx, int ny, int my_id, int x_id, int y_id, int proc_dim);
void receive_messages(double *u, int nx, int ny, int my_id, int x_id, int y_id, int proc_dim);
void update_solution(double *u, double *u_temp, int nx, int ny, double const_mult);
void write_array(double *u, int nx, int ny, int my_id, int x_id, int y_id);

int main(int argc, char *argv[])
{

	/*
	MPI related fluff
	*/
	int ierr, num_procs, my_id;
	ierr = MPI_Init(&argc, &argv);

	/* find out MY process ID, and how many processes were started. 
	   Note that num_procs should be divisible by 4
	*/
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
	ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	/*
	Locate process in physical space
	*/
	int proc_dim = sqrt(num_procs);
	int x_id = my_id%proc_dim;
	int y_id = my_id/proc_dim;

	const int nx = nx_global/proc_dim;
	const int ny = ny_global/proc_dim;
	
	/*
	Find mesh sizes
	*/
	const double dx = lx/((float) nx_global);
	const double dy = ly/((float) ny_global);

	/*
	Initializing local storage - 0/0 and nx+1/ny+1 are the `ghost' cells
	*/
	double *u = malloc(sizeof(*u) * (nx+2) * (ny+2)); //Pointer to first element of u array
	double *u_temp = malloc(sizeof(*u) * (nx+2) * (ny+2)); //Pointer to first element of u_temp array

	/*
	We will time only master node
	*/
	clock_t start, end;
	double cpu_time_used;

	if (my_id==0)
	{
		start = clock();
	}


	/* 
	Initial conditions on all processors and copying to temporary array
	*/
	initialize_array(u,nx,ny,my_id,x_id,y_id,proc_dim);//The argument definition is the type but the argument is the address to an array

	send_messages(u,nx,ny,my_id,x_id,y_id,proc_dim); //Send boundary information among processes
	receive_messages(u,nx,ny,my_id,x_id,y_id,proc_dim); //Receive boundary information among processes

	MPI_Barrier(MPI_COMM_WORLD);

	copy_array(u,u_temp,nx,ny);


	/*
	Time integration variables
	*/
	double t, dt, const_mult;
	const double ft = 1.0;

	dt = stab_param*dx*dx/(4.0*alpha);
	const_mult = alpha*dt/(dx*dx);

	//FTCS integration
	t = 0.0;
	do{
		
		update_solution(u,u_temp,nx,ny,const_mult);
		send_messages(u,nx,ny,my_id,x_id,y_id,proc_dim); //Send boundary information among processes
		receive_messages(u,nx,ny,my_id,x_id,y_id,proc_dim); //Receive boundary information among processes

		MPI_Barrier(MPI_COMM_WORLD);

		copy_array(u,u_temp,nx,ny);

		t = t + dt;
	}while(t<ft);

	if (my_id==0)
	{
		end = clock();
		cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("CPU time used = %f\n", cpu_time_used);
	}


	/*
	Output of data to separate files
	*/
	write_array(u,nx,ny,my_id,x_id,y_id);

	/*
	Free local memory
	*/
	free(u);
	free(u_temp);
	

	ierr = MPI_Finalize();

	return 0;
}

/*
Initializing array
Note that in C functions the argument definitions should be the types
The argument itself is the address to an array
*/
void initialize_array(double *u, int nx, int ny, int my_id, int x_id, int y_id, int proc_dim)
{
	double lx_local = lx/proc_dim;
	double ly_local = ly/proc_dim;

	double x_mult = (double) x_id/proc_dim;
	double y_mult = (double) y_id/proc_dim;

	double x,y;
	
	for (int i = 1; i < nx+1; i++)
	{
		for (int j = 1; j < ny+1; j++)
		{
			x = x_mult*lx + ((double) (i-1)/nx)*lx_local;
			y = y_mult*ly + ((double) (j-1)/ny)*ly_local;

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

/*
MPI message passing routines
*/
void send_messages(double *u, int nx, int ny, int my_id, int x_id, int y_id, int proc_dim)
{
	/*
	Note that message passing commands take the address of the first element of the array and a length
	of contiguous memory to send
	*/

	/*x-direction message passing*/
	if (x_id !=0)
	{
		/*Left boundary send left*/
		MPI_Send((u + (ny+2) + 1), ny, MPI_DOUBLE, my_id-1, send_left_tag, MPI_COMM_WORLD); //Correct
	}
	
	if (x_id !=proc_dim-1)
	{
		/*Right boundary send right*/
		MPI_Send((u + (ny+2)*(nx) + 1), ny, MPI_DOUBLE, my_id+1, send_right_tag, MPI_COMM_WORLD); //Correct
	}

	/*y-direction message passing*/
	if (y_id !=proc_dim-1)
	{
		double *temp = malloc(sizeof(*temp) * nx);
		/*Dummy array for placeholder*/
		for (int i = 1; i < nx+1; i++)
		{
			(*(temp+i-1)) = (*(u + (ny+2)*(i+1) - 2));
		}

		/*Top boundary send up*/
		MPI_Send(temp, nx, MPI_DOUBLE, my_id+proc_dim, send_up_tag, MPI_COMM_WORLD); //Correct

		free(temp);
	}
	
	if (y_id != 0)
	{
		double *temp = malloc(sizeof(*temp) * nx);
		/*Dummy array for placeholder*/
		for (int i = 1; i < nx+1; i++)
		{
			(*(temp+i-1)) = (*(u + (ny+2)*(i) + 1)); //Correct
		}

		/*Bottom boundary send down*/
		MPI_Send(temp, nx, MPI_DOUBLE, my_id-proc_dim, send_down_tag, MPI_COMM_WORLD);

		free(temp);
	}

	/*
	Periodic exchange of information - x
	*/
	if (x_id==0)
	{
		/*Left boundary send to right domain*/
		MPI_Send((u + (ny+2) + 1), ny, MPI_DOUBLE, my_id+proc_dim-1, send_left_tag, MPI_COMM_WORLD); //Correct
	}
	
	if (x_id==proc_dim-1)
	{
		/*Right boundary send right*/
		MPI_Send((u + (ny+2)*(nx) + 1), ny, MPI_DOUBLE, my_id-proc_dim+1, send_right_tag, MPI_COMM_WORLD); //Correct
	}

	/*
	Periodic exchange of information - y
	*/
	if (y_id == proc_dim-1)
	{
		double *temp = malloc(sizeof(*temp) * nx);
		/*Dummy array for placeholder*/
		for (int i = 1; i < nx+1; i++)
		{
			(*(temp+i-1)) = (*(u + (ny+2)*(i+1) - 2)); //Correct
		}

		int target_id = my_id - (proc_dim-1)*proc_dim;

		/*Top boundary send up*/
		MPI_Send(temp, nx, MPI_DOUBLE, target_id, send_up_tag, MPI_COMM_WORLD);

		free(temp);
	}
	
	if (y_id == 0)
	{
		double *temp = malloc(sizeof(*temp) * nx);
		/*Dummy array for placeholder*/
		for (int i = 1; i < nx+1; i++)
		{
			(*(temp+i-1)) = (*(u + (ny+2)*(i) + 1));
		}

		int target_id = my_id + (proc_dim-1)*proc_dim;

		/*Bottom boundary send down*/
		MPI_Send(temp, nx, MPI_DOUBLE, target_id, send_down_tag, MPI_COMM_WORLD);

		free(temp);
	}

}

void receive_messages(double *u, int nx, int ny, int my_id, int x_id, int y_id, int proc_dim)
{

	MPI_Status status;              // status returned by MPI calls

	/*x-direction message receiving*/
	if (x_id !=0)
	{
		/*Left boundary receives from left*/
		MPI_Recv((u+1), ny, MPI_DOUBLE, my_id-1, send_right_tag, MPI_COMM_WORLD, &status); //correct

	}
	
	if (x_id !=proc_dim-1)
	{
		/*Right boundary receives from right*/
		MPI_Recv((u+(ny+2)*(nx+1)+1), ny, MPI_DOUBLE, my_id+1, send_left_tag, MPI_COMM_WORLD, &status); //correct

	}

	/*y-direction message passing*/
	if (y_id !=proc_dim-1)
	{
		double *temp = malloc(sizeof(*temp) * nx);
		/*Top boundary receives from top*/
		MPI_Recv(temp, nx, MPI_DOUBLE, my_id+proc_dim, send_down_tag, MPI_COMM_WORLD, &status);

		for (int i = 1; i < nx+1; i++)
		{
			(*(u + (ny+2)*(i+1) - 1)) = (*(temp+i-1));
		}

		free(temp);
	}
	
	if (y_id != 0)
	{
		double *temp = malloc(sizeof(*temp) * nx);
		/*Bottom boundary receives from bottom*/
		MPI_Recv(temp, nx, MPI_DOUBLE, my_id-proc_dim, send_up_tag, MPI_COMM_WORLD, &status);

		for (int i = 1; i < nx+1; i++)
		{
			(*(u + (ny+2)*(i))) = (*(temp+i-1));
		}

		free(temp);
	}

	/*
	Periodic exchange of information - x
	*/
	if (x_id==0)
	{
		/*Left receives from extreme right domain*/
		MPI_Recv((u+1), ny, MPI_DOUBLE, my_id+proc_dim-1, send_right_tag, MPI_COMM_WORLD, &status); //correct

	}
	
	if (x_id==proc_dim-1)
	{
		/*Right boundary receives from extreme left domain*/
		MPI_Recv((u+(ny+2)*(nx+1)+1), ny, MPI_DOUBLE, my_id-proc_dim+1, send_left_tag, MPI_COMM_WORLD, &status); //correct
	}

	/*
	Periodic exchange of information - y
	*/
	if (y_id ==proc_dim-1)
	{
		double *temp = malloc(sizeof(*temp) * ny);
		/*Dummy array for placeholder*/

		/*
		Top boundary receives from extreme bottom domain
		*/
		int target_id = my_id - proc_dim*(proc_dim-1);
		MPI_Recv(temp, nx, MPI_DOUBLE, target_id, send_down_tag, MPI_COMM_WORLD, &status);

		for (int i = 1; i < nx+1; i++)
		{
			(*(u + (ny+2)*(i+1) - 1)) = (*(temp+i-1));
		}

		free(temp);
	}
	
	if (y_id == 0)
	{
		double *temp = malloc(sizeof(*temp) * nx);
		/*Dummy array for placeholder*/
		
		/*
		Bottom boundary receives from extreme top domain
		*/
		int target_id = my_id + proc_dim*(proc_dim-1);
		MPI_Recv(temp, nx, MPI_DOUBLE, target_id, send_up_tag, MPI_COMM_WORLD, &status);

		for (int i = 1; i < nx+1; i++)
		{
			(*(u + (ny+2)*(i))) = (*(temp+i-1));
		}

		free(temp);
	}

}

/*
Finite difference bit
*/

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
}

/*
Parallel IO
*/
void write_array(double *u, int nx, int ny, int my_id, int x_id, int y_id)
{
	FILE *fp;
	char filename[50];
	sprintf(filename,"%i",my_id);

	fp = fopen(filename,"wb");

	for(int i=1;i<nx+1;i++) {
		for(int j=1;j<ny+1;j++) {
			double value = (*(u + (ny+2)*i + j));
    		fprintf(fp,"%f ",value);
		}
		fprintf(fp,"\n");
	}
}


/*
Some tricks we can use for validation of message passing

	// if (my_id == 0)
	// {
	// 	printf("Bottom ghost of PE %i: %f\n", my_id, *(u+(ny+2)));//Bottom ghost
	// 	printf("Bottom true of PE %i: %f\n",my_id, *(u+(ny+2)+1));//Bottom true
	// 	// printf("Top ghost of PE %i: %f\n",my_id, *(u+2*(ny+2)-1));//Top ghost
	// 	// printf("Top true of PE %i: %f\n", my_id, *(u+2*(ny+2)-2));//Top true

	// 	// printf("Left ghost of PE %i: %f\n", my_id, (*(u+1)) );
	// 	// printf("Left true of PE %i: %f\n", my_id, (*(u+(ny+2)+1)) );

	// 	// printf("Right ghost of PE %i: %f\n", my_id, (*(u+(ny+2)*(nx+1)+1)) );
	// 	// printf("Right true of PE %i: %f\n", my_id, (*(u+(ny+2)*(nx)+1)) );

	// }

	// if (my_id == 12)
	// {
	// 	// printf("Bottom ghost of PE %i: %f\n", my_id, *(u+(ny+2)));//Bottom ghost
	// 	// printf("Bottom true of PE %i: %f\n",my_id, *(u+(ny+2)+1));//Bottom true
	// 	printf("Top ghost of PE %i: %f\n",my_id, *(u+2*(ny+2)-1));//Top ghost
	// 	printf("Top true of PE %i: %f\n", my_id, *(u+2*(ny+2)-2));//Top true

	// 	// printf("Left ghost of PE %i: %f\n", my_id, (*(u+1)) );
	// 	// printf("Left true of PE %i: %f\n", my_id, (*(u+(ny+2)+1)) );

	// 	// printf("Right ghost of PE %i: %f\n", my_id, (*(u+(ny+2)*(nx+1)+1)) );
	// 	// printf("Right true of PE %i: %f\n", my_id, (*(u+(ny+2)*(nx)+1)) );
	// }

	// exit(0);

*/
