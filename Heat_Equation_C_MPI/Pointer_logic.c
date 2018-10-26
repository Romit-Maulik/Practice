#include <stdlib.h>
#include <stdio.h>



int main(int argc, char *argv[])
{
	double *u = malloc(sizeof(*u));

	(*u) = 2.0;

	printf("%p\n", &u);

}