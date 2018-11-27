# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>


int main(int argc, char *argv[]);
double test01(int n, double x[], double y[]);
double test02(int n, double x[], double y[]);

/******************************************************************************/

int main(int argc, char *argv[])
{
	double factor;
	int i;
	int n;
	double wtime;
	double *x;
	double xdoty;
	double *y;
	int thrednum = 1;

	n = 10000000;

		x = (double *)malloc(n * sizeof(double));
		y = (double *)malloc(n * sizeof(double));

		factor = (double)(n);
		factor = 1.0 / sqrt(2.0 * factor * factor + 3 * factor + 1.0);

		for (i = 0; i < n; i++)
		{
			x[i] = (i + 1) * factor;
		}

		for (i = 0; i < n; i++)
		{
			y[i] = (i + 1) * 6 * factor;
		}
	
		printf("Sequential/Parallel | threads | Length | Result | time\n");
	
		/*
		  Test #1
		*/
		//...YOU NEED TO FILL HERE ...

		// starting time, terminating time
		double starttime, stoptime;

		starttime = omp_get_wtime();

		xdoty = test01(n, x, y);

		stoptime = omp_get_wtime();
		wtime = stoptime - starttime;

		printf("  Sequential %7d  %8d  %14.6e  %15.10f\n", thrednum, n, xdoty, wtime);
	
		
		while (thrednum <= 16)
	{
		
		/*
		  Test #2
		*/
		//...YOU NEED TO FILL HERE ...

		starttime = omp_get_wtime();
		xdoty = test02(n, x, y, thrednum);

		stoptime = omp_get_wtime();
		wtime = stoptime - starttime;

		printf("  Parallel  %8d  %8d  %14.6e  %15.10f\n",thrednum, n, xdoty, wtime);

		thrednum = thrednum * 2;
	}

	/*
	Test #1
	*/
	//...YOU NEED TO FILL HERE ...

	// starting time, terminating time

	starttime = omp_get_wtime();

	xdoty = test01(n, x, y);

	stoptime = omp_get_wtime();
	wtime = stoptime - starttime;

	printf("  Sequential %7d  %8d  %14.6e  %15.10f\n", thrednum, n, xdoty, wtime);

	free(x);
	free(y);

	/*
	  Terminate.
	*/

	printf("\n");
	printf("DOT_PRODUCT\n");
	printf("  Normal end of execution.\n");
	char c;
	c = getchar();
	return 0;
}

//Sequential version
double test01(int n, double x[], double y[])

{
	int i;
	double xdoty;

	xdoty = 0.0;

	//...YOU NEED TO FILL HERE...

	for (i = 0; i < n; i++)
	{
		xdoty = xdoty + x[i] * y[i];
	}

	return xdoty;
}

//Parallel version
double test02(int n, double x[], double y[], int thrednum)

{
	int i;
	double xdoty;

	xdoty = 0.0;

	//...YOU NEED TO FILL

  //parallelize the products
#pragma omp parallel for reduction(+:xdoty) numthreads(thrednum)
	for (i = 0; i < n; i++)
	{
		xdoty = xdoty + x[i] * y[i];
	}

	return xdoty;
}