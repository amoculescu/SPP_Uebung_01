#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// This function partitions array A into two subarrays A_1 and A_2
// Input:
//     *l is the index of the first element in array A 
//     *r is the index of the last element in array A
//
//     A              
//     [  |  |  |  |  ...  |  |  |  |  ]
//      *l                           *r
//
// Output:
//     *l is now the index of the first element of array A_2, which still needs to be sorted
//     *r is now the index of the last  element of array A_1, which still needs to be sorted
//
//     A_1              A_2
//     [  |  | ... |  ] [  |  | ... |  ]
//                  *r   *l
void partition(int* A, int* l, int* r)
{
	int pivotPos = *l;
	int pivot = A[*l];
	int mem;

	*l = *l + 1;									//skip pivot 

	while (*l <= *r) {

		if (A[*l] >= pivot && A[*r] < pivot) {
			mem = A[*l];
			A[*l] = A[*r];
			A[*r] = mem;

			*l = *l + 1;
			*r = *r - 1;
			continue;
		}

		if (A[*l] < pivot) *l = *l + 1;
		if (A[*r] >= pivot) *r = *r - 1;

	}
	//printf("%d", omp_get_thread_num());
	mem = A[*r];									//Swap pivot with last element smaller pivot
	A[*r] = A[pivotPos];
	A[pivotPos] = mem;
	*r = *r - 1;

}

// Input:
//     l is the index of the first element in array A 
//     r is the index of the last element in array A
//
//     A              
//     [  |  |  |  |  ...  |  |  |  |  ]
//      l                            r 
void quicksort(int* A, int l, int r)
{
	int oldL = l;
	int oldR = r;

	if (r - l > 0)
		partition(A, &l, &r);

#pragma omp task final(r - oldL < 99) shared(A) firstprivate(r,oldL)
	if (r - oldL > 0)
		quicksort(A, oldL, r);


#pragma omp task final(oldR - l < 99)  shared(A) firstprivate(oldR, l)
	if (oldR - l > 0)
		quicksort(A, l, oldR);

}

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		printf("Usage: %s <array length>\n", argv[0]);
		return 1;
	}
	// Read in number of elements
	int length = strtol(argv[1], NULL, 10);
	srand(14811);

	// Allocate array
	int *A = malloc(sizeof(int) * length);


	// Initialize array
	for (int i = 0; i < length; i++) {
		A[i] = rand() % length;
	}

	// Time the execution
	omp_set_num_threads(16);
	double starttime, stoptime, timespan;
	starttime = omp_get_wtime();

#pragma omp parallel
	{
#pragma omp single
		quicksort(A, 0, length - 1);
#pragma omp barrier
	}


	stoptime = omp_get_wtime();
	timespan = stoptime - starttime;

	// Verify sorted order
	int sorted = 1;
	for (int i = 0; i < length - 1; i++) {
		sorted = sorted && A[i] <= A[i + 1];
	}

	if (sorted)
		printf("The array with lenght %d was sortet in %.16g s \n", length, timespan);
	else
		printf("sorting array faild");

	free(A);
	return 0;
}
