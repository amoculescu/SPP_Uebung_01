// pielst :toint

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>

#include "timing.h"


#define MAX_NUM_LOCAL_ELEMS   3


/**
 * Checks whether arr is sorted locally
 **/
int is_arr_sorted(int* arr, int len) {

	int i;
	for (i = 0; i < len - 1; ++i)
		if (arr[i] > arr[i + 1])
			return 0;
	return 1;
}


/**
* Checks whether arr is sorted globally.
**/
int verify_results(int* arr, int len, int myrank, int nprocs) {
	int is_sorted_global = 0;
	// TODO: From here on!
	int failed = 0;
	if (!(is_arr_sorted(arr, len)) || (len > 1))
		failed = 1;
	MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
	if (failed)
		return 0;

	int neighbor_up, neighbor_down, len_next, recv_next, sent_to_prev;
	MPI_Status status;
	// send length
	if (myrank + 1 > nprocs - 1)
		neighbor_up = MPI_PROC_NULL;
	else {
		neighbor_up = myrank + 1;
	}
	if (myrank - 1 < 0)
		neighbor_down = MPI_PROC_NULL;
	else
		neighbor_down = myrank - 1;
	// check if next proc recieved an element
	MPI_Sendrecv(&len, 1, MPI_INT, myrank, 0, &len_next, 1, MPI_INT, neighbor_up, 0, MPI_COMM_WORLD, &status);
	MPI_Sendrecv(&len, 1, MPI_INT, neighbor_down, 0, &sent_to_prev, 1, MPI_INT, myrank, 0, MPI_COMM_WORLD, &status);

	//exchange
	MPI_Sendrecv(&arr[MAX_NUM_LOCAL_ELEMS - 1], 1, MPI_INT, myrank, 0, &recv_next, 1, MPI_INT, neighbor_up, 0, MPI_COMM_WORLD, &status);
	MPI_Sendrecv(&arr[MAX_NUM_LOCAL_ELEMS - 1], 1, MPI_INT, neighbor_down, 0, &sent_to_prev, 1, MPI_INT, myrank, 0, MPI_COMM_WORLD, &status);

	if (len_next) {
		if (recv_next < sent_to_prev)
			failed = 1;
	}

	if (!failed)
		is_sorted_global = 1;
	return is_sorted_global;
}


/**
 * This function compares two integers.
 */
int comp_func(const void* a, const void* b) {
	return (*(int*)a - *(int*)b);
}


/**
 * Returns unique random integer.
 */
int get_unique_rand_elem(unsigned int total_count_bits, unsigned int index, double scale) {

	int random_elem = (int)(scale * drand48());
	int unique_random_element = (random_elem << total_count_bits) | index;

	return unique_random_element;
}


/**
 * Merges arr_from into arr.
 */
void merge_arr(int* arr1, int len1, int* arr2, int len2, int* arr_out, int* len_out) {

	int idx1 = 0, idx2 = 0, idx3 = 0;
	while (idx1 < len1) {
		while (idx2 < len2 &&
			arr2[idx2] < arr1[idx1]) {
			arr_out[idx3++] = arr2[idx2++];
		}
		arr_out[idx3++] = arr1[idx1++];
	}
	while (idx2 < len2) {
		arr_out[idx3++] = arr2[idx2++];
	}
	*len_out = idx3;
}


/**
 * All-gather-merge using hypercube approach.
 */
void all_gather_merge(int* arr, int len, int** out_arr, int* out_len,
	int nprocs, MPI_Comm comm) {

    //TODO: From here on!

	// get own rank
	int myrank;
	MPI_Comm_rank(comm, &myrank);

	//communication partner parameters
	int kp_rank, kp_rank_resp;
	int kp_array_size, kp_array_resp;
	int* kp_array = NULL;

	//array for merging
	int* merged_array = NULL;;

	MPI_Status status;
	int n_dim = log2(nprocs);
	for (int i = 0; i < n_dim; i++) {
		
		kp_rank = myrank ^ (int)pow(2, i);

		//	MPI_Sendrecv(&len, 1, MPI_INT, myrank, 0, &len_next, 1, MPI_INT, neighbor_up, 0, MPI_COMM_WORLD, &status);
		//	MPI_Sendrecv(&len, 1, MPI_INT, neighbor_down, 0, &sent_to_prev, 1, MPI_INT, myrank, 0, MPI_COMM_WORLD, &status);


		// exchange arrays size
		MPI_Sendrecv(&len, 1, MPI_INT, kp_rank, 0, &kp_array_size, 1, MPI_INT, kp_rank, 0, MPI_COMM_WORLD, &status); //TODO: Comm Missing -> Check whether MPI_COMM_WORLD is ok

		//create output arrays
		kp_array = malloc(sizeof(int) * kp_array_size);
		merged_array = malloc(sizeof(int) * (kp_array_size + len));
		// exchange arrays
		MPI_Sendrecv(arr, len, MPI_INT, kp_rank, 0, kp_array, kp_array_size, MPI_INT, kp_rank, 0, MPI_COMM_WORLD, &status); //TODO: Comm Missing -> Check whether MPI_COMM_WORLD is ok

		// merge array output: merged_array..  len =  new lenght
		merge_arr(arr, len, kp_array, kp_array_size, merged_array, &len);

		//free old arrays
		free(kp_array);

		// set the pointer arr to point at the same adress as merged_array; 
		arr = merged_array;
	}
	printf("myrank: %d: mergedarr:[", myrank);
	for(int i = 0; i < len; i++){
		printf("%d, ", merged_array[i]);
	}
	printf("]\n");
	// arr and merged_array point both point to the finel array
	// len is th length of the meged array

	// setze den pointer auf den **out_arr zeigt auf die adresse von merged_array
	*out_arr = merged_array;
	*out_len = len;

	// TODO
}


/**
 * Initilizes the input. Each process will have a random local array of numbers. The
 * length of this array is anywhere between 0 to MAX_NUM_LOCAL_ELEMS
 */
void init_input(int w_myrank, int w_nprocs, int* input_arr,
	int* input_len, int* total_elems) {

	// Initialize random seed
	srand48(w_nprocs);

	// Total number of elements is 65% of the number of processes
	*total_elems = (int)(w_nprocs * 0.65);
	int* global_arr = NULL;
	int* sendcnts = NULL;
	int* displs = NULL;

	if (w_myrank == 0) {
		printf("Total number of input elements: %d\n", *total_elems);

		global_arr = malloc(*total_elems * sizeof(int));

		double scale = *total_elems * 5;
		int total_count_bits = (int)ceil(log(*total_elems) / log(2.0));

		// Init global array with random elements
		for (int i = 0; i < *total_elems; ++i)
			global_arr[i] = get_unique_rand_elem(total_count_bits, i, scale);

		// Randomly decide how much elements each rank will get
		sendcnts = malloc(w_nprocs * sizeof(int));
		memset(sendcnts, 0, w_nprocs * sizeof(int));
		int total_elem_cnt = *total_elems;
		for (int i = 0; i < w_nprocs; ++i) {
			double coin_flip = drand48();
			if (coin_flip < 0.45) {
				sendcnts[i]++;
				total_elem_cnt--;
				if (total_elem_cnt == 0) break;
				coin_flip = drand48();
				if (coin_flip < 0.35) {
					sendcnts[i]++;
					total_elem_cnt--;
					if (total_elem_cnt == 0) break;
					if (coin_flip < 0.15) {
						sendcnts[i]++;
						total_elem_cnt--;
						if (total_elem_cnt == 0) break;
					}
				}
			}
		}

		// Redistribute remaining counts
		int curr_rank = 0;
		while (total_elem_cnt > 0) {
			while (sendcnts[curr_rank] >= MAX_NUM_LOCAL_ELEMS)
				++curr_rank;
			sendcnts[curr_rank]++;
			total_elem_cnt--;
		}

		displs = malloc(w_nprocs * sizeof(int));
		displs[0] = 0;
		for (int i = 1; i < w_nprocs; ++i)
			displs[i] = displs[i - 1] + sendcnts[i - 1];
	}

	// Redistribute the input length
	MPI_Scatter(sendcnts, 1, MPI_INT, input_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Redistribute the input
	MPI_Scatterv(global_arr, sendcnts, displs, MPI_INT, input_arr, *input_len, MPI_INT,
		0, MPI_COMM_WORLD);

	free(global_arr);
	free(sendcnts);
	free(displs);
}

int main(int argc, char** argv) {
	int w_myrank, w_nprocs;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &w_myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &w_nprocs);

	// split rows
	int color_rows = w_myrank / 2;
	MPI_Comm row_comm;
	MPI_Comm_split(MPI_COMM_WORLD, color_rows, w_myrank, &row_comm);

	// set rank and number of procs in row
	int r_myrank, r_nprocs;
	MPI_Comm_rank( row_comm, &r_myrank );
	MPI_Comm_size( row_comm, &r_nprocs );

	//split columns
	int color_cols = r_myrank ;
	MPI_Comm col_comm;
	MPI_Comm_split(MPI_COMM_WORLD, color_cols, r_myrank, &col_comm);

	//set rank and number of procs in column
	int c_myrank, c_nprocs;
	MPI_Comm_rank( col_comm, &c_myrank );
	MPI_Comm_size( col_comm, &c_nprocs );

	//
	// Initialization phase
	//
	int n = 0;
	int total_n;
	int elem_arr[MAX_NUM_LOCAL_ELEMS];

	init_input(w_myrank, w_nprocs, elem_arr, &n, &total_n);
	double start = get_clock_time();
	printf("start: %d\n", start);

	//visualize split
	printf("w_myrank: %d, w_nprocs %d, r_myrank: %d, r_nprocs %d, c_myrank: %d, c_nprocs: %d, my n: %d\n", w_myrank, w_nprocs, r_myrank, r_nprocs, c_myrank, c_nprocs, n);


	// see elements in elem_Arr
	qsort(elem_arr, n, sizeof(int), comp_func);
	printf("myrank: %d, elem_arr[", w_myrank);
	for(int i = 0; i < n; i++){
		printf("%d, ", elem_arr[i]);
	}
	printf("]\n");

	int* merged_array_rows;
	int* merged_array_cols;
	int merged_array_size_rows, merged_array_size_cols;

	all_gather_merge(elem_arr, n, &merged_array_rows, &merged_array_size_rows, w_nprocs, MPI_COMM_WORLD);
	
	//
	// Measure the execution time after all the steps are finished, 
	// but before verifying the results
	//


	//TODO: Until here!

	//
	// Verify the data is sorted globally
	//
	int res = verify_results(elem_arr, n, w_myrank, w_nprocs);
	if (w_myrank == 0) {
		if (res) {
			printf("Results correct!\n");
		}
		else {
			printf("Results incorrect!\n");
		}
	}

	//TODO: This as well

	// Get timing - max across all ranks
	double elapsed_global;
    double elapsed = get_clock_time() - start;

    MPI_Reduce(&elapsed, &elapsed_global, 1, MPI_DOUBLE,
		MPI_MAX, 0, MPI_COMM_WORLD); //TODO: Elapsed is unkown!

	if (w_myrank == 0) {
		printf("Elapsed time (ms): %f\n", elapsed_global);
	}

	MPI_Finalize();

	return 0;
}