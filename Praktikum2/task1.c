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
int is_arr_sorted( int* arr, int len ) {

	int i;
	for( i = 0; i < len - 1; ++i )
		if( arr[ i ] > arr[ i + 1 ] )
			return 0;
	return 1;
}


/**
 * Checks whether arr is sorted globally. 
 **/
int verify_results( int* arr, int len, int myrank, int nprocs ) {
	int is_sorted_global = 0;
	int failed = 0;
	if( !(is_arr_sorted(arr, len)) || (len > 1))
		failed = 1;
	MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
	if(failed)
		return 0;

	int neighbor_up, neighbor_down, len_next,recv_next, sent_to_prev;
	MPI_Status status;
	// send length
	if(myrank + 1 > nprocs - 1) 
		neighbor_up = MPI_PROC_NULL;
	else{
		neighbor_up = myrank + 1;
	}
	if(myrank - 1 < 0)
		neighbor_down = MPI_PROC_NULL;
	else
		neighbor_down = myrank -1;
	// check if next proc recieved an element
	MPI_Sendrecv(&len, 1, MPI_INT, neighbor_up, 0, &len_next, 1, MPI_INT, neighbor_down, 0, MPI_COMM_WORLD, &status);

	//exchange
	MPI_Sendrecv(&arr[MAX_NUM_LOCAL_ELEMS - 1], 1, MPI_INT, neighbor_up, 0, &recv_next, 1, MPI_INT, neighbor_down, 0, MPI_COMM_WORLD, &status);
	
	if (len_next){
		if(recv_next < sent_to_prev)
			failed = 1;
	}

	if(!failed)
		is_sorted_global = 1;
	return is_sorted_global;
}


/**
 * This function compares two integers.
 */
int comp_func( const void* a, const void* b ) {
	return ( *(int*)a - *(int*)b );
}


/**
 * Returns unique random integer.
 */
int get_unique_rand_elem( unsigned int total_count_bits, unsigned int index, double scale ) {

	int random_elem = (int)( scale * drand48() );
	int unique_random_element = ( random_elem << total_count_bits ) | index;

	return unique_random_element;
}


/**
 * Initilizes the input. Each process will have a random local array of numbers. The
 * length of this array is anywhere between 0 to MAX_NUM_LOCAL_ELEMS
 */
void init_input( int w_myrank, int w_nprocs, int* input_arr, 
					int* input_len, int* total_elems ) {

	// Initialize random seed
	srand48( w_nprocs );

	// Total number of elements is 65% of the number of processes
	*total_elems = (int)( w_nprocs * 0.65 );
	int* global_arr = NULL;
	int* sendcnts = NULL;
	int* displs = NULL;

	if( w_myrank == 0 ) {
		global_arr = malloc( *total_elems * sizeof(int) );

		double scale = *total_elems * 5;
		int total_count_bits = (int)ceil( log( *total_elems ) / log( 2.0 ) );

		// Init global array with random elements
		for( int i = 0; i < *total_elems; ++i )
			global_arr[i] = get_unique_rand_elem( total_count_bits, i, scale );

		// Randomly decide how much elements each rank will get
		sendcnts = malloc( w_nprocs * sizeof(int) );
		memset( sendcnts, 0, w_nprocs * sizeof(int) );
		int total_elem_cnt = *total_elems;
		for( int i = 0; i < w_nprocs; ++i ) {
			double coin_flip = drand48();
			if( coin_flip < 0.45 ) {
				sendcnts[i]++;
				total_elem_cnt--;
				if( total_elem_cnt == 0 ) break;
				coin_flip = drand48();
				if( coin_flip < 0.35 ) {
					sendcnts[i]++;
					total_elem_cnt--;
					if( total_elem_cnt == 0 ) break;
					if( coin_flip < 0.15 ) {
						sendcnts[i]++;
						total_elem_cnt--;
						if( total_elem_cnt == 0 ) break;
					}
				}
			}
		}

		// Redistribute remaining counts
		int curr_rank = 0;
		while( total_elem_cnt > 0 ) {
			while( sendcnts[curr_rank] >= MAX_NUM_LOCAL_ELEMS )
				++curr_rank;
			sendcnts[curr_rank]++;
			total_elem_cnt--;
		}

		displs = malloc( w_nprocs * sizeof(int) );
		displs[0] = 0;
		for( int i = 1; i < w_nprocs; ++i )
			displs[i] = displs[i - 1] + sendcnts[i - 1];
	}

	// Redistribute the input length    
	MPI_Scatter( sendcnts, 1, MPI_INT, input_len, 1, MPI_INT, 0, MPI_COMM_WORLD );

	// Redistribute the input
	MPI_Scatterv( global_arr, sendcnts, displs, MPI_INT, input_arr, *input_len, 
					MPI_INT, 0, MPI_COMM_WORLD );

	free( global_arr );
	free( sendcnts );
	free( displs );
}

int main( int argc, char** argv ) {

	int w_myrank, w_nprocs;
	MPI_Init( &argc, &argv );
	MPI_Comm_rank( MPI_COMM_WORLD, &w_myrank );
	MPI_Comm_size( MPI_COMM_WORLD, &w_nprocs );

	init_clock_time();

	//
	// Initialization phase
	//

	int n = 0;
	int total_n;
	int elem_arr[MAX_NUM_LOCAL_ELEMS];

	init_input( w_myrank, w_nprocs, elem_arr, &n, &total_n );

	// split rows
	int color_rows = w_myrank / sqrt(w_nprocs);
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

	//visualize split
	//  printf("w_myrank: %d, w_nprocs %d, r_myrank: %d, r_nprocs %d, c_myrank: %d, c_nprocs: %d, my n: %d\n", w_myrank, w_nprocs, r_myrank, r_nprocs, c_myrank, c_nprocs, n);


	double start = get_clock_time();

	// see elements in elem_Arr
	// printf("w_myrank: %d, elem_arr[", w_myrank);
	// for(int i = 0; i < sizeof(elem_arr) / sizeof (int); i++){
	// 	printf("%d, ", elem_arr[i]);
	// }
	// printf("]\n");

	//dislacement arrays for allgatherv
	int* row_displs = malloc(sizeof(int) * r_nprocs);
	int* col_displs = malloc(sizeof(int) * r_nprocs);

	//number of items sent by each process
	int* rows_send_counts = malloc(sizeof(int) * r_nprocs);
	int* cols_send_counts = malloc(sizeof(int) * c_nprocs);
	
	//gather number of elements sent by each process
	MPI_Allgather(&n, 1, MPI_INT, rows_send_counts, 1, MPI_INT, row_comm);
	MPI_Allgather(&n, 1, MPI_INT, cols_send_counts, 1, MPI_INT, col_comm);

	// print cnt arrays
	// printf("w_myrank: %d, sendcntrows[", w_myrank);
	// for(int i = 0; i < r_nprocs; i++){
	// 	printf("%d, ", sendcntsrows[i]);
	// }
	// printf("]\n");
	// printf("w_myrank: %d, sendcntcols[", w_myrank);
	// for(int i = 0; i < c_nprocs; i++){
	// 	printf("%d, ", sendcntscols[i]);
	// }
	// printf("]\n");


	// fill displacement arrays: add amount of items sent by each process to the exisiting amount of elements; first process can put items at the beginning thus displacement = 0
	row_displs[0] = 0;
	col_displs[0] = 0;
	for( int i = 1; i < r_nprocs; ++i ){
		row_displs[i] = row_displs[i - 1] + rows_send_counts[i - 1];
		col_displs[i] = col_displs[i - 1] + cols_send_counts[i - 1];
	}


	int row_arr_size = 0;
	int col_arr_size = 0;

	// create arrays for elements in row and column
	for(int i = 0; i < r_nprocs; i ++){
		row_arr_size += rows_send_counts[i];
		col_arr_size += cols_send_counts[i];
	}
	int* row_arr = malloc(sizeof(int) * row_arr_size);
	int* col_arr = malloc(sizeof(int) * col_arr_size);

	// gather items across row and column
	MPI_Allgatherv(elem_arr, n, MPI_INT, row_arr, rows_send_counts, row_displs, MPI_INT, row_comm);
	MPI_Allgatherv(elem_arr, n, MPI_INT, col_arr, cols_send_counts, col_displs, MPI_INT, col_comm);

	// sort items
	qsort(row_arr, row_arr_size, sizeof(int), comp_func);
	qsort(col_arr, col_arr_size, sizeof(int), comp_func);

	//print row and col arrays
	// printf("w_myrank: %d, row_arr[", w_myrank);
	// for(int i = 0; i < row_arr_size; i++){
	// 	printf("%d, ", row_arr[i]);
	// }
	// printf("]\n");

	// printf("w_myrank: %d, col_arr[", w_myrank);
	// for(int i = 0; i < col_arr_size; i++){
	// 	printf("%d, ", col_arr[i]);
	// }
	// printf("]\n");

	// create arrays holding ranks. size of array is the highest number given. each index will hold the rank of the item of that value
	int local_ranks_size = 0;
	if(col_arr_size > 0)
		local_ranks_size = col_arr[col_arr_size - 1];
	MPI_Allreduce(MPI_IN_PLACE, &local_ranks_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	int* local_ranks = malloc(sizeof(int) * local_ranks_size);
	int* arr_global_ranks = malloc(sizeof(int) * local_ranks_size);

	//init both arrays holding local and global ranks with 0;
	for (int i = 0; i < local_ranks_size; i++){
		local_ranks[i] = 0;
		arr_global_ranks[i] = 0;
	}
	
	// calculate ranks of elements. if rank of element is 0 set it as 100000 as to not be confused with an empty space in the array.
	if(col_arr_size > 0){
		int count;
		for(int i = 0; i < col_arr_size; i++){
			while( (col_arr[i] > row_arr[count]) && (count < row_arr_size) ){
				//if (row_arr_size > 0)
					count++;
			}
			if(count == 0)
				local_ranks[col_arr[i] - 1] = 100000;
			else
				local_ranks[col_arr[i] - 1] = count;
		}
	}

	// printf("w_myrank: %d, local_rank[", w_myrank);
	// for(int i = 0; i < local_ranks_size; i++){
	// 	printf("%d, ", local_ranks[i]);
	// }
	// printf("]\n");	

	// make local ranks
	MPI_Allreduce(local_ranks, arr_global_ranks, local_ranks_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	// // // print global array
	// if(w_myrank == 0){
	// 	printf("w_myrank: %d, global_ranks[", w_myrank);
	// 	for(int i = 0; i < local_ranks_size; i++){
	// 		printf("%d, ", arr_global_ranks[i]);
	// 	}
	// 	printf("]\n");	
	// }
	//
	// Redistribute data
	// Adjust this code to your needs
	//
	MPI_Request req_arr[MAX_NUM_LOCAL_ELEMS];
	MPI_Status stat_arr[MAX_NUM_LOCAL_ELEMS];
	int n_req = 0;
	int n_stat = 0;

	// for (int i = 0; i < local_ranks_size; i++){
	// 	arr_global_ranks[i] = arr_global_ranks[i] % 100000000;
	// }


	int recv;
	int i;
	// distribute items make a send request for every item 
	do{
		for(int j = 0; j < local_ranks_size; j++){
			if ((j + 1)  == elem_arr[i]) {
				// printf("%d: w_myrank: %d sending %d to %d\n", j, w_myrank, elem_arr[i], arr_global_ranks[j] % 100000);
				MPI_Isend( &(elem_arr[i]), 1, MPI_INT, arr_global_ranks[j] % 100000, 0, MPI_COMM_WORLD, req_arr + n_req );
				elem_arr[i] = 0;
				n_req++;
			}
		}
		i++;
	}while(i < n);

	// get item assigned to process
	for(int j = 0; j < local_ranks_size; j++){
		if( (w_myrank != 0) && (w_myrank == arr_global_ranks[j] % 100000)){
			// printf("myrank: %d, making a recv req for %d\n", w_myrank, j + 1);
			MPI_Recv(&recv + n_stat, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, stat_arr + n_stat);
			elem_arr[MAX_NUM_LOCAL_ELEMS - 1 - n_stat] = recv;
			n_stat++;
		}
		if((w_myrank == 0) && (arr_global_ranks[j] != 0) && (arr_global_ranks[j] % 100000 == 0)){
			// printf("myrank: %d, making a reccv req for %d\n", w_myrank, j + 1);
			MPI_Recv(&recv + n_stat, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, stat_arr + n_stat);
			elem_arr[MAX_NUM_LOCAL_ELEMS - 1 - n_stat] = recv;
			n_stat++;
		}
	}
	n = n_stat;
	MPI_Waitall( n_req, req_arr, stat_arr );
	// printf("w_myrank: %d, elem_arr: %d\n", w_myrank, elem_arr[MAX_NUM_LOCAL_ELEMS - 1]);

	//
	// Measure the execution time after all the steps are finished, 
	// but before verifying the results
	//
	double elapsed = get_clock_time() - start;

	//
	// Verify the data is sorted globally
	//
	int res = verify_results( elem_arr, n, w_myrank, w_nprocs );
	if( w_myrank == 0 ) {
		if( res ) {
			printf( "Results correct!\n" );
		}
		else {
			printf( "Results incorrect!\n" );
		}
	}

	// Get timing - max across all ranks
	double elapsed_global;
	MPI_Reduce( &elapsed, &elapsed_global, 1, MPI_DOUBLE, 
				MPI_MAX, 0, MPI_COMM_WORLD );

	if( w_myrank == 0 ) {
		printf( "Elapsed time (ms): %f\n", elapsed_global );
	}

	MPI_Finalize();

	return 0;
}