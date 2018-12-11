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

    // TODO

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
 * Merges arr_from into arr.
 */
void merge_arr( int* arr1, int len1, int* arr2, int len2, int* arr_out, int* len_out ) {

    int idx1 = 0, idx2 = 0, idx3 = 0;
    while( idx1 < len1 ) {
        while( idx2 < len2 && 
               arr2[ idx2 ] < arr1[ idx1 ] ) {
            arr_out[ idx3++ ] = arr2[ idx2++ ];
        }
        arr_out[ idx3++ ] = arr1[ idx1++ ];
    }
    while( idx2 < len2 ) {
        arr_out[ idx3++ ] = arr2[ idx2++ ];
    }
    *len_out = idx3;
}


/**
 * All-gather-merge using hypercube approach.
 */
void all_gather_merge( int* arr, int len, int** out_arr, int* out_len, 
                       int nprocs, MPI_Comm comm ) {
    // TODO
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
        printf( "Total number of input elements: %d\n", *total_elems );

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
    MPI_Scatterv( global_arr, sendcnts, displs, MPI_INT, input_arr, *input_len, MPI_INT, 
                  0, MPI_COMM_WORLD );

    free( global_arr );
    free( sendcnts );
    free( displs );
}

int main( int argc, char** argv ) {

    int w_myrank, w_nprocs;
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &w_myrank );
    MPI_Comm_size( MPI_COMM_WORLD, &w_nprocs );

    //
    // Initialization phase
    //
    int n = 0;
    int total_n;
    int elem_arr[MAX_NUM_LOCAL_ELEMS];
    
    init_input( w_myrank, w_nprocs, elem_arr, &n, &total_n );

    
    //
    // TODO
    //


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
 
    MPI_Finalize();

    return 0;
}
