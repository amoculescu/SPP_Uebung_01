#include <stdio.h>
#include <stdlib.h>

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
void partition ( int* A, int* l, int* r )
{

}

// Input:
//     l is the index of the first element in array A 
//     r is the index of the last element in array A
//
//     A              
//     [  |  |  |  |  ...  |  |  |  |  ]
//      l                            r 
void quicksort( int* A, int l, int r )
{

}

int main( int argc, char** argv )
{
    if( argc < 2 )
    {
        printf( "Usage: %s <array length>\n", argv[0] );
        return 1;
    }
    // Read in number of elements



    srand( 14811 );

    // Allocate array
   

    // Initialize array
  
 
    // Time the execution


    // Verify sorted order
    



    return 0;
}
