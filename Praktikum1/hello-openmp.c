#include <stdio.h>
#include <omp.h>

int main ( int argc, char *argv[] ) {
	// Current output may differ from the one asked -> CHECK AGAIN! 
    printf("Hello parallel world!.\n");
    printf("Currently there can be up to %d threads being used.\n", omp_get_max_threads());
    omp_set_num_threads(3); //Change number of used-threads here!
    printf("But we will only use %d\n", omp_get_max_threads());
#pragma omp parallel
    {
        printf("Hello, thread %d\n", omp_get_thread_num());
    }
}
