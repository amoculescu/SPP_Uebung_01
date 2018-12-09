#include <stdio.h>
#include <omp.h>

int main ( int argc, char *argv[] ) {
    printf("Hello parallel world!.\n");
    printf("Currently up to %d threads can be used.\n", omp_get_max_threads());
    omp_set_num_threads(3); //Change number of used-threads here!
    printf("But we will only use %d threads\n", omp_get_max_threads());
#pragma omp parallel
    {
        printf("Hello, Thread-ID %d\n", omp_get_thread_num());
    }
}
