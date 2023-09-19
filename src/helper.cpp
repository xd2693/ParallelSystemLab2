#include "io.h"
#include "argparse.h"

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;
int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}
void kmeans_srand(unsigned int seed) {
    next = seed;
}

void random_centers(int        seed, 
                    int        n_cluster,
                    int        n_vals,
                    int        dims,
                    double**   centers,
                    double**   input_vals) {
    kmeans_srand(seed); // cmd_seed is a cmdline arg
    int in=0;
    for (int i=0; i<n_cluster; i++){
        int index = (kmeans_rand() % n_vals) * dims;
        // you should use the proper implementation of the following
        // code according to your data structure
        for (int j=0, j<dims; j++){
            (*centers)[in] = (*input_vals) [index+j];
            in++;
        }
        
    }
}


