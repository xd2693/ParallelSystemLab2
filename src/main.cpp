#include <iostream>
#include "argparse.h"
#include "io.h"


int main(int argc, char **argv){

    // Parse args
    struct options_t opts;
    printf("%d args\n",argc);
    get_opts(argc, argv, &opts);
    printf("d=%d, c=%d\n",opts.dims, opts.n_cluster);

    double *input_vals, *centers;
    int *labels;
    int n_vals;
    read_file(&opts, &n_vals, &input_vals, &labels, &centers);

    //printf("%12f ",input_vals[100]);

    
    return 0;

}