#include <iostream>
#include "argparse.h"
#include "io.h"
#include "helpers.h"

int main(int argc, char **argv){

    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);

    double *input_vals, *clusters;
    int *labels;
    read_file(&opts, &input_vals, &lables, &clusters);

}