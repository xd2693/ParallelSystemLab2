#ifndef _ARGPARSE_H
#define _ARGPARSE_H

#include <getopt.h>
#include <stdlib.h>
#include <iostream>

struct options_t {      
    int n_cluster;
    int dims;
    char *in_file; 
    int max_iter;
    double threshold;
    bool c_flag;
    int seed;
};

void get_opts(int argc, char **argv, struct options_t *opts);
#endif
