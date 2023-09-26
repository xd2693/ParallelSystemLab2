#ifndef _KMEANS__SEQ_H
#define _KMEANS_SEQ_H

#include "argparse.h"
#include "io.h"
#include "helper.h"
#include <math.h>
#include <float.h>
#include <cstring>

double get_distance(kmeans_args *args, 
                    int          input_index,
                    int          center_index);

int get_label (kmeans_args *args, int index);

void get_new_centers(kmeans_args *args);

bool test_converge(kmeans_args *args, double *old_centers);

int kmeans_cpu(kmeans_args *args);
#endif