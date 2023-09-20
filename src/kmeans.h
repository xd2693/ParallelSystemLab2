#ifndef _KMEANS_H
#define _KMEANS_H

#include <argparse.h>
#include <io.h>
#include <helper.h>
#include <math.h>
#include <float.h>
#include <cstring>

double get_distance(kmeans_args* args, 
                    int          input_index,
                    int          center_index);

int get_label (kmeans_args* args, int index);

void get_new_centers(kmeans_args* args, double* new_centers);

bool test_converge(kmeans_args* args, double* new_centers);

void kmeans_cpu(kmeans_args* args);
#endif