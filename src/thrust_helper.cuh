#pragma once
#include <cmath>
#include <cfloat>
#include "argparse.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <iostream>

struct CentoidAssignFunctor {
    double * input_vals;
    double * center_vals;
    int * label_vals;
    int * label_vals_reduce;
    int * own_points;
    int dims;
    int clusters;

    CentoidAssignFunctor(double* input, double* center, int* label, int* label_reduce, int* points, int dims, int clusters)
    : input_vals(input), center_vals(center), label_vals(label), label_vals_reduce(label_reduce), own_points(points), dims(dims), clusters(clusters)
    {}

    __host__ __device__
    void operator()(int& index) const {
        int owner;
        double distance = DBL_MAX;
        for (int i = 0; i < clusters; i++) {
            double sum = 0.0;
            for (int j = 0; j < dims; j++) {
                sum += pow(input_vals[index*dims+j] - center_vals[i*dims+j], 2);
            }
            if (sum < distance) {
                distance = sum;
                owner = i; 
            }
        }
        for (int i = 0; i < dims; i++) {
            label_vals_reduce[index * dims + i] = owner * dims + i;
        }
        own_points[owner]++;
        label_vals[index] = owner;
    }
};

void get_label_thrust(thrust::device_vector<double> & input_vals, 
                      thrust::device_vector<double> & old_centers,
                      thrust::device_vector<double> & new_centers,
                      thrust::device_vector<int> & buffer,
                      thrust::device_vector<int> & labels,
                      thrust::device_vector<int> & labels_for_reduce,
                      thrust::device_vector<int> & n_points,
                      int    dims,
                      int    n_vals,
                      int    n_cluster);
