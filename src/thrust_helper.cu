#include "thrust_helper.cuh"
#include <iostream>
#include <set>



void get_label_thrust(thrust::device_vector<double> & input_vals, 
                      thrust::device_vector<double> & old_centers,
                      thrust::device_vector<double> & new_centers,
                      thrust::device_vector<int> & buffer,
                      thrust::device_vector<int> & labels,
                      thrust::device_vector<int> & labels_for_reduce,
                      thrust::device_vector<int> & n_points,
                      int    dims,
                      int    n_vals,
                      int    n_cluster)
{


    double* input_vals_p = thrust::raw_pointer_cast(input_vals.data());
    double* old_centers_p = thrust::raw_pointer_cast(old_centers.data());
    double* new_centers_p = thrust::raw_pointer_cast(new_centers.data());
    int* buffer_p = thrust::raw_pointer_cast(buffer.data());
    int* labels_reduce_p = thrust::raw_pointer_cast(labels_for_reduce.data());
    int* n_points_p = thrust::raw_pointer_cast(n_points.data());
    thrust::sequence(thrust::device, labels.begin(), labels.end(), 0);
    CentoidAssignFunctor functor(input_vals_p, old_centers_p, labels_reduce_p, dims, n_cluster);
    
    thrust::for_each(thrust::device, labels.begin(), labels.end(), functor);
    
 
    thrust::device_vector<int> own_sort(n_points.size());
    thrust::device_vector<int> vector_one(n_vals, 1);
    thrust::device_vector<int> labels_copy(labels);
    int *vector_one_p = thrust::raw_pointer_cast(vector_one.data());
    int *own_sort_p = thrust::raw_pointer_cast(own_sort.data());
    int *labels_copy_p = thrust::raw_pointer_cast(labels_copy.data());

    thrust::device_vector<int> labels_for_reduce_copy(labels_for_reduce);
    thrust::device_vector<double> input_vals_copy(input_vals);
    int *labels_for_reduce_copy_p = thrust::raw_pointer_cast(labels_for_reduce_copy.data());
    double *input_vals_copy_p = thrust::raw_pointer_cast(input_vals_copy.data());

    thrust::stable_sort_by_key(thrust::device, labels_copy_p, labels_copy_p+n_vals, vector_one_p, thrust::less<int>());
    thrust::reduce_by_key(thrust::device, labels_copy_p, labels_copy_p+n_vals, vector_one_p, own_sort_p, n_points_p);
    thrust::stable_sort_by_key(thrust::device, own_sort_p, own_sort_p+n_cluster, n_points_p, thrust::less<int>());

    thrust::stable_sort_by_key(thrust::device, labels_for_reduce_copy_p, labels_for_reduce_copy_p+n_vals*dims, input_vals_copy_p);
    thrust::reduce_by_key(thrust::device, labels_for_reduce_copy_p, labels_for_reduce_copy_p+n_vals*dims, input_vals_copy_p, buffer_p, new_centers_p);
    thrust::stable_sort_by_key(thrust::device, buffer_p, buffer_p+n_cluster*dims, new_centers_p, thrust::less<int>());

    
}

void check_vector(thrust::device_vector<double> &input)
{
    thrust::host_vector<double> helper(input.begin(), input.end());
    printf("double check\n");
    for (int i = 0; i < helper.size(); i++)
    {
        printf("%.5f\n",helper[i]);
    }
}

void check_vector(thrust::device_vector<int> &input)
{
    thrust::host_vector<int> helper(input.begin(), input.end());
    printf("int check\n");
    for (int i = 0; i < helper.size(); i++)
    {
        printf("%d\n", helper[i]);
    }
}