#include "thrust_helper.cuh"
#include <iostream>

/*void get_label_thrust(thrust::device_vector<double> & input_vals, 
                      thrust::device_vector<double> & old_centers,
                      thrust::device_vector<int> & labels,
                      thrust::device_vector<int> & labels_for_reduce,
                      thrust::device_vector<double> & new_centers, 
                      thrust::device_vector<int> & n_points,
                      int    dims,
                      int    n_vals,
                      int    n_cluster)
{
    thrust::counting_iterator<int> first_point(0);
    thrust::counting_iterator<int> last_point = first_point + n_vals;

}*/

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

    printf("Sizes %lu %lu %lu %lu %lu %lu\n", input_vals.size(), old_centers.size(), new_centers.size(), labels.size(), labels_for_reduce.size(), n_points.size());
    double* input_vals_p = thrust::raw_pointer_cast(input_vals.data());
    double* old_centers_p = thrust::raw_pointer_cast(old_centers.data());
    double* new_centers_p = thrust::raw_pointer_cast(new_centers.data());
    int* buffer_p = thrust::raw_pointer_cast(buffer.data());
    int* labels_p = thrust::raw_pointer_cast(labels.data());
    int* labels_reduce_p = thrust::raw_pointer_cast(labels_for_reduce.data());
    int* n_points_p = thrust::raw_pointer_cast(n_points.data());
    thrust::sequence(thrust::device, labels.begin(), labels.end(), 0);
    CentoidAssignFunctor functor(input_vals_p, old_centers_p, labels_p, labels_reduce_p, n_points_p, dims, n_cluster);
    thrust::for_each(thrust::device, labels.begin(), labels.end(), functor);
    thrust::reduce_by_key(thrust::device, labels_reduce_p, labels_reduce_p+labels_for_reduce.size(), input_vals_p, buffer_p, new_centers_p);
    thrust::stable_sort_by_key(thrust::device, buffer_p, buffer_p+buffer.size(), new_centers_p, thrust::less<int>());
}