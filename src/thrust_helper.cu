#include "thrust_helper.cuh"
#include <iostream>
#include <set>

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

    printf("Sizes %lu %lu %lu %lu %lu %lu %lu\n", input_vals.size(), old_centers.size(), new_centers.size(), buffer.size(), labels.size(), labels_for_reduce.size(), n_points.size());
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
    
    int check_range = 1000;
    thrust::device_vector<int> label_check(labels_for_reduce.begin(), labels_for_reduce.begin()+check_range);
    thrust::device_vector<int> owner(n_points.begin(), n_points.end());
    int max_label = 0;
    int min_label = 0;
    std::set<int> test;
    printf("Centoids own ");
    for (int i = 0; i < owner.size()) {
        printf("%d ", owner[i]);
    }
    printf("\n");
    for (int i = 0; i < check_range; i++) {
        int temp = label_check[i];
        max_label = std::max(max_label, temp);
        min_label = std::min(min_label, temp);
        test.emplace(temp);
    }
    printf("Label range (%d-%d) with %lu labels\n", min_label, max_label, test.size());
    
    thrust::reduce_by_key(thrust::device, labels_reduce_p, labels_reduce_p+check_range, input_vals_p, buffer_p, new_centers_p);
    thrust::stable_sort_by_key(thrust::device, buffer_p, buffer_p+buffer.size(), new_centers_p, thrust::less<int>());
}