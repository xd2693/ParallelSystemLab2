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
    
    int check_range = 50;
    thrust::host_vector<int> owner_before(n_points.begin(), n_points.end());
    printf("Centoids own before");
    for (int i = 0; i < owner_before.size(); i++) {
        printf("%d ", owner_before[i]);
    }
    printf("\n");

    thrust::host_vector<int> label_check_1(labels.begin(), labels.begin()+check_range);
    printf("label_check_1");
    for (int i = 0; i < label_check_1.size(); i++) {
        printf("%d ", label_check_1[i]);
    }
    printf("\n");

    thrust::host_vector<double> input_check(input_vals.begin(), input_vals.begin()+check_range);
    printf("Input check");
    for (int i = 0; i < input_check.size(); i++)
    {
        printf("%.5f ", input_check[i]);
    }
    printf("\n");

    thrust::host_vector<double> newc_check_b(new_centers.begin(), new_centers.end());
    printf("newc_check_b");
    for (int i = 0; i < newc_check_b.size(); i++)
    {
        printf("%.5f ", newc_check_b[i]);
    }
    printf("\n");

    thrust::for_each(thrust::device, labels.begin(), labels.end(), functor);
    
    thrust::host_vector<int> label_a(labels.begin(), labels.end());
    printf("label_a");
    for (int i = 0; i < label_a.size(); i++) {
        printf("%d\n", label_a[i]);
    }
    printf("\n");
    thrust::host_vector<int> label_check(labels_for_reduce.begin(), labels_for_reduce.begin()+check_range);
    thrust::host_vector<int> owner(n_points.begin(), n_points.end());
    int max_label = 0;
    int min_label = 0;
    std::set<int> test;
    printf("Centoids own ");
    for (int i = 0; i < owner.size(); i++) {
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

    thrust::host_vector<double> newc_check_a(new_centers.begin(), new_centers.end());
    printf("newc_check_a");
    for (int i = 0; i < newc_check_a.size(); i++)
    {
        printf("%.5f ", newc_check_a[i]);
    }
    printf("\n");
}