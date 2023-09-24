#include "kmeans_kernel.cuh"

__global__ void get_label(double *input_vals_c, 
                         double *centers_c, 
                         int    *labels_c,
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c,
                         int *n_points_c){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int array_index = index * dims;
    if (index < n_vals){
        
        double distance = DBL_MAX;
        double temp = DBL_MAX;
        for (int i = 0; i < n_cluster; i++){
            double sum=0.0;
            for (int j = 0; j < dims; j++){
                sum+=pow((input_vals_c[array_index+j] - centers_c[i*dims+j]), 2);
            }
            temp = sqrt(sum);
            if (temp < distance){
                distance = temp;
                labels_c[index] = i;
            }
                
                
        }
        

        int center_index = labels_c[index];
        //printf("label:%d\n",center_index);
        
        atomicAdd(&n_points_c[center_index], 1);
        for (int i = 0; i < dims; i++){
            //temp_centers_c[center_index+j]+= input_vals_c[array_index+i];
            atomicAdd(&temp_centers_c[center_index*dims+i], input_vals_c[array_index+i]);  
                  
        }

    } 
}

void wrapper_get_label(double *input_vals_c, 
                         double *centers_c, 
                         int    *labels_c,
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c,
                         int *n_points_c,
                         int blocks,
                         int threads)
{
    get_label<<<blocks, threads>>>(input_vals_c,
                                   centers_c,
                                   labels_c,
                                   dims,
                                   n_vals,
                                   n_cluster,
                                   temp_centers_c,
                                   n_points_c);

}

