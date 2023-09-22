#include <cmath.h>

__global__void get_label(double *input_vals_c, 
                         double *centers_c, 
                         int    *label,
                         int    dims,
                         int    n_vals,
                         int    n_cluster,
                         double *temp_centers_c){
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
            if (temp < distance)
                distance = temp;
                label[index] = i;
        }
    __syncthreads();

    int center_index = label[index];
    for (int i = 0; i < dims; i++){
        temp_centers_c[center_index+j]+= input_vals_c[array_index+i];

    }

    }
}

__global__void 