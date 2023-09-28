#include "kmeans_seq.h"

/***get distance before square root between input and centroid
****since distance is only need to be compared for picking the closest centroid,
****we don't have to do squre root for comparing.
***/ 
double get_distance(kmeans_args *args, 
                int          input_index,
                int          center_index){
    
    double sum = 0.0;
    for (int i = 0; i < args->dims; i++){
        double d1 = args->input_vals[input_index+i]-args->centers[center_index+i];
        sum += (d1 * d1);
    }
    //return sqrt(sum);
    return sum;
}

//set label for the point, index is for point index in input_vals
int get_label (kmeans_args *args, int index){
    double distance = DBL_MAX;
    double temp = DBL_MAX;
    int label = 0;
    for (int i = 0; i < args->n_cluster; i++){
        temp = get_distance(args, index, i * args->dims);
        if (temp < distance){
            label = i;
            distance = temp;
        }
    }
    
    return label;
}

//get new centroids from reassigned points
void get_new_centers(kmeans_args *args){
    int n_cluster = args->n_cluster;
    int n_vals = args->n_vals;
    int dims = args->dims;
    int n_points[n_cluster]={}; //for count how many points in each cluster
    double new_centers[args->n_cluster * args->dims]={};

    for (int i = 0; i < n_vals; i++){
        int center = args->labels[i];
        n_points[center] ++;
        for (int j = 0; j < dims; j++){
            
            int p_index = i * dims + j;
            
            (new_centers[center*dims+j])+= args->input_vals[p_index]; 
            
        }
    }
    //update centroids
    for (int i = 0; i < n_cluster; i++){
        //if all points got reassigned, don't update the centroid
        if (n_points[i] == 0)
            continue;
        for (int j = 0; j < dims; j++){
            int index = i * dims + j;
            args->centers[index] = (new_centers[index]) / n_points[i];
        }
    }
}

bool test_converge(kmeans_args *args, double *old_centers, int n_cluster, int dims){
    
    for (int i = 0; i < n_cluster * dims; i++){
        if (fabs(old_centers[i] - args->centers[i]) > (args->threshold)){
            return false;
        }
    }

    return true;
}

int kmeans_cpu(kmeans_args *args){
    double *old_centers;
    int size = (args->dims) * (args->n_cluster) * sizeof(double);
    old_centers = (double*) malloc(size);
    for (int i = 0; i < args->max_iter; i++){
        memcpy(old_centers, args->centers, size);
        
        //set label for each point
        for (int j =0; j < args->n_vals; j++){
            args->labels[j] = get_label(args, j * (args->dims));
        }
        
        //compute new centroids
        get_new_centers(args);
        
        //convergence test
        if(test_converge(args,old_centers, args->n_cluster, args->dims)){
            i++;
            return i;
        }
        
    }
    return args->max_iter;
}
