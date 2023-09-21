#include "kmeans.h"

//get distance between input and centroid
double get_distance(kmeans_args *args, 
                int          input_index,
                int          center_index){
    
    double sum = 0.0;
    for (int i = 0; i < args->dims; i++){
        sum+= pow(((args->input_vals[input_index+i])-(args->centers[center_index+i])),2);
    }
    return sqrt(sum);
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
    //printf("point %d closest to center %d distance %.5f\n", index/args->dims, label, distance);
    return label;
}

void get_new_centers(kmeans_args *args){
    int n_cluster = args->n_cluster;
    int n_vals = args->n_vals;
    int dims = args->dims;
    int n_points[n_cluster]={}; //for count how many points in each cluster
    double new_centers[args->n_cluster * args->dims]={};
    //memset(args->centers, 0, sizeof(double) * args->dims * args->n_cluster);
    for (int i = 0; i < n_vals; i++){
        int center = args->labels[i];
        n_points[center] ++;
        for (int j = 0; j < dims; j++){
            
            int p_index = i * dims + j;
            
            (new_centers[center*dims+j])+= args->input_vals[p_index]; 
            
        }
    }
    for (int i = 0; i < n_cluster; i++){
        for (int j = 0; j < dims; j++){
            int index = i * dims + j;
            if (n_points[i] != 0)
                args->centers[index] = (new_centers[index]) / n_points[i];
            //printf("test1: %lf  ",args->centers[index]);
        }
    }
}

bool test_converge(kmeans_args *args, double *old_centers){
    bool converge = true;
    for (int i = 0; i < sizeof(old_centers); i++){
        if (fabs(old_centers[i] - args->centers[i]) > (args->threshold)){
            converge = false;
            break;
        }
    }

    return converge;
}

void kmeans_cpu(kmeans_args *args){
    double *old_centers;
    int size = (args->dims) * (args->n_cluster) * sizeof(double);
    old_centers = (double*) malloc(size);
    for (int i = 0; i < args->max_iter; i++){
        memcpy(old_centers, args->centers, size);
        //printf("old_centers1: %lf\n", old_centers[7*args->dims]);
        //set label for each point
        for (int j =0; j < args->n_vals; j++){
            args->labels[j] = get_label(args, j * (args->dims));
        }
        //printf("centers1 : %lf\n", args->centers[7*args->dims]);
        //compute new centroids
        get_new_centers(args);
        //printf("old_centers2: %lf\n", old_centers[7*args->dims]);
        //convergence test
        if(test_converge(args,old_centers)){
            //printf("iter= %d\n", i);
            break;
        }
        //printf("centers2 : %lf\n", args->centers[7*args->dims]);
    }
}
