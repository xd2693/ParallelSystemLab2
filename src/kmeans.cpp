#include "kmeans.h"

//get distance between input and center
double get_distance(kmeans_args* args, 
                int          input_index,
                int          center_index){
    
    double sum = 0.0;
    for (int i = 0; i < args->dims; i++){
        sum+= pow((args->input_vals[input_index+i]-args->centers[center_index+i]),2)
    }
    return sqrt(sum)
}

//set label for the point of index
int get_label (kmeans_args* args, int index){
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

void get_new_centers(kmeans_args* args, double* new_centers){
    int n_points[args->n_cluster]={}; //count how many points in each cluster
    for (int i = 0; i < args->n_vals; i++){
        int center = args->labels[i];
        n_points[center] ++;
        for (int j = 0; j < args->dims; j++){
            int p_index = i * args->dims + j;
            new_centers[center+j]+= args->input_vals[p_index];            
        }
    }
    for (int i = 0; i < args->n_cluster; i++){
        for (int j = 0; j < args->dims; j++){
            int index = i * dim + j;
            new_centers[index] = new_centers / n_points[i];
        }
    }
}

bool test_converge(kmeans_args* args, double* new_centers){
    bool converge = true;
    for (int i = 0; i < sizeof(new_centers); i++){
        if (fabs(new_centers[i] - args->centers[i]) > args->threshold){
            converge = false;
            args->centers[i] = new_centers[i];
        }
    }
    return converge;
}

void kmeans_cpu(kmeans_args* args){
    for (int i = 0; i <= args->max_iter; i++){
        //set label for each point
        for (int j =0; j < args->n_vals; j++){
            label[j] = get_label()
        }
    }
}
