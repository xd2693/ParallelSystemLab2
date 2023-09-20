#include "io.h"
#include "argparse.h"
#include "helper.h"
#include <math.h>

double distance(kmeans_args *args, int input_index, int center_index){
    
    double sum = 0.0;
    for (int i =0; i<args->dims; i++){
        sum+= pow((args->input_vals[input_index+i]-args->Scenters[center_index+i]),2)
    }
    return sqrt(sum)



}

