#include "io.h"
#include "argparse.h"
#include <math.h>

double distance(int      dims, 
                int      input_index,
                int      center_index,
                double*  input_vals,
                double*  centers){
    
    double sum = 0.0;
    for (int i =0; i<dims; i++){
        sum+= pow((input_vals[input_index+i]-centers[center_index+i]),2)
    }
    return sqrt(sum)



}

