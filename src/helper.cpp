#include "io.h"
#include "argparse.h"
#include "helper.h"

void fill_kmeans_args(kmeans_args *args,
                      int         n_cluster,
                      int         n_vals,
                      int         dims,
                      int         max_iter,
                      double      threshold,
                      double      *input_vals,
                      double      *centers,
                      int         *labels){

    *args={n_cluster, n_vals, dims, max_iter, threshold,
          input_vals, centers, labels};
}

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;
int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}
void kmeans_srand(unsigned int seed) {
    next = seed;
}

void random_centers(int seed, kmeans_args *args) {
    kmeans_srand(seed); 
    int in=0;
    for (int i=0; i<args->n_cluster; i++){
        int index = (kmeans_rand() % args->n_vals);

        int my_index= index * args->dims;
        printf("\n index= %d\n",index);
        for (int j=0; j< args->dims; j++){
            args->centers[in] = args->input_vals[my_index+j];
            //printf("\t centers= %.12f",args->centers[in]);
            in++;
        }
        
    }
}

void output(kmeans_args *args, bool c_flag){
    if(c_flag){
        for (int clusterId = 0; clusterId < args->n_cluster; clusterId ++){
            printf("%d ", clusterId);
            for (int d = 0; d < args->dims; d++)
                printf("%lf ", centers[clusterId * args->dims + d ]);
            printf("\n");
        }
    }
    else{
        printf("cluster:");
        for (int p=0; p < args->n_vals; p++)
            printf(" %d", args->labels[p]);
    }
}


