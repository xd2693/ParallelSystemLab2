#include "io.h"


void read_file(struct options_t* args,
               int*                 n_vals,
               double**             input_vals,
               int**             	labels,
			   double**             centers) {

  	// Open file
	std::ifstream in;
	in.open(args->in_file);
	// Get num vals
	in >> *n_vals;
	printf("n_vals=%d dims=%d ",*n_vals, args->dims);

	// Alloc input and lables and clusters arrays
	size_t input_vals_size = (*n_vals) * (args->dims);
	size_t labels_size = *n_vals * sizeof(int);
	size_t centers_size = args->n_cluster * args->dims * sizeof(double);
	*input_vals = (double*) malloc(input_vals_size * sizeof(double));
	*labels = (int*) malloc(input_vals_size);
	*centers = (double*) malloc(centers_size);

	//printf("size=%lu ",sizeof(**input_vals));

	// Read input vals
    int j = 0;
	double temp=0.0;
	for (int i = 0; i< input_vals_size; ++i) {//*n_vals * args->dims
		if(j == 0){
			in >> temp;
			printf("temp=%5.2f\n",temp);			
		}
			
		in >> (*input_vals)[i];
		//in >> temp;
		
		j++;
		if (j >= args->dims){//
			j = 0;
		}
		//printf("%.12f\n", (*input_vals)[i]);
	}
	
}

/*void write_file(struct options_t*         args,
               	struct prefix_sum_args_t* opts) {
  // Open file
	std::ofstream out;
	out.open(args->out_file, std::ofstream::trunc);

	// Write solution to output file
	for (int i = 0; i < opts->n_vals; ++i) {
		out << opts->output_vals[i] << std::endl;
	}

	out.flush();
	out.close();
	
	// Free memory
	free(opts->input_vals);
	free(opts->output_vals);
}*/
