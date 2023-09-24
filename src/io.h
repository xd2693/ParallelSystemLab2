#ifndef _IO_H
#define _IO_H

#include "argparse.h"
#include <iostream>
#include <fstream>
#include <chrono>

void read_file(struct options_t* args,
               int*                 n_vals,
               double**             input_vals,
               int**             	labels,
			   double**             centers);

/*struct timing_device {
    auto start, end;
    uint64_t elapsed_time {0};
    uint64_t last_start;
    void start_timing() {
        start = std::chrono::high_resolution_clock::now();
        //last_start = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
    }
    void end_timing() {
        end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        elapsed_time += diff.count();
    }
    void show_time() {
        printf("%lu passed\n", elapsed_time);
    }
};*/

#endif
