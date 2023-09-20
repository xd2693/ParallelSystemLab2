#include <argparse.h>

void get_opts(int argc,
              char **argv,
              struct options_t *opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t--n_cluster or -k <num_cluster>" << std::endl;
        std::cout << "\t--dims or -d <dimension>" << std::endl;
        std::cout << "\t--in or -i <file_path>" << std::endl;
        std::cout << "\t--max_iter or -m <max_num_iter>" << std::endl;
        std::cout << "\t--threshold or -t <threshold>" << std::endl;    
        std::cout << "\t[Optional] --con_flag or -c" << std::endl;
        std::cout << "\t--seed or -s <seed>" << std::endl;
        exit(0);
    }

    opts->c_flag = false;

    struct option l_opts[] = {
        {"n_cluster", required_argument, NULL, 'k'},
        {"dims", required_argument, NULL, 'd'},
        {"in", required_argument, NULL, 'i'},
        {"max_iter", required_argument, NULL, 'm'},
        {"threshold", required_argument, NULL, 't'},    
        {"seed", required_argument, NULL, 's'},
        {"c_flag", no_argument, NULL, 'c'}
    };

    int ind, c;
    while ((c = getopt_long(argc, argv, "k:d:i:m:t:cs:", l_opts, &ind)) != -1)
    {
        switch (c)
        {
        case 0:
            break;        
        case 'k':
            opts->n_cluster = atoi((char *)optarg);
            break;
        case 'd':
            opts->dims = atoi((char *)optarg);
            break;
        case 'i':
            opts->in_file = (char *)optarg;
            break;
        case 'm':
            opts->max_iter = atoi((char *)optarg);
            break;
        case 't':
            opts->threshold = atof((char *)optarg);
            break;
        case 'c':
            opts->c_flag = true;
            break;
        case 's':
            //printf("Reached s %s\n", (char *)optarg);
            opts->seed = atoi((char *)optarg);
            break;
        case ':':
            exit(1);
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
}
