#include <iostream>
#include <chrono>
#include <atomic>
#include <assert.h>
#include <fstream>
#include <string>
#include <math.h>
#include <sstream>
#include <map>
#include <vector>
#include <mpi.h>

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::map;
using std::vector;

inline std::chrono::steady_clock::time_point get_current_time_fenced() {
    assert(std::chrono::steady_clock::is_steady &&
           "Timer should be steady (monotonic).");
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::steady_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}

template<class D>
inline long long to_us(const D& d)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
}

struct Parameters {
    static const int m = 5;
    double c[m] = {2.0, 1.0, 4.0, 7.0, 2.0};
    double a1[m] = {1.0, 2.0, 1.0, 1.0, 5.0};
    double a2[m] = {4.0, 5.0, 1.0, 2.0, 4.0};
    double minX = -10;
    double maxX = 10;
    double minY = -10;
    double maxY = 10;
    double rel_err = 0.001;
    double abs_err = 0.05;
    size_t initial_steps = 100;
    size_t max_steps = 1000;
    size_t n_threads = 1;
    double expect_res = -1.604646665;
};

map<string, string> read_config(const string& filename, Parameters& params) {
    std::ifstream config_stream(filename);
    
    if(!config_stream.is_open()) {
        cerr << "Failed to open configuration file " << filename << endl;
        exit(2);
    }
    
    map<string, string> parsed_params;
    string line;
    
    while (config_stream >> line) {
        std::istringstream is_line(line);
        std::string key;
        if (std::getline(is_line, key, '=')) {
            std::string value;
            if (std::getline(is_line, value)) {
                parsed_params[key] = value;
            }
        }
    }
    
    config_stream.close();
    
    return parsed_params;
}

void assign_config(Parameters &params, const map<string, string>& conf) {
    try {
        params.n_threads = std::stoul(conf.at("threads"));
        params.minX = std::stod(conf.at("minX"));
        params.maxX = std::stod(conf.at("maxX"));
        params.minY = std::stod(conf.at("minY"));
        params.maxY = std::stod(conf.at("maxY"));
        params.rel_err = std::stod(conf.at("rel_err"));
        params.abs_err = std::stod(conf.at("abs_err"));
        params.initial_steps = std::stoul(conf.at("initial_steps"));
        params.max_steps = std::stoul(conf.at("max_steps"));
    } catch (...) {
        cerr << "Error: some problem with conversion occured" << endl;
        exit(3);
    }
}

double function_to_integrate(double x1, double x2, const Parameters& params) {
    double result = 0.0;
    for (int i = 0; i < params.m; ++i) {
        result += params.c[i] * exp(-(1/M_PI) * (pow((x1 - params.a1[i]), 2) + pow((x2 - params.a2[i]), 2))) * cos(M_PI * (pow((x1 - params.a1[i]), 2) + pow((x2 - params.a2[i]), 2)));
    }
    return -result;
}

template <typename func_T>
void integrate(func_T myfunc, const Parameters& params, double& result) {
    double res = 0.0;
    double x_start = params.minX;
    double y_start = params.minY;
    double delta_x = (params.maxX - params.minX) / params.initial_steps;
    double delta_y = (params.maxY - params.minY) / params.initial_steps;
    size_t N = params.initial_steps;

    for (size_t i = 0; i < N; ++i) {
        double x = ((x_start + i * delta_x) + (x_start + (i+1) * delta_x))/2;
        for (int j = 0; j < N; ++j) {
            double y = ((y_start + j * delta_y) + (y_start + (j+1) * delta_y))/2;
            res += myfunc(x, y, params) * delta_x * delta_y;
        }
    }
    result = res;
}

int main(int argc, const char * argv[]) {
    Parameters params;
    string filename("conf.txt");
    auto conf = read_config(filename, params);
    assign_config(params, conf);
    
    if (argc == 2) {
        filename = argv[1];
        conf = read_config(filename, params);
        assign_config(params, conf);
        
    } else if (argc > 2) {
        cerr << "Error: too many arguments passed" << endl;
        exit(-1);
    }
    
    double result = 0.0;
    double prev_result;
    double error_abs, error_rel;
    int commsize, rank, len;
    char procname[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(procname, &len);
    int numOfThreads = commsize;
    double from_to[commsize];
    
    do {

        prev_result = result;
        auto start = get_current_time_fenced();
        
        integrate(function_to_integrate, params, result);
        cout << result << endl;
        
        
        error_abs = abs(prev_result - result);
        error_rel = abs(error_abs / prev_result);
        cout << "Steps frequency: " << params.initial_steps << endl;
        cout << "Absolute error: " << error_abs << endl;
        cout << "Relative error: " << error_rel << endl;
        params.initial_steps *= 2;
        
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(&res, 1, MPI_DOUBLE, from_to, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        for (size_t i = 0; i < commsize; ++i)
        {
            result += from_to[i];
        }
        
        MPI_Finalize();
        auto stop = get_current_time_fenced();
        cout << "time: " << to_us(total) << "ms" << endl;
    } while ((error_abs > params.abs_err) && (error_rel > params.rel_err));
    
    
    if (rank == 1)
    {
        
        cout << "Last result = " << result << endl;
    }
    
    return 0;
}
