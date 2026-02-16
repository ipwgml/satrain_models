#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

class BMCICore {
private:
    std::vector<std::vector<double>> X;  // Training data
    std::vector<double> y;               // Training targets
    std::vector<double> Sinv;           // Inverse variances
    int primary_axis;
    double cutoff;
    double delta_t;
    bool has_cutoff;

public:
    BMCICore(const std::vector<double>& sigma, double cutoff_val = -1.0) 
        : cutoff(cutoff_val), has_cutoff(cutoff_val > 0) {
        
        Sinv.resize(sigma.size());
        double min_sigma = std::numeric_limits<double>::max();
        primary_axis = 0;
        
        for (size_t i = 0; i < sigma.size(); i++) {
            Sinv[i] = 1.0 / (sigma[i] * sigma[i]);
            if (sigma[i] < min_sigma) {
                min_sigma = sigma[i];
                primary_axis = i;
            }
        }
        delta_t = cutoff_val * sigma[primary_axis];
    }
    
    void fit(py::array_t<double> X_input, py::array_t<double> y_input) {
        auto X_buf = X_input.request();
        auto y_buf = y_input.request();
        
        if (X_buf.ndim != 2 || y_buf.ndim != 1) {
            throw std::runtime_error("X must be 2D, y must be 1D");
        }
        
        size_t m = X_buf.shape[0];
        size_t n = X_buf.shape[1];
        
        if (y_buf.shape[0] != m) {
            throw std::runtime_error("X and y must have same number of samples");
        }
        
        double* X_ptr = static_cast<double*>(X_buf.ptr);
        double* y_ptr = static_cast<double*>(y_buf.ptr);
        
        // Filter out invalid samples
        std::vector<std::pair<double, size_t>> primary_values;
        
        for (size_t i = 0; i < m; i++) {
            bool valid = true;
            for (size_t j = 0; j < n; j++) {
                if (!std::isfinite(X_ptr[i * n + j])) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                primary_values.push_back({X_ptr[i * n + primary_axis], i});
            }
        }
        
        // Sort by primary axis
        std::sort(primary_values.begin(), primary_values.end());
        
        // Store sorted data
        X.resize(primary_values.size());
        y.resize(primary_values.size());
        
        for (size_t i = 0; i < primary_values.size(); i++) {
            size_t orig_idx = primary_values[i].second;
            X[i].resize(n);
            for (size_t j = 0; j < n; j++) {
                X[i][j] = X_ptr[orig_idx * n + j];
            }
            y[i] = y_ptr[orig_idx];
        }
    }
    
    double retrieve_single(const std::vector<double>& x) {
        if (X.empty()) return 0.0;
        
        // Check if primary axis has valid value
        if (!std::isfinite(x[primary_axis])) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        
        size_t start_idx = 0;
        size_t end_idx = X.size();
        double final_weight_sum = 0.0;
        
        // Use cutoff optimization if applicable
        if (has_cutoff) {
            double primary_val = x[primary_axis];
            
            // Use std::lower_bound for efficient binary search
            auto comparator = [this](const std::vector<double>& sample, double value) {
                return sample[primary_axis] < value;
            };
            
            auto it = std::lower_bound(X.begin(), X.end(), primary_val - delta_t, comparator);
            start_idx = std::distance(X.begin(), it);
            it = std::lower_bound(X.begin(), X.end(), primary_val + delta_t, comparator);
            end_idx = std::distance(X.begin(), it);

            // If no samples in interval, return NaN
            if (start_idx >= end_idx) {
                return std::numeric_limits<double>::quiet_NaN();
            }

            // Calculate weighted sum directly without storing weights
            double weighted_sum = 0.0;
            final_weight_sum = 0.0;

            for (size_t i = start_idx; i < end_idx; i++) {
                double d_squared = 0.0;
                for (size_t j = 0; j < x.size(); j++) {
                    // Skip NaN values in distance calculation
                    if (std::isfinite(x[j]) && std::isfinite(X[i][j])) {
                        double diff = X[i][j] - x[j];
                        d_squared += diff * diff * Sinv[j];
                    }
                }
                double weight = std::exp(-0.5 * d_squared);
                final_weight_sum += weight;
                weighted_sum += y[i] * weight;
            }
            
            // Return NaN if no valid weights
            if (final_weight_sum == 0.0) return std::numeric_limits<double>::quiet_NaN();
            
            return weighted_sum / final_weight_sum;

        } else {
            // No cutoff case - calculate weighted sum directly
            double weighted_sum = 0.0;
            final_weight_sum = 0.0;
            
            for (size_t i = start_idx; i < end_idx; i++) {
                double d_squared = 0.0;
                for (size_t j = 0; j < x.size(); j++) {
                    // Skip NaN values in distance calculation
                    if (std::isfinite(x[j]) && std::isfinite(X[i][j])) {
                        double diff = X[i][j] - x[j];
                        d_squared += diff * diff * Sinv[j];
                    }
                }
                double weight = std::exp(-0.5 * d_squared);
                final_weight_sum += weight;
                weighted_sum += y[i] * weight;
            }
            
            // Return NaN if no valid weights
            if (final_weight_sum == 0.0) return std::numeric_limits<double>::quiet_NaN();
            
            return weighted_sum / final_weight_sum;
        }
    }
    
    py::array_t<double> predict_batch(py::array_t<double> X_input, int n_threads = -1) {
        auto buf = X_input.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Input must be 2D array");
        }
        
        size_t batch_size = buf.shape[0];
        size_t n_features = buf.shape[1];
        double* X_ptr = static_cast<double*>(buf.ptr);
        
        auto result = py::array_t<double>(batch_size);
        double* result_ptr = static_cast<double*>(result.request().ptr);
        
#ifdef _OPENMP
        // Set number of threads if specified
        int original_threads = omp_get_max_threads();
        if (n_threads > 0) {
            omp_set_num_threads(n_threads);
        }
        
        // Parallel processing
        #pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < batch_size; i++) {
            std::vector<double> x(n_features);
            for (size_t j = 0; j < n_features; j++) {
                x[j] = X_ptr[i * n_features + j];
            }
            result_ptr[i] = retrieve_single(x);
        }
        
        // Restore original thread count
        if (n_threads > 0) {
            omp_set_num_threads(original_threads);
        }
#else
        // Fallback to serial processing if OpenMP not available
        for (size_t i = 0; i < batch_size; i++) {
            std::vector<double> x(n_features);
            for (size_t j = 0; j < n_features; j++) {
                x[j] = X_ptr[i * n_features + j];
            }
            result_ptr[i] = retrieve_single(x);
        }
#endif
        
        return result;
    }
    
    py::array_t<double> predict_batch_vectorized(py::array_t<double> X_input, int n_threads = -1) {
        auto buf = X_input.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Input must be 2D array");
        }
        
        // For no-cutoff case, we can vectorize the distance calculations
        if (has_cutoff) {
            return predict_batch(X_input, n_threads);
        }
        
        size_t batch_size = buf.shape[0];
        size_t n_features = buf.shape[1];
        size_t n_samples = X.size();
        double* X_ptr = static_cast<double*>(buf.ptr);
        
        auto result = py::array_t<double>(batch_size);
        double* result_ptr = static_cast<double*>(result.request().ptr);
        
#ifdef _OPENMP
        // Set number of threads if specified
        int original_threads = omp_get_max_threads();
        if (n_threads > 0) {
            omp_set_num_threads(n_threads);
        }
        
        // Parallel processing over batch
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < batch_size; i++) {
            // Find valid features for this observation
            std::vector<size_t> valid_indices;
            for (size_t j = 0; j < n_features; j++) {
                if (std::isfinite(X_ptr[i * n_features + j])) {
                    valid_indices.push_back(j);
                }
            }
            
            if (valid_indices.empty()) {
                result_ptr[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            
            // Calculate weighted sum directly without storing weights
            double weighted_sum = 0.0;
            double weight_sum = 0.0;
            
            // This inner loop could also be parallelized, but may cause overhead
            // for smaller datasets due to nested parallelism
            for (size_t s = 0; s < n_samples; s++) {
                double d_squared = 0.0;
                for (size_t j : valid_indices) {
                    double diff = X[s][j] - X_ptr[i * n_features + j];
                    d_squared += diff * diff * Sinv[j];
                }
                double weight = std::exp(-0.5 * d_squared);
                weight_sum += weight;
                weighted_sum += y[s] * weight;
            }
            
            if (weight_sum == 0.0) {
                result_ptr[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            
            result_ptr[i] = weighted_sum / weight_sum;
        }
        
        // Restore original thread count
        if (n_threads > 0) {
            omp_set_num_threads(original_threads);
        }
#else
        // Fallback to serial processing
        for (size_t i = 0; i < batch_size; i++) {
            std::vector<double> x(n_features);
            for (size_t j = 0; j < n_features; j++) {
                x[j] = X_ptr[i * n_features + j];
            }
            result_ptr[i] = retrieve_single(x);
        }
#endif
        
        return result;
    }
};

PYBIND11_MODULE(bmci_c, m) {
    py::class_<BMCICore>(m, "BMCICore")
        .def(py::init<const std::vector<double>&, double>(), 
             py::arg("sigma"), py::arg("cutoff") = -1.0)
        .def("fit", &BMCICore::fit)
        .def("predict_batch", &BMCICore::predict_batch, py::arg("X"), py::arg("n_threads") = -1)
        .def("predict_batch_vectorized", &BMCICore::predict_batch_vectorized, py::arg("X"), py::arg("n_threads") = -1);
        
#ifdef _OPENMP
    m.def("openmp_available", []() { return true; });
    m.def("openmp_max_threads", []() { return omp_get_max_threads(); });
#else
    m.def("openmp_available", []() { return false; });
    m.def("openmp_max_threads", []() { return 1; });
#endif
}
