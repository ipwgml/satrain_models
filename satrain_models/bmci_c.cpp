#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <immintrin.h>  // For SIMD intrinsics
#ifdef _OPENMP
#include <omp.h>
#endif

// Compiler optimization hints
#if defined(__GNUC__) || defined(__clang__)
    #define LIKELY(x)    __builtin_expect(!!(x), 1)
    #define UNLIKELY(x)  __builtin_expect(!!(x), 0)
    #define FORCE_INLINE __attribute__((always_inline)) inline
    #define RESTRICT     __restrict__
#elif defined(_MSC_VER)
    #define LIKELY(x)    (x)
    #define UNLIKELY(x)  (x)
    #define FORCE_INLINE __forceinline
    #define RESTRICT     __restrict
#else
    #define LIKELY(x)    (x)
    #define UNLIKELY(x)  (x)
    #define FORCE_INLINE inline
    #define RESTRICT
#endif

namespace py = pybind11;

// SIMD-optimized distance calculation for aligned data
FORCE_INLINE double calculate_distance_simd(
    const double* RESTRICT sample_ptr, 
    const double* RESTRICT query_ptr, 
    const double* RESTRICT Sinv_ptr, 
    size_t n_features) {
    
    double d_squared = 0.0;
    
#ifdef __AVX2__
    // AVX2 vectorized loop for 4 doubles at a time
    size_t simd_end = n_features & ~3;  // Round down to multiple of 4
    __m256d sum_vec = _mm256_setzero_pd();
    
    for (size_t j = 0; j < simd_end; j += 4) {
        __m256d sample = _mm256_loadu_pd(&sample_ptr[j]);
        __m256d query = _mm256_loadu_pd(&query_ptr[j]);
        __m256d sinv = _mm256_loadu_pd(&Sinv_ptr[j]);
        
        // Check for finite values (simplified check)
        __m256d diff = _mm256_sub_pd(sample, query);
        __m256d diff_sq = _mm256_mul_pd(diff, diff);
        __m256d weighted = _mm256_mul_pd(diff_sq, sinv);
        
        sum_vec = _mm256_add_pd(sum_vec, weighted);
    }
    
    // Horizontal sum of the vector
    double temp[4];
    _mm256_storeu_pd(temp, sum_vec);
    d_squared = temp[0] + temp[1] + temp[2] + temp[3];
    
    // Handle remaining elements
    for (size_t j = simd_end; j < n_features; j++) {
        if (LIKELY(std::isfinite(query_ptr[j]) && std::isfinite(sample_ptr[j]))) {
            double diff = sample_ptr[j] - query_ptr[j];
            d_squared += diff * diff * Sinv_ptr[j];
        }
    }
#else
    // Fallback scalar implementation with optimization hints
    for (size_t j = 0; j < n_features; j++) {
        if (LIKELY(std::isfinite(query_ptr[j]) && std::isfinite(sample_ptr[j]))) {
            double diff = sample_ptr[j] - query_ptr[j];
            d_squared += diff * diff * Sinv_ptr[j];
        }
    }
#endif
    
    return d_squared;
}

class BMCICore {
private:
    std::vector<double> X;              // Training data (continuous storage)
    std::vector<double> y;               // Training targets
    std::vector<double> Sinv;           // Inverse variances
    size_t n_samples;                   // Number of training samples
    size_t n_features;                  // Number of features
    int primary_axis;
    double cutoff;
    double delta_t;
    bool has_cutoff;

public:
    BMCICore(const std::vector<double>& sigma, double cutoff_val = -1.0) 
        : cutoff(cutoff_val), has_cutoff(cutoff_val > 0), n_samples(0), n_features(sigma.size()) {
        
        Sinv.reserve(sigma.size());
        double min_sigma = std::numeric_limits<double>::max();
        primary_axis = 0;
        
        for (size_t i = 0; i < sigma.size(); i++) {
            double inv_var = 1.0 / (sigma[i] * sigma[i]);
            Sinv.push_back(inv_var);
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
        
        if (y_buf.shape[0] != m || n != n_features) {
            throw std::runtime_error("X and y must have same number of samples and X must match expected features");
        }
        
        double* X_ptr = static_cast<double*>(X_buf.ptr);
        double* y_ptr = static_cast<double*>(y_buf.ptr);
        
        // Filter out invalid samples and collect indices
        std::vector<std::pair<double, size_t>> primary_values;
        primary_values.reserve(m);
        
        for (size_t i = 0; i < m; i++) {
            bool valid = true;
            // Check validity more efficiently by checking primary axis first
            if (!std::isfinite(X_ptr[i * n + primary_axis])) {
                continue;
            }
            
            for (size_t j = 0; j < n; j++) {
                if (!std::isfinite(X_ptr[i * n + j])) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                primary_values.emplace_back(X_ptr[i * n + primary_axis], i);
            }
        }
        
        // Sort by primary axis for efficient cutoff operations
        std::sort(primary_values.begin(), primary_values.end());
        
        // Update sample count
        n_samples = primary_values.size();
        
        // Allocate continuous memory for X and y
        X.resize(n_samples * n_features);
        y.resize(n_samples);
        
        // Copy sorted data to continuous storage
        for (size_t i = 0; i < n_samples; i++) {
            size_t orig_idx = primary_values[i].second;
            
            // Copy row efficiently using pointer arithmetic
            const double* src = X_ptr + orig_idx * n;
            double* dst = X.data() + i * n_features;
            std::copy(src, src + n_features, dst);
            
            y[i] = y_ptr[orig_idx];
        }
    }
    
    // Helper function to get X value at [sample_idx, feature_idx]
    inline double get_X(size_t sample_idx, size_t feature_idx) const {
        return X[sample_idx * n_features + feature_idx];
    }
    
    double retrieve_single(const std::vector<double>& x) {
        if (n_samples == 0) return 0.0;
        
        // Check if primary axis has valid value
        if (!std::isfinite(x[primary_axis])) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        
        size_t start_idx = 0;
        size_t end_idx = n_samples;
        
        // Use cutoff optimization if applicable
        if (has_cutoff) {
            double primary_val = x[primary_axis];
            
            // Binary search for cutoff range using raw pointer arithmetic
            const double* X_ptr = X.data();
            
            // Find lower bound: first sample >= primary_val - delta_t
            size_t left = 0, right = n_samples;
            while (left < right) {
                size_t mid = left + (right - left) / 2;
                if (X_ptr[mid * n_features + primary_axis] < primary_val - delta_t) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            start_idx = left;
            
            // Find upper bound: first sample >= primary_val + delta_t
            left = start_idx;
            right = n_samples;
            while (left < right) {
                size_t mid = left + (right - left) / 2;
                if (X_ptr[mid * n_features + primary_axis] < primary_val + delta_t) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            end_idx = left;

            // If no samples in interval, return NaN
            if (start_idx >= end_idx) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }

        // Calculate weighted sum with optimized loop
        double weighted_sum = 0.0;
        double weight_sum = 0.0;
        const double* X_ptr = X.data();
        const double* Sinv_ptr = Sinv.data();

        for (size_t i = start_idx; i < end_idx; i++) {
            const double* sample_ptr = X_ptr + i * n_features;
            
            // Prefetch next cache line for better memory access pattern
            if (LIKELY(i + 1 < end_idx)) {
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(X_ptr + (i + 1) * n_features, 0, 1);
#endif
            }
            
            // Use SIMD-optimized distance calculation
            double d_squared = calculate_distance_simd(sample_ptr, x.data(), Sinv_ptr, n_features);
            
            double weight = std::exp(-0.5 * d_squared);
            weight_sum += weight;
            weighted_sum += y[i] * weight;
        }
        
        // Return NaN if no valid weights
        if (weight_sum == 0.0) return std::numeric_limits<double>::quiet_NaN();
        
        return weighted_sum / weight_sum;
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
        
        size_t batch_size = buf.shape[0];
        size_t input_features = buf.shape[1];
        
        if (input_features != n_features) {
            throw std::runtime_error("Input features don't match training features");
        }
        
        double* X_input_ptr = static_cast<double*>(buf.ptr);
        
        auto result = py::array_t<double>(batch_size);
        double* result_ptr = static_cast<double*>(result.request().ptr);
        
        // Pre-compute pointers for better performance
        const double* X_train_ptr = X.data();
        const double* y_ptr = y.data();
        const double* Sinv_ptr = Sinv.data();
        
#ifdef _OPENMP
        // Set number of threads if specified
        int original_threads = omp_get_max_threads();
        if (n_threads > 0) {
            omp_set_num_threads(n_threads);
        }
        
        // Parallel processing over batch with optimized memory access
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < batch_size; i++) {
            const double* query_ptr = X_input_ptr + i * n_features;
            
            // Check if primary axis has valid value
            if (!std::isfinite(query_ptr[primary_axis])) {
                result_ptr[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            
            size_t start_idx = 0;
            size_t end_idx = n_samples;
            
            // Use cutoff optimization if applicable
            if (has_cutoff) {
                double primary_val = query_ptr[primary_axis];
                
                // Binary search for cutoff range
                size_t left = 0, right = n_samples;
                while (left < right) {
                    size_t mid = left + (right - left) / 2;
                    if (X_train_ptr[mid * n_features + primary_axis] < primary_val - delta_t) {
                        left = mid + 1;
                    } else {
                        right = mid;
                    }
                }
                start_idx = left;
                
                left = start_idx;
                right = n_samples;
                while (left < right) {
                    size_t mid = left + (right - left) / 2;
                    if (X_train_ptr[mid * n_features + primary_axis] < primary_val + delta_t) {
                        left = mid + 1;
                    } else {
                        right = mid;
                    }
                }
                end_idx = left;

                if (start_idx >= end_idx) {
                    result_ptr[i] = std::numeric_limits<double>::quiet_NaN();
                    continue;
                }
            }
            
            // Optimized weighted sum calculation
            double weighted_sum = 0.0;
            double weight_sum = 0.0;
            
            for (size_t s = start_idx; s < end_idx; s++) {
                const double* sample_ptr = X_train_ptr + s * n_features;
                
                // Prefetch next cache line for better memory access pattern
                if (LIKELY(s + 1 < end_idx)) {
#if defined(__GNUC__) || defined(__clang__)
                    __builtin_prefetch(X_train_ptr + (s + 1) * n_features, 0, 1);
#endif
                }
                
                // Use SIMD-optimized distance calculation
                double d_squared = calculate_distance_simd(sample_ptr, query_ptr, Sinv_ptr, n_features);
                
                double weight = std::exp(-0.5 * d_squared);
                weight_sum += weight;
                weighted_sum += y_ptr[s] * weight;
            }
            
            if (weight_sum == 0.0) {
                result_ptr[i] = std::numeric_limits<double>::quiet_NaN();
            } else {
                result_ptr[i] = weighted_sum / weight_sum;
            }
        }
        
        // Restore original thread count
        if (n_threads > 0) {
            omp_set_num_threads(original_threads);
        }
#else
        // Serial fallback - use optimized single prediction
        for (size_t i = 0; i < batch_size; i++) {
            std::vector<double> x(n_features);
            const double* query_ptr = X_input_ptr + i * n_features;
            std::copy(query_ptr, query_ptr + n_features, x.data());
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
