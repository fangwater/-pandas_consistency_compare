#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>
#include <execution>
#include <tbb/parallel_invoke.h>
#include <xsimd/xsimd.hpp>
#include <omp.h>
#include <limits>

std::tuple<double, double, double, double, double> min_max_start_end_sum_xsimd(const std::vector<double>& input) {
    if (input.empty()) {
        throw std::invalid_argument("Input vector must not be empty.");
    }

    using simd_type = xsimd::simd_type<double>;
    std::size_t simd_size = simd_type::size;
    std::cout << simd_size << std::endl;

    simd_type min_val = input[0], max_val = input[0], sum_val = 0;

    std::size_t i = 0;

    for (; i + simd_size <= input.size(); i += simd_size) {
        simd_type chunk = xsimd::load_unaligned(&input[i]);

        min_val = xsimd::min(min_val, chunk);
        max_val = xsimd::max(max_val, chunk);
        sum_val = sum_val + chunk;
    }
    double min_value = xsimd::reduce_min(min_val);
    double max_value = xsimd::reduce_max(max_val);
    double sum_value = xsimd::reduce_add(sum_val);

    // 对于未能对齐到SIMD宽度的剩余部分，我们使用普通循环处理
    for (; i < input.size(); ++i) {
        min_value = std::min(min_value, input[i]);
        max_value = std::max(max_value, input[i]);
        sum_value += input[i];
    }
    return std::make_tuple(min_value, max_value, input.front(), input.back(), sum_value);
}


std::tuple<double, double, double, double, double> min_max_start_end_sum_xsimd2(const  std::vector<double, xsimd::aligned_allocator<double>>& input) {
    if (input.empty()) {
        throw std::invalid_argument("Input vector must not be empty.");
    }
    using simd_type = xsimd::simd_type<double>;
    std::size_t simd_size = simd_type::size;
    std::cout << simd_size << std::endl;

    simd_type min_val = input[0], max_val = input[0], sum_val = 0;

    std::size_t i = 0;

    for (; i + simd_size <= input.size(); i += simd_size) {
        simd_type chunk = xsimd::load_aligned(&input[i]);

        min_val = xsimd::min(min_val, chunk);
        max_val = xsimd::max(max_val, chunk);
        sum_val = sum_val + chunk;
    }
    double min_value = xsimd::reduce_min(min_val);
    double max_value = xsimd::reduce_max(max_val);
    double sum_value = xsimd::reduce_add(sum_val);

    // 对于未能对齐到SIMD宽度的剩余部分，我们使用普通循环处理
    for (; i < input.size(); ++i) {
        min_value = std::min(min_value, input[i]);
        max_value = std::max(max_value, input[i]);
        sum_value += input[i];
    }
    return std::make_tuple(min_value, max_value, input.front(), input.back(), sum_value);
}



std::tuple<double, double, double, double, double> min_max_start_end_sum_omp(const std::vector<double>& input) {
    if (input.empty()) {
        throw std::invalid_argument("Input vector must not be empty.");
    }

    double max_val = std::numeric_limits<double>::lowest();
    double min_val = std::numeric_limits<double>::max();
    double sum_val = 0.0;

    #pragma omp simd reduction(max: max_val) reduction(min: min_val) reduction(+: sum_val)
    for (size_t i = 0; i < input.size(); i++) {
        max_val = max_val > input[i] ? max_val : input[i];
        min_val = min_val < input[i] ? min_val : input[i];
        sum_val += input[i];
    }
    return std::make_tuple(min_val, max_val, input.front(), input.back(), sum_val);
}

std::tuple<double, double, double, double, double> min_max_start_end_sum_stl(const std::vector<double>& input) {
    if (input.empty()) {
        throw std::invalid_argument("Input vector must not be empty.");
    }

    auto min_max = std::minmax_element(input.begin(), input.end());
    double sum = std::accumulate(input.begin(), input.end(), 0.0);
    return std::make_tuple(*min_max.first, *min_max.second, input.front(), input.back(), sum);
}

std::tuple<double, double, double, double, double> min_max_start_end_sum_pstl(const std::vector<double>& input) {
    if (input.empty()) {
        throw std::invalid_argument("Input vector must not be empty.");
    }

    auto min_max = std::minmax_element(std::execution::unseq, input.begin(), input.end());
    double sum = std::reduce(std::execution::unseq, input.begin(), input.end(), 0.0);
    return std::make_tuple(*min_max.first, *min_max.second, input.front(), input.back(), sum);
}

template <std::size_t I = 0>
constexpr bool compare_tuples_helper(const std::tuple<double, double, double, double, double>& t1,
                                     const std::tuple<double, double, double, double, double>& t2) {
    if constexpr (I < 5) {
        const double epsilon = 1e-6;
        if (std::abs(std::get<I>(t1) - std::get<I>(t2)) > epsilon) {
            return false;
        }
        return compare_tuples_helper<I + 1>(t1, t2);
    }
    return true;
}

bool compare_tuples(const std::tuple<double, double, double, double, double>& t1,
                    const std::tuple<double, double, double, double, double>& t2) {
    return compare_tuples_helper(t1, t2);
}

int main() {
    std::vector<double> data(500000);
    for(int i = 0; i < data.size(); i++){
        data[i] = i;
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto [min_val_eve, max_val_eve, start_val_eve, end_val_eve, sum_val_eve] = min_max_start_end_sum_xsimd(data);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_simd = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::vector<double, xsimd::aligned_allocator<double>> input(data.begin(), data.end());
    start = std::chrono::high_resolution_clock::now();
    auto [min_val_xsimd2, max_val_xsimd2, start_val_xsimd2, end_val_xsimd2, sum_val_xsimd2] = min_max_start_end_sum_stl(data);
    end = std::chrono::high_resolution_clock::now();
    auto duration_simd2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    auto [min_val_stl, max_val_stl, start_val_stl, end_val_stl, sum_val_stl] = min_max_start_end_sum_stl(data);
    end = std::chrono::high_resolution_clock::now();
    auto duration_stl = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    auto [min_val_omp, max_val_omp, start_val_omp, end_val_omp, sum_val_omp] = min_max_start_end_sum_omp(data);
    end = std::chrono::high_resolution_clock::now();
    auto duration_omp = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    auto [min_val_pstl, max_val_pstl, start_val_pstl, end_val_pstl, sum_val_pstl] = min_max_start_end_sum_pstl(data);
    end = std::chrono::high_resolution_clock::now();
    auto duration_pstl = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    std::cout << "xsimd 耗时: " << duration_simd << " ns" << std::endl;
    std::cout << "xsimd 对齐耗时: " << duration_simd2 << " ns" << std::endl;
    std::cout << "stl 耗时: " << duration_stl << " ns" << std::endl;
    std::cout << "omp 耗时: " << duration_omp << " ns" << std::endl;
    std::cout << "pstl 耗时: " << duration_pstl << " ns" << std::endl;

    bool compare_eve_stl = compare_tuples(std::make_tuple(min_val_eve, max_val_eve, start_val_eve, end_val_eve, sum_val_eve),
                                          std::make_tuple(min_val_stl, max_val_stl, start_val_stl, end_val_stl, sum_val_stl));

    bool compare_eve_pstl = compare_tuples(std::make_tuple(min_val_eve, max_val_eve, start_val_eve, end_val_eve, sum_val_eve),
                                           std::make_tuple(min_val_pstl, max_val_pstl, start_val_pstl, end_val_pstl, sum_val_pstl));

    bool compare_stl_pstl = compare_tuples(std::make_tuple(min_val_stl, max_val_stl, start_val_stl, end_val_stl, sum_val_stl),
                                           std::make_tuple(min_val_pstl, max_val_pstl, start_val_pstl, end_val_pstl, sum_val_pstl));

    std::cout << "EVE 与 STL 结果一致: " << (compare_eve_stl ? "是" : "否") << std::endl;
    std::cout << "EVE 与 PSTL 结果一致: " << (compare_eve_pstl ? "是" : "否") << std::endl;
    std::cout << "STL 与 PSTL 结果一致: " << (compare_stl_pstl ? "是" : "否") << std::endl;

}
