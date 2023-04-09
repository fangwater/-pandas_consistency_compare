#include <xsimd/xsimd.hpp>
#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>
#include <bits/stdc++.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>
#include <execution>
#include <tbb/parallel_invoke.h>
#include <omp.h>
#include <limits>

template<typename T>
std::tuple<T, T, T, T> min_max_start_end_aligned(const std::unique_ptr<std::vector<T,xsimd::aligned_allocator<T>>>& input_ptr) {
    if (!input_ptr->size()) {
        throw std::invalid_argument("Input vector must not be empty.");
    }
    using simd_type = xsimd::simd_type<T>;
    std::size_t simd_size = simd_type::size;
    simd_type min_val = input_ptr->at(0), max_val = input_ptr->at(0);
    std::size_t i = 0;
    for (; i + simd_size <= input_ptr->size(); i += simd_size) {
        simd_type chunk = xsimd::load_aligned(&input_ptr->at(i));

        min_val = xsimd::min(min_val, chunk);
        max_val = xsimd::max(max_val, chunk);
    }
    T min_value = xsimd::reduce_min(min_val);
    T max_value = xsimd::reduce_max(max_val);

    for (; i < input_ptr->size(); ++i) {
        min_value = std::min(min_value, input_ptr->at(i));
        max_value = std::max(max_value, input_ptr->at(i));
    }
    return std::make_tuple(min_value, max_value, input_ptr->front(), input_ptr->back());
}

template<typename T>
std::tuple<T, T, T, T> min_max_start_end(const std::unique_ptr<std::vector<T>>& input_ptr) {
    if (!input_ptr->size()) {
        throw std::invalid_argument("Input vector must not be empty.");
    }
    using simd_type = xsimd::simd_type<T>;
    std::size_t simd_size = simd_type::size;
    simd_type min_val = input_ptr->at(0), max_val = input_ptr->at(0);
    std::size_t i = 0;
    for (; i + simd_size <= input_ptr->size(); i += simd_size) {
        simd_type chunk = xsimd::load_aligned(&input_ptr->at(i));

        min_val = xsimd::min(min_val, chunk);
        max_val = xsimd::max(max_val, chunk);
    }
    T min_value = xsimd::reduce_min(min_val);
    T max_value = xsimd::reduce_max(max_val);

    for (; i < input_ptr->size(); ++i) {
        min_value = std::min(min_value, input_ptr->at(i));
        max_value = std::max(max_value, input_ptr->at(i));
    }
    return std::make_tuple(min_value, max_value, input_ptr->front(), input_ptr->back());
}




template<typename T>
T sum_aligned(const std::unique_ptr<std::vector<T,xsimd::aligned_allocator<T>>>& input_ptr) {
    if (!input_ptr->size()) {
        return static_cast<T>(0);
    }
    using simd_type = xsimd::simd_type<T>;
    std::size_t simd_size = simd_type::size;
    simd_type sum_val = 0;
    std::size_t i = 0;
    for (; i + simd_size <= input_ptr->size(); i += simd_size) {
        simd_type chunk = xsimd::load_aligned(&input_ptr->at(i));
        sum_val = sum_val + chunk;
    }
    T sum_value = xsimd::reduce_add(sum_val);

    for (; i < input_ptr->size(); ++i) {
        sum_value += input_ptr->at(i);
    }
    return sum_value;
}

template<typename T>
T sum_normal(const std::unique_ptr<std::vector<T>>& input_ptr) {
    if (!input_ptr->size()) {
        return static_cast<T>(0);
    }
    using simd_type = xsimd::simd_type<T>;
    std::size_t simd_size = simd_type::size;
    simd_type sum_val = 0;
    std::size_t i = 0;
    for (; i + simd_size <= input_ptr->size(); i += simd_size) {
        simd_type chunk = xsimd::load_unaligned(&input_ptr->at(i));
        sum_val = sum_val + chunk;
    }
    T sum_value = xsimd::reduce_add(sum_val);

    for (; i < input_ptr->size(); ++i) {
        sum_value += input_ptr->at(i);
    }
    return sum_value;
}


template<typename T>
std::unique_ptr<std::vector<T, xsimd::aligned_allocator<T>>> hadamard_product_aligned(const std::unique_ptr<std::vector<T, xsimd::aligned_allocator<T>>>& a_ptr, const std::unique_ptr<std::vector<T, xsimd::aligned_allocator<T>>>& b_ptr) {
    if (a_ptr->size() != b_ptr->size()) {
        throw std::invalid_argument("Input vectors must have the same size.");
    }

    using simd_type = xsimd::simd_type<T>;
    std::size_t simd_size = simd_type::size;

    std::unique_ptr<std::vector<T, xsimd::aligned_allocator<T>>> res_ptr = std::make_unique<std::vector<T, xsimd::aligned_allocator<T>>>(a_ptr->size());
    std::size_t i = 0;
    for (; i + simd_size <= a_ptr->size(); i += simd_size) {
        simd_type chunk_a = xsimd::load_aligned(&a_ptr->at(i));
        simd_type chunk_b = xsimd::load_aligned(&b_ptr->at(i));
        simd_type product_chunk = chunk_a * chunk_b;
        xsimd::store_aligned(&res_ptr->at(i), product_chunk);
    }

    // 对于未能对齐到SIMD宽度的剩余部分，我们使用普通循环处理
    for (; i < a_ptr->size(); ++i) {
        res_ptr->at(i) = a_ptr->at(i)* b_ptr->at(i);
    }
    return res_ptr;
}

template<typename T>
std::unique_ptr<std::vector<T>> hadamard_product_normal(const std::unique_ptr<std::vector<T>>& a_ptr, const std::unique_ptr<std::vector<T>>& b_ptr) {
    if (a_ptr->size() != b_ptr->size()) {
        throw std::invalid_argument("Input vectors must have the same size.");
    }

    using simd_type = xsimd::simd_type<T>;
    std::size_t simd_size = simd_type::size;

    std::unique_ptr<std::vector<T>> res_ptr = std::make_unique<std::vector<T>>(a_ptr->size());
    std::size_t i = 0;
    for (; i + simd_size <= a_ptr->size(); i += simd_size) {
        simd_type chunk_a = xsimd::load_unaligned(&a_ptr->at(i));
        simd_type chunk_b = xsimd::load_unaligned(&b_ptr->at(i));
        simd_type product_chunk = chunk_a * chunk_b;
        xsimd::store_aligned(&res_ptr->at(i), product_chunk);
    }

    // 对于未能对齐到SIMD宽度的剩余部分，我们使用普通循环处理
    for (; i < a_ptr->size(); ++i) {
        res_ptr->at(i) = a_ptr->at(i)* b_ptr->at(i);
    }
    return res_ptr;
}



template<typename T>
using data_ptr = std::unique_ptr<std::vector<T>>; 

struct ticker{
    double tradv;
    double tradp;
    int32_t B_or_S;
    ticker(double v,double p,int32_t flag):tradv(v),tradp(p),B_or_S(flag){};
};

int main() {
    int test_size = 10;
    std::vector<std::shared_ptr<ticker>> tmp;
    std::cout << "test size: "<< test_size << std::endl;
    for(int i = 0; i < test_size; i++){
        std::shared_ptr<ticker> sp = std::make_shared<ticker>(ticker(0.1*i,i,1));
        tmp.push_back(sp);
    }
    std::shared_ptr<std::vector<std::shared_ptr<ticker>>> ticker_1min_buffer_sp = std::make_shared<std::vector<std::shared_ptr<ticker>>>(std::move(tmp));

    auto aa_tradp = std::make_unique<std::vector<double,xsimd::aligned_allocator<double>>>(ticker_1min_buffer_sp->size());
    auto aa_S_tradp = std::make_unique<std::vector<double,xsimd::aligned_allocator<double>>>();
    auto aa_B_tradp = std::make_unique<std::vector<double,xsimd::aligned_allocator<double>>>();
    

    auto aa_tradv = std::make_unique<std::vector<double,xsimd::aligned_allocator<double>>>(ticker_1min_buffer_sp->size());
    auto aa_S_tradv = std::make_unique<std::vector<double,xsimd::aligned_allocator<double>>>();
    auto aa_B_tradv = std::make_unique<std::vector<double,xsimd::aligned_allocator<double>>>();
    
    for(int i = 0; i < test_size; i++){
        std::shared_ptr<ticker> ticker_sp = ticker_1min_buffer_sp->at(i);
        if(ticker_sp->tradp > 0){
            aa_tradp->at(i) = ticker_sp->tradp;
            aa_tradv->at(i) = ticker_sp->tradv;
            if( ticker_sp->B_or_S == 1 ){
                aa_B_tradp->emplace_back(ticker_sp->tradp);
                aa_B_tradv->emplace_back(ticker_sp->tradv);
            }else{
                aa_S_tradp->emplace_back(ticker_sp->tradp);
                aa_S_tradv->emplace_back(ticker_sp->tradv);  
            }   
        }
    }
    //cjbs
    uint64_t cjb = aa_tradp->size();
    //bcjbs
    uint64_t bcjb = aa_B_tradp->size();
    //scjbs
    uint64_t scjb = aa_S_tradp->size();

    auto aa_tradmt = hadamard_product_aligned<double>(aa_tradp,aa_tradv);
    auto aa_B_tradmt = hadamard_product_aligned<double>(aa_B_tradp,aa_B_tradv);
    auto aa_S_tradmt = hadamard_product_aligned<double>(aa_S_tradp,aa_S_tradv);
    
    //volume_nfq
    auto volume_nfq = sum_aligned<double>(aa_tradv);
    //bvolume_nfq
    auto bvolume_nfq = sum_aligned<double>(aa_B_tradv);
    //svolume_nfq
    auto svolume_nfq = sum_aligned<double>(aa_S_tradv);

    //amount
    auto amount = sum_aligned<double>(aa_tradmt);
    //bamount
    auto bamount = sum_aligned<double>(aa_B_tradmt);
    //samount
    auto samount = sum_aligned<double>(aa_S_tradmt);

    auto [lowprice_nfq, highprice_nfq, openprice_nfq, closeprice_nfq] = min_max_start_end_aligned<double>(aa_tradp);
    

    return 0;
}
