#ifndef UTILS_H_
#define UTILS_H_

#include "defs.h"
#include <fstream>
#include <string>
#include <cassert>
#include <iostream>
#include <vector>

// 计算欧氏距离平方
inline float dis(const point_t<float>& a, const point_t<float>& b) {
    float dist = 0.f;
    for (int i = 0; i < DIM; ++i) {
        float diff = a.coordinates[i] - b.coordinates[i];
        dist += diff * diff;
    }
    return dist;
}

// 读取 fvecs 格式
template<typename T>
int read_vecs(const std::string& file, T*& vecs, int d) {
    std::ifstream ifs(file, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Error opening file: " << file << std::endl;
        exit(1);
    }
    int dim;
    ifs.read((char*)&dim, sizeof(int));
    if(dim != d) {
        std::cerr << "Dimension mismatch: expected " << d << ", got " << dim << std::endl;
        exit(1);
    }
    
    ifs.seekg(0, std::ios::end);
    size_t fsize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    
    size_t n = fsize / (sizeof(int) + dim * sizeof(T));
    vecs = new T[n * d];
    
    for (size_t i = 0; i < n; i++) {
        ifs.ignore(sizeof(int));
        ifs.read((char*)&vecs[i * d], dim * sizeof(T));
    }
    return (int)n;
}

// 写入 fvecs 格式
template<typename T>
void write_vecs(const std::string& file, const T* vecs, int n, int dim) {
    std::ofstream ofs(file, std::ios::binary);
    for (int i = 0; i < n; i++) {
        ofs.write((char*)&dim, sizeof(int));
        ofs.write((char*)&vecs[i * dim], dim * sizeof(T));
    }
}

// 读取二进制文件到 vector
template<typename T>
void read_binary_vector(const std::string& file, std::vector<T>& vec) {
    std::ifstream ifs(file, std::ios::binary | std::ios::ate);
    if(!ifs) {
        std::cerr << "Error opening binary file: " << file << std::endl;
        exit(1);
    }
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    vec.resize(size / sizeof(T));
    ifs.read((char*)vec.data(), size);
}

// 读取 ivecs 格式 (ground truth)
inline int read_ivecs(const std::string& file, int*& vecs, int& d) {
    std::ifstream ifs(file, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Error opening file: " << file << std::endl;
        exit(1);
    }
    
    ifs.read((char*)&d, sizeof(int));
    ifs.seekg(0, std::ios::end);
    size_t fsize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    
    size_t n = fsize / (sizeof(int) + d * sizeof(int));
    vecs = new int[n * d];
    
    for (size_t i = 0; i < n; i++) {
        ifs.ignore(sizeof(int));
        ifs.read((char*)&vecs[i * d], d * sizeof(int));
    }
    return (int)n;
}

#endif // UTILS_H_