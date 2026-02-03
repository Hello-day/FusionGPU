# 1. 设置 CMake 最低版本
cmake_minimum_required(VERSION 3.21)
project(FusionANNS LANGUAGES CXX CUDA)

# =========================================================
# 设置目标 GPU 架构 (A100 = sm_80)
# =========================================================
set(CMAKE_CUDA_ARCHITECTURES "80")


# 2. 设置语言标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# 3. 设置 NVCC 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler -fopenmp")

# 4. 查找依赖库
find_package(OpenMP REQUIRED)

# 查找 cuVS 和 RMM (cuVS 会自动引入 raft 核心依赖)
find_package(cuvs REQUIRED)
find_package(rmm REQUIRED)

# [关键修复] 查找 CUDA 核心数学库
# RAFT 依赖这些库，但在某些环境下需要显式链接
find_package(CUDAToolkit REQUIRED)

# 5. 头文件路径
include_directories(src)

# --- 目标 1: 'build_global' (构建索引) ---
add_executable(build_global src/build_global.cu)

target_link_libraries(build_global PUBLIC
    OpenMP::OpenMP_CXX
    cuvs::cuvs
    rmm::rmm
    # [关键修复] 显式链接 CUDA 数学库
    CUDA::cublas
    CUDA::cusolver
    CUDA::cusparse 
    CUDA::curand
)

# --- 目标 2: 'search_pipeline' (搜索流水线) ---
add_executable(search_pipeline 
    src/search_pipeline.cu
    src/gpu_global.cu
)

target_link_libraries(search_pipeline PUBLIC
    OpenMP::OpenMP_CXX
    cuvs::cuvs
    rmm::rmm
    # [关键修复] 同样需要链接
    CUDA::cublas
    CUDA::cusolver
    CUDA::cusparse
    CUDA::curand
)

# --- 安装 ---
install(TARGETS build_global search_pipeline DESTINATION bin)

