# ------------------------------------------------------------------------------
# 1. 编译器设置
# ------------------------------------------------------------------------------
# 尝试从环境变量获取 CUDA_HOME，如果没有则默认使用 /usr/local/cuda
CUDA_HOME ?= /usr/local/cuda

# 使用系统安装的 nvcc
NVCC     := $(CUDA_HOME)/bin/nvcc
CXX      := g++

# GPU 架构 (请根据实际情况修改: V100=sm_70, A100=sm_80, RTX3090=sm_86)
CUDA_ARCH := sm_80

# ------------------------------------------------------------------------------
# 2. 路径配置 (混合环境的关键)
# ------------------------------------------------------------------------------
# [头文件路径]
# 1. $CONDA_PREFIX/include: 包含 RAFT, libcudacxx (cuda/stream_ref), thrust 等
# 2. $CUDA_HOME/include: 包含系统 CUDA 标准头文件
# 注意：CONDA_PREFIX 必须放在前面，以确保优先使用新版的 CCCL/RAFT 头文件
INC_PATHS := -I$(CONDA_PREFIX)/include -I$(CUDA_HOME)/include

# [库文件路径]
# 1. $CUDA_HOME/lib64: 系统 CUDA 运行时库 (libcudart.so)
# 2. $CONDA_PREFIX/lib: RAFT 库 (libraft.so)
LIB_PATHS := -L$(CUDA_HOME)/lib64 -L$(CONDA_PREFIX)/lib

# ------------------------------------------------------------------------------
# 3. 编译与链接选项
# ------------------------------------------------------------------------------
# RAFT 需要的特殊标志
RAFT_FLAGS := --expt-relaxed-constexpr --expt-extended-lambda -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

# Host (CPU) 编译器标志
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -fopenmp -g $(INC_PATHS)

# Device (GPU) 编译器标志
# -ccbin: 指定 host 编译器
NVCCFLAGS := -std=c++17 -O3 -g -arch=$(CUDA_ARCH) \
             -ccbin $(CXX) \
             -Xcompiler "-fopenmp -Wall -Wextra" \
             $(RAFT_FLAGS) $(INC_PATHS)

# 链接标志
LDFLAGS  := -Xcompiler -fopenmp $(LIB_PATHS)
LDLIBS   := -lcudart -lraft

# ------------------------------------------------------------------------------
# 4. 构建目标
# ------------------------------------------------------------------------------
HEADERS  := defs.h utils.h pq_utils.h cagra_adapter.cuh kmeans_gpu.cuh

.PHONY: all clean

all: build_global tune_parameters

# 索引构建工具
build_global: build_global.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# 参数调优工具
tune_parameters: tune_parameters.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f build_global tune_parameters *.o