# FusionGPU

基于 CUDA + RAFT 的 GPU 加速向量检索框架，包含参数调优与相关 GPU 计算代码。

## 目录结构

- `tune_parameters.cu`：参数调优入口
- `build_global.cu` / `gpu_global.cu` / `search_pipeline.cu`：GPU 计算逻辑
- `defs.h` / `utils.h` / `pq_utils.h` / `cagra_adapter.cuh`：公共头文件
- `Makefile`：构建脚本

## 依赖与环境

- CUDA Toolkit（需要 `nvcc`）
- RAFT 库（通过 `CONDA_PREFIX` 提供）
- C++17 编译器（如 `g++`）

关键环境变量：

- `CUDA_HOME`：CUDA 安装路径，默认 `/usr/local/cuda`
- `CONDA_PREFIX`：Conda 环境路径（包含 RAFT 头文件与库）
- `CUDA_ARCH`：GPU 架构（如 `sm_80`）

## 构建

```bash
make
```

如需指定 GPU 架构：

```bash
make CUDA_ARCH=sm_86
```

## 运行

构建完成后执行：

```bash
./tune_parameters
```

## 清理

```bash
make clean
```
