# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 构建命令

```bash
# 基本 CPU 构建
cmake -B build
cmake --build build --config Release

# 启用 CUDA（NVIDIA GPU）
# 注意：必须指定 VS 2022 生成器，VS 18 Insiders 缺少 CUDA toolset 会报错
# CMake 配置阶段 CUDA 检测很慢（约 20-30 分钟），属正常现象
cmake -B build-cuda -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
# CUDA 内核编译也很慢（约 10-20 分钟），nvcc 编译期间无逐行输出，耐心等待
cmake --build build-cuda --config Release
# 产物在 build-cuda/bin/Release/，包含 ggml-cuda.dll + ggml-cpu.dll（两者协同工作）

# 启用 Vulkan（跨平台 GPU）
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release

# 使用预设构建（查看 CMakePresets.json 获取完整列表）
cmake --preset x64-windows-msvc-release
cmake --build build-x64-windows-msvc-release
```

构建产物输出到 `build/bin/`（CPU）或 `build-cuda/bin/Release/`（CUDA）目录。

### Windows CUDA 构建注意事项

- 本机环境：CUDA 13.1 + RTX 4070 (sm_89) + VS 2022 Community (MSVC 19.38)
- 必须使用 `-G "Visual Studio 17 2022"` 指定生成器，默认的 VS 18 Insiders 没有 CUDA toolset 集成
- CMake 配置阶段 CUDA 检测非常慢（约 20-30 分钟），这是正常的
- CUDA 内核编译阶段（ggml-cuda）也很慢（约 10-20 分钟），MSBuild 不会逐行输出 .cu 文件编译进度
- 编译完成后 `ggml-cuda.dll`（~29MB）和 `ggml-cpu.dll` 会同时存在，两者协同工作（CPU 作为 fallback）
- 推理时用 `-ngl 99` 将所有层 offload 到 GPU
- 只编译单个目标可以节省时间：`cmake --build build-cuda --config Release --target llama-cli -j`

## 推理使用方法

本机模型路径：`D:/models/gguf/`

### llama-cli（交互式对话）

```bash
# CUDA GPU 推理，交互式多轮对话（默认模式）
./build-cuda/bin/Release/llama-cli.exe \
  -m "D:/models/gguf/Qwen3.5-0.8B-Q5_K_S.gguf" \
  -ngl 99

# 单轮对话模式（-st），跑完一轮自动退出，适合快速测试
./build-cuda/bin/Release/llama-cli.exe \
  -m "D:/models/gguf/Qwen3.5-0.8B-Q5_K_S.gguf" \
  -p "Hello, who are you?" \
  -n 256 -ngl 99 -st

# 指定系统提示词
./build-cuda/bin/Release/llama-cli.exe \
  -m "D:/models/gguf/Qwen3.5-0.8B-Q5_K_S.gguf" \
  -sys "You are a helpful assistant. Always respond in Chinese." \
  -ngl 99
```

### 常用参数说明

| 参数 | 说明 |
|------|------|
| `-m <path>` | 模型文件路径（GGUF 格式） |
| `-ngl 99` | 将所有层 offload 到 GPU（数字越大越多层上 GPU） |
| `-n <N>` | 最大生成 token 数（默认 -1 无限） |
| `-p <prompt>` | 初始提示词 |
| `-sys <prompt>` | 系统提示词 |
| `-st` | 单轮对话模式，生成完毕自动退出 |
| `-c <N>` | 上下文窗口大小（默认从模型加载） |
| `-t <N>` | CPU 线程数 |
| `-fa` | Flash Attention（auto/on/off） |
| `--no-cnv` | 禁用对话模式（llama-cli 不支持，需用 llama-completion） |

### 注意事项

- Qwen3.5 等模型默认启用思考模式（thinking mode），输出会先显示 `[Start thinking]` 思考过程，再输出最终回答
- `-n 64` 对思考模型太短，思考过程就会用完 token，建议至少 `-n 256` 或更大
- Windows 终端中文可能乱码，建议用英文提示词测试，或在 Windows Terminal 中设置 UTF-8 编码
- 性能参考（RTX 4070 Laptop + Qwen3.5-0.8B-Q5_K_S）：Prompt ~330 t/s，Generation ~222 t/s

## 测试命令

```bash
# 运行所有测试
ctest --test-dir build

# 运行单个测试
ctest --test-dir build -R test-tokenizer-0

# 详细输出
ctest --test-dir build -V

# 本地完整 CI
bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

### 服务器集成测试

```bash
pip install -r tools/server/tests/requirements.txt
cd tools/server/tests
PORT=8080 LLAMA_SERVER_BIN_PATH=../../../build/bin/llama-server ./tests.sh
# 调试模式
DEBUG=1 ./tests.sh -s -v -x
# 包含慢速测试
SLOW_TESTS=1 ./tests.sh
```

## 代码风格

- 4 空格缩进，Unix 换行符（LF），K&R 括号风格
- `snake_case` 命名函数、变量和类型
- 命名优化最长公共前缀：`number_small` 而非 `small_number`
- 枚举使用大写加前缀：`LLAMA_VOCAB_TYPE_SPM`
- 指针和引用：`void * ptr`、`int & a`
- 公共 API 使用固定大小整数类型（`int32_t`、`uint64_t`）
- 用 `struct foo {}` 而非 `typedef struct foo {} foo`
- C++ 代码中省略可选的 `struct`/`enum` 关键字
- 避免第三方依赖、花哨的现代 STL 构造和模板，保持简单
- 使用 `clang-format`（v15+）格式化代码

## 项目架构

### 核心层次

```
include/          公共 C API 头文件（llama.h、llama-cpp.h）
src/              核心 llama 库（libllama）
  src/models/     112+ 模型架构实现
ggml/             GGML 张量计算库（子项目）
  ggml/src/ggml-cpu/    CPU 后端（含架构特定优化）
  ggml/src/ggml-cuda/   NVIDIA CUDA 后端
  ggml/src/ggml-metal/  Apple Metal 后端
  ggml/src/ggml-vulkan/ Vulkan 后端
  ggml/src/ggml-hip/    AMD HIP 后端
  （以及 sycl、opencl、webgpu、rpc、blas、cann、musa 等后端）
common/           共享工具库（参数解析、采样、聊天模板、GBNF 语法、Jinja 模板引擎等）
tools/            命令行工具
  tools/server/   HTTP 服务器（llama-server），含 SvelteKit WebUI
  tools/cli/      主 CLI 工具（llama-cli）
  tools/quantize/ 模型量化工具
  tools/mtmd/     多模态支持
examples/         示例程序
tests/            C++ 单元测试（CTest）
```

### 关键设计要点

- **张量存储**：行优先顺序。维度 0 = 列，1 = 行，2 = 矩阵
- **矩阵乘法**：`C = ggml_mul_mat(ctx, A, B)` 意味着 C^T = A B^T
- **后端系统**：可插拔架构，支持动态加载，CPU+GPU 混合推理
- **服务器架构**：`server_context` 单线程推理 + 多线程 HTTP 层。JSON 解析和聊天模板逻辑必须在 HTTP 层处理，避免在 `server_slot` 间传递原始 JSON

### 添加新模型

参考 [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md)。关键步骤：在 `src/llama-arch.h/cpp` 定义架构 → 在 `src/llama-model.cpp` 实现图构建 → 使用 `convert_hf_to_gguf.py` 转换模型。

## 重要注意事项

- 修改 `ggml` 算子后，必须运行 `test-backend-ops` 验证后端一致性
- 新功能应先聚焦 CPU 支持，GPU 后端在后续 PR 中添加
- 使用 `llama-perplexity` 和 `llama-bench` 验证变更不会影响质量和性能
- 第三方依赖位于 `vendor/`（cpp-httplib、nlohmann/json、miniaudio、stb、subprocess.h）
