# Qwen3.5-0.8B Q5_K_S 量化指南 — 复刻 Unsloth 官方效果

## 项目介绍与分工

### llama.cpp (`C:\Projects\llama.cpp`)

GGML 张量计算库 + LLM 推理引擎。提供两个核心工具：
- `convert_hf_to_gguf.py`：将 HuggingFace 模型转换为 GGUF 格式
- `llama-quantize`：对 GGUF 模型执行各种量化（Q4_K_S, Q5_K_S, Q8_0 等）
- `llama-cli`：加载 GGUF 模型进行推理

### unsloth (`C:\Projects\unsloth`)

Unsloth 主包，提供高层 API（如 `FastLanguageModel`）。
调用 `unsloth-zoo` 完成模型保存和 GGUF 导出，本身不直接参与量化。

### unsloth-zoo (`C:\Projects\unsloth-zoo`)

Unsloth 的底层工具库，核心文件是 `unsloth_zoo/llama_cpp.py`。负责：
1. 自动克隆并编译 llama.cpp
2. 下载 `convert_hf_to_gguf.py` 并打 3 个 patch（gguf 属性兼容、metadata 品牌、Qwen MoE num_experts）
3. 调用 `convert_hf_to_gguf.py` 执行 HF → BF16 GGUF 转换
4. 调用 `llama-quantize` 执行最终量化（如 Q5_K_S）

### 三者关系

```
用户代码 → unsloth (高层API) → unsloth-zoo (底层工具) → llama.cpp (实际执行)
```

## 关键发现

通过对比 Unsloth 官方发布的 `Qwen3.5-0.8B-Q5_K_S.gguf` 与源码，发现以下事实：

1. **Unsloth 使用了 imatrix 量化**，但开源代码中没有 imatrix 逻辑。
   imatrix 文件 `imatrix_unsloth.gguf` 随模型一起发布在 HuggingFace 仓库中。
   校准数据集为 `unsloth_calibration_Qwen3.5-0.8B.txt`。

2. **SSM 层采用更保守的量化策略**（非默认行为）：
   - `ssm_alpha.weight` / `ssm_beta.weight` → Q8_0（默认会被量化为 Q5_K）
   - `ssm_out.weight` → Q6_K（默认会被量化为 Q5_K）
   需要通过 `--tensor-type` 手动覆盖才能复刻。

3. **Unsloth 的 3 个 patch 对量化数值无影响**：
   - Patch 1（gguf 属性兼容）：防止旧版 gguf-py 缺少新属性报错
   - Patch 2（metadata 品牌）：写入 `quantized_by=Unsloth`、`repo_url` 等元数据
   - Patch 3（Qwen MoE num_experts）：仅影响 MoE 模型，对 Qwen3.5-0.8B 不生效

## 前置条件

- 本地已编译 llama.cpp（CUDA build 在 `build-cuda/bin/Release/`）
- HuggingFace 模型：`D:/models/hf/Qwen/Qwen3.5_0.8b`
- 已安装 `uv`（Python 包管理器）
- 工作目录：`c:/Projects/llama.cpp`

## 量化过程

### Step 1: 编译 llama.cpp

```bash
# CUDA build（已有 build-cuda 目录则跳过 cmake 配置）
cmake -B build-cuda -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release

# 只编译需要的目标，节省时间
cmake --build build-cuda --config Release --target llama-quantize llama-cli -j
```

### Step 2: HF → GGUF BF16 转换

```bash
cd c:/Projects/llama.cpp

uv run --no-project \
  --extra-index-url "https://download.pytorch.org/whl/cpu" \
  --with numpy --with sentencepiece --with "transformers>=4.35.2" \
  --with protobuf --with torch --with "./gguf-py" \
  python convert_hf_to_gguf.py "D:/models/hf/Qwen/Qwen3.5_0.8b" \
  --outfile "D:/models/gguf/Qwen3.5-0.8B-BF16.gguf" \
  --outtype bf16
```

输出 320 个 tensor，约 1.51GB。Qwen3.5 是 VLM 模型，但这里只转换 text 部分（不加 `--mmproj`）。

### Step 3: 下载 Unsloth imatrix 文件

```bash
curl -L "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/imatrix_unsloth.gguf_file" \
  -o "D:/models/gguf/imatrix_unsloth.gguf"
```

约 1.1MB。这是 Unsloth 用 `unsloth_calibration_Qwen3.5-0.8B.txt` 校准数据集生成的重要性矩阵。

### Step 4: BF16 → Q5_K_S 量化（带 imatrix + tensor-type 覆盖）

```bash
./build-cuda/bin/Release/llama-quantize.exe \
  --imatrix "D:/models/gguf/imatrix_unsloth.gguf" \
  --tensor-type "blk\..*\.ssm_alpha\.weight=q8_0" \
  --tensor-type "blk\..*\.ssm_beta\.weight=q8_0" \
  --tensor-type "blk\..*\.ssm_out\.weight=q6_k" \
  "D:/models/gguf/Qwen3.5-0.8B-BF16.gguf" \
  "D:/models/gguf/Qwen3.5-0.8B-Q5_K_S-self.gguf" \
  Q5_K_S
```

输出 532 MiB (5.93 BPW)，约 568MB。三个 `--tensor-type` 参数是复刻的关键，
确保 SSM 层使用与 Unsloth 官方一致的更高精度量化。

### Step 5: 清理中间文件

```bash
rm "D:/models/gguf/Qwen3.5-0.8B-BF16.gguf"
rm "D:/models/gguf/imatrix_unsloth.gguf"
```

## 验证

### 文件大小对比

| 文件 | 大小 (bytes) |
|------|-------------|
| 自量化 `Qwen3.5-0.8B-Q5_K_S-self.gguf` | 568,889,440 |
| 官方 `Qwen3.5-0.8B-Q5_K_S.gguf` | 568,889,600 |
| 差异 | 160 bytes（仅元数据字段） |

160 bytes 差异来自官方额外的元数据字段（`general.quantized_by`、`general.basename`、`general.repo_url` 等），
不影响模型权重。

### Tensor 类型对比

320/320 tensor 的量化类型和形状完全匹配（使用 gguf-py 的 GGUFReader 逐一比对）。

### 元数据差异

仅存在品牌/来源相关的元数据差异，不影响模型行为：

| 字段 | 自量化 | 官方 |
|------|--------|------|
| `general.name` | `Qwen3.5_0.8b` | `Qwen3.5-0.8B` |
| `general.size_label` | `752M` | `0.8B` |
| `general.quantized_by` | （无） | `Unsloth` |
| `general.basename` | （无） | `Qwen3.5-0.8B` |
| `general.repo_url` | （无） | `https://huggingface.co/unsloth` |
| `quantize.imatrix.*` | （无） | 有 imatrix 元数据 |

### 推理测试

```bash
# 自量化模型
./build-cuda/bin/Release/llama-cli.exe \
  -m "D:/models/gguf/Qwen3.5-0.8B-Q5_K_S-self.gguf" \
  -p "Hello, how are you?" -n 50 -ngl 99 -st --no-display-prompt -s 42

# 官方模型
./build-cuda/bin/Release/llama-cli.exe \
  -m "D:/models/gguf/Qwen3.5-0.8B-Q5_K_S.gguf" \
  -p "Hello, how are you?" -n 50 -ngl 99 -st --no-display-prompt -s 42
```

两个模型推理行为一致：都进入 thinking mode，思考结构相同，内存占用完全一致（532 MiB model），
性能在同一水平（RTX 4070 Laptop: Prompt ~330 t/s, Generation ~230 t/s）。

---

## 对称量化实验：Q8_0 与 Q5_0

### 实验目的

原始 Q5_K_S 量化中，SSM 层使用了非对称的 Q6_K（ssm_out）和对称的 Q8_0（ssm_alpha/beta）。
本实验将所有量化类型替换为纯对称量化（以 0 为中心，无偏移量），观察输出质量是否下降。

- **Q8_0**：8-bit 对称量化，公式 `x = d * q`，精度最高
- **Q5_0**：5-bit 对称量化，公式 `x = d * q`，与 Q5_K_S 同为 5-bit 但无 K-quant 的子块 scale/min 优化
- **Q5_K_S**（基线）：5-bit K-quant，部分层使用非对称量化（Q4_K/Q5_K 带 `dmin`），SSM 层手动覆盖为 Q8_0/Q6_K

### 量化命令

```bash
# Q8_0：全对称，不需要 tensor-type 覆盖（Q8_0 本身就是对称的）
./build-cuda/bin/Release/llama-quantize.exe \
  --imatrix "D:/models/gguf/imatrix_unsloth.gguf" \
  "D:/models/gguf/Qwen3.5-0.8B-BF16.gguf" \
  "D:/models/gguf/Qwen3.5-0.8B-Q8_0-self-0.gguf" \
  Q8_0

# Q5_0：全对称，SSM alpha/beta 保持 Q8_0，SSM out 改为 Q5_0（替代原来的非对称 Q6_K）
./build-cuda/bin/Release/llama-quantize.exe \
  --imatrix "D:/models/gguf/imatrix_unsloth.gguf" \
  --tensor-type "blk\..*\.ssm_alpha\.weight=q8_0" \
  --tensor-type "blk\..*\.ssm_beta\.weight=q8_0" \
  --tensor-type "blk\..*\.ssm_out\.weight=q5_0" \
  "D:/models/gguf/Qwen3.5-0.8B-BF16.gguf" \
  "D:/models/gguf/Qwen3.5-0.8B-Q5_0-self-0.gguf" \
  Q5_0
```

### 文件大小对比

| 模型 | 大小 | BPW | 量化类型 |
|------|------|-----|----------|
| Q8_0-self-0 | 775 MiB | 8.52 | 全对称 |
| Q5_K_S-self（基线） | 543 MiB | 5.93 | 混合（对称+非对称） |
| Q5_0-self-0 | 539 MiB | 5.89 | 全对称 |

Q5_0 比 Q5_K_S 小 4 MiB，因为 Q5_0 不存储 `dmin` 偏移量字段。

### 推理对比

测试环境：RTX 4070 Laptop GPU, seed=42, `-ngl 99 -st --no-display-prompt`

#### 性能对比

| 模型 | Prompt (t/s) | Generation (t/s) | 显存占用 (model) |
|------|-------------|-------------------|------------------|
| Q8_0-self-0 | ~370-490 | ~183 | 763 MiB |
| Q5_K_S-self | ~225-410 | ~226-230 | 532 MiB |
| Q5_0-self-0 | ~200-460 | ~228-232 | 527 MiB |

Q8_0 的 Generation 速度明显更慢（~183 t/s vs ~230 t/s），因为模型更大，显存带宽成为瓶颈。
Q5_0 和 Q5_K_S 的生成速度基本一致。

#### 输出质量对比

**测试 1：数学题 "What is 15 + 27?"（-n 256）**

| 模型 | Thinking 质量 | 最终输出 |
|------|--------------|----------|
| Q5_K_S | 简洁高效，直接算出 42 | 完整输出 `15 + 27 = **42**` |
| Q8_0 | 详细列竖式，过程正确 | 256 token 内未完成 thinking，无最终输出 |
| Q5_0 | 出现犹豫和错误（`7+7=14`），自我纠正后得到 42 | 256 token 内未完成 thinking，无最终输出 |

**测试 2：列举题 "List 3 benefits of exercise."（-n 512）**

三个模型的 thinking 质量相当，都能正确列出心血管健康、心理健康、体重管理等方面。
但 512 token 内三个模型都未能完成 thinking 输出最终答案（Qwen3.5-0.8B 的 thinking 模式较为冗长）。

**测试 3：问候 "Hello, how are you?"（-n 256）**

三个模型表现一致，thinking 结构相同，都能正常进入对话模式。

### 结论

1. **Q8_0 vs Q5_K_S**：Q8_0 精度更高但生成速度下降约 20%（183 vs 230 t/s），显存多占 231 MiB。thinking 过程更详细但不一定更高效。对于 0.8B 小模型，Q8_0 的精度优势不明显。

2. **Q5_0 vs Q5_K_S**：Q5_0 在数学推理中出现了明显的质量下降——thinking 过程出现错误计算（`7+7=14`）和反复犹豫，而 Q5_K_S 直接算出正确答案。这说明 K-quant 的非对称量化（子块 scale + min）确实比纯对称 Q5_0 保留了更多精度，尤其在需要精确计算的场景。

3. **对称 vs 非对称的实际影响**：在同为 5-bit 的情况下，Q5_K_S（混合对称+非对称）比 Q5_0（纯对称）质量更好，验证了非对称量化的 `dmin` 偏移量对低比特量化精度的重要性。文件大小差异仅 4 MiB（539 vs 543 MiB），但质量差异可观。
