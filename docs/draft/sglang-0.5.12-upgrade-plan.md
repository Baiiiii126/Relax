# sglang 升级到 v0.5.12.post1-cu129 方案

> 草稿 / 调研产物。记录把 Relax 的 sglang 从 **v0.5.9** 升级到 **v0.5.12.post1-cu129** 所需的全部工作,
> 以及与 [THUDM/slime](https://github.com/THUDM/slime) 的对照分析。
> 生成时间:2026-06-30。本地参考仓库:`/root/data/slime`(已含 v0.2.3 / v0.2.4 / v0.3.0 tag)。

## 0. 背景:当前版本是如何确定的

sglang 版本**不在** `requirements.txt` / `pyproject.toml` 里固定,而是完全由 Docker 构建流水线决定:

```
lmsysorg/sglang:v0.5.9 (基线镜像, docker/Dockerfile:4)   ← 自带 sglang 0.5.9 + transformers 4.57
        │
        ▼ [改动1] rsync 覆盖 update-transformers-v5 分支源码 (Dockerfile:32-34)
        │         目的:把 transformers 5.x 支持 backport 到 v0.5.9；装 transformers==5.3.0
        ▼ [改动2] git apply docker/patch/latest/sglang.patch --3way (Dockerfile:114-124)
        │         受 PATCH_VERSION / ENABLE_SGLANG_PATCH 控制
        ▼ [改动3] 运行时 --sglang-* 参数 (relax/utils/arguments.py)
   最终运行的 sglang（importlib.metadata 自报 0.5.9，实为定制版）
```

关键事实:
- **基线 = v0.5.9**(本机 `pip show sglang` = 0.5.9 即来自此)。
- **当前 CUDA = 12.9**(由基线镜像继承)。
- `update-transformers-v5` 是从 **v0.5.9 时间点的 main** 切出的特性分支(API 比对:behind v0.5.9 仅 11 commit,behind v0.5.10 达 846),与基线 v0.5.9 同代际,作用就是补 transformers 5.x。

## 1. 目标版本与核心判断

| 项 | 当前 | 目标 |
|---|---|---|
| sglang 基线镜像 | `lmsysorg/sglang:v0.5.9` | `lmsysorg/sglang:v0.5.12.post1-cu129`(已确认存在) |
| CUDA | 12.9 | 12.9(无变化) |
| transformers | 4.57(基线) + overlay 到 5.3.0 | **5.6.0(v0.5.12.post1 原生)** |

**核心判断:`update-transformers-v5` overlay 升级后不再需要。**
其唯一目的是给只认 transformers 4.57 的 v0.5.9 backport 5.x;而 v0.5.12.post1 的 `pyproject.toml` 已原生 `transformers==5.6.0`。slime 的 Dockerfile 自始至终不做 transformers overlay,也印证这一点。

## 2. 参考:slime 是怎么升的(v0.2.3 → v0.3.0)

slime 在 **v0.3.0** 做了与本次完全对应的跳变(v0.5.9 → v0.5.12.post1-cu129)。其 `docker/Dockerfile` diff:

| # | slime 改动 | 是否适用 Relax |
|---|---|---|
| 1 | `SGLANG_IMAGE_TAG` v0.5.9 → **v0.5.12.post1-cu129** | ✅(注意 Relax 用 `lmsysorg/`,slime 用自建 `slimerl/`) |
| 2 | `MEGATRON_COMMIT` 跟进 | ⚠️ 复核 Relax 的 Megatron-Bridge commit |
| 3 | 新增 `ARG TMS_CUDA_MAJOR=` + 自动探测 `torch.version.cuda` | ✅ 替换 Relax 硬编码的 `TMS_CUDA_MAJOR=12` |
| 4 | torch_memory_saver commit 跟进 | ⚠️ 复核 |
| 5 | Megatron-Bridge 换源 `fzyzcjy/dev_rl`→`radixark/bridge` | ⚠️ Relax 用的是 NeMo 官方源,自行决定 |
| 6 | 新增 FlashQLA(Qwen3.5/Next GDN backend,SM90+) | 可选 |
| 7 | requirements 前加 `pip install --ignore-installed PyJWT` | ✅ 大概率需要(解依赖冲突) |
| 8 | sgl-router 装 **slime 私有 wheel** + 断言 `'slime' in __version__` | ❌ **不要照搬**,Relax 用官方 `sglang-router` |
| 9 | tilelang 用 cu128 wheel 源 | ✅ 跟随,保持 cu128 |
| 10 | `latest` patch 指向重写过的 **v0.5.12.post1** 那套 | ✅ 见第 4 节 |

> 注:slime 与 Relax 的 tilelang 都用 cu128 wheel,保持不变即可。

## 3. Relax `docker/Dockerfile` 需改清单

| 位置 | 当前 | 改为 |
|---|---|---|
| `:4` | `BASE_IMAGE=lmsysorg/sglang:v0.5.9` | `lmsysorg/sglang:v0.5.12.post1-cu129` |
| `:32-34` | update-transformers-v5 rsync overlay | **整段删除** |
| `:40` | `pip install transformers==5.3.0 ...` | 删 transformers(基线自带 5.6.0);cudnn 视情保留 |
| `:46` | tilelang `cu128` | **保持 cu128** |
| `:61` | `TMS_CUDA_MAJOR=12` 硬编码 | 改为自动探测 `torch.version.cuda` |
| `:84-94` | Megatron-Bridge `2faedbf...` | 复核是否需跟 slime 调整 |
| `:98-100` | requirements 安装 | 视情在前面加 `pip install --ignore-installed PyJWT` |
| `:114-124` | sglang.patch 应用逻辑 | 逻辑不变,**替换 patch 内容**(第 4 节) |
| `docker/Dockerfile.npu:56` | `git clone -b v0.5.9` | `v0.5.12.post1`,并复核 npu patch |
| `requirements.txt` | `transformers==5.3.0`、`sglang-router>=0.2.3` | transformers 跟 5.6.0 或删;router 复核兼容 |

> ⚠️ 改 `requirements.txt` / Megatron-Bridge 属 CLAUDE.md "添加依赖需先确认" 范畴,动手前对齐。

## 4. 最大头:`docker/patch/latest/sglang.patch` 的 rebase

### 4.1 血缘关系(已验证)

- **Relax 的 sglang.patch 派生自 slime ≈v0.2.3**:与 slime 各版本 `latest/sglang.patch` 的差异行数为
  v0.2.3 **1338**(最近) < v0.2.4 2595 < v0.3.0 4012;且两者都基于 v0.5.9。
- slime 在 **v0.2.4** 继续往 v0.5.9 patch 加料(+1925 行),Relax **未跟进**。
- slime 在 **v0.3.0** 已把整套 patch rebase 到 v0.5.12.post1(`docker/patch/v0.5.12.post1/sglang.patch`,41 文件),
  可作为 Relax 新 patch 的**骨架**直接复用。

### 4.2 slime v0.2.3 → v0.3.0 patch 文件级增量(52 → 41:丢 22 / 增 11 / 留 30)

**已被上游合并、升级后大概率可删的 22 个**(部分 Relax 也有,见 4.4 风险):
```
nsa/index_buf_accessor.py, nsa/utils.py, communicator_nsa_cp.py,
ep_moe/deepep_bf16_kernels.py, ep_moe/layer.py, token_dispatcher/deepep.py,
routed_experts_capturer.py, logits_processor.py, rotary_embedding.py,
dp_attention.py, scheduler_pp_mixin.py, scheduler_metrics_mixin.py,
disaggregation/common/conn.py, distributed/parallel_state.py,
mem_cache/allocator.py, deepseek_common/attention_backend_handler.py,
deepseek_nextn.py, gpt_oss.py, qwen3_5.py, eagle_info.py, eagle_worker.py,
tokenizer_communicator_mixin.py
```
**为 v0.5.12.post1 新增的 11 个**(投机解码 v2、GLM 新变体、deep_gemm runner 等):
```
eagle_worker_v2.py, multi_layer_eagle_worker_v2.py, glm4_moe_lite.py,
glm4_moe_nextn.py, moe_runner/deep_gemm.py, environ.py,
tokenizer_control_mixin.py, req_time_stats.py, model_loader/weight_utils.py,
disaggregation/base/conn.py, disaggregation/utils.py
```

### 4.3 Relax 相对 slime v0.2.3 的三类文件(决定要自己 port 什么)

Relax patch 共 56 文件。剔除 patch 里 `@@` 行号偏移后,按真实改动内容分类:

**A 类 — 与 slime 改动完全一致的 28 个 → 直接复用 slime v0.3.0 已 rebase 版本,无需人工**
```
distributed/parallel_state.py, entrypoints/engine.py, entrypoints/http_server.py,
nsa/index_buf_accessor.py, nsa/nsa_indexer.py, communicator_nsa_cp.py,
dp_attention.py, logits_processor.py, ep_moe/deepep_bf16_kernels.py,
routed_experts_capturer.py, token_dispatcher/deepep.py,
compressed_tensors.py, compressed_tensors_wNa16_moe.py, io_struct.py,
schedule_batch.py, scheduler_output_processor_mixin.py, scheduler_profiler_mixin.py,
tokenizer_communicator_mixin.py, tokenizer_manager.py, tp_worker.py,
hiradix_cache.py, radix_cache.py, attention_backend_handler.py, gpt_oss.py,
processors/glm4v.py, processors/qwen_vl.py, eagle_info.py, eagle_worker.py
```

**B 类 — Relax 改过/扩展的 21 个共有文件 → 把 Relax 增量叠加到 slime 骨架上**
(括号为真实差异行数;升级后若该文件已进 4.2 的"丢弃"名单,则只需在新源码上确认 Relax 增量是否仍必要)

| 文件 | Δ行 | Relax 私有要点 |
|---|---|---|
| `server_args.py` | 40 | 多节点 master-node 端口检查(colocate 避开 Megatron NVSHMEM 端口) |
| `fused_moe_triton/layer.py` | 28 | 非对称量化 zero-point 加载 |
| `eagle_draft_cuda_graph_runner.py` | 18 | topk clamp 防越界 |
| `disaggregation/prefill.py` | 18 | PD 超时/CP 微调 |
| `mem_cache/memory_pool.py` | 15 | NSA pool / mamba 索引修正 |
| `scheduler_pp_mixin.py` | 13 | PP+CP last-rank 本地结果队列 |
| `scheduler_update_weights_mixin.py` | 11 | disagg 队列内存释放钩子 |
| `model_executor/model_runner.py` | 10 | `post_process_weights` / draft worker |
| `disaggregation/common/conn.py` | 10 | CP 路由 |
| `models/glm4v_moe.py` | 7 | encoder_only / rope 兼容 |
| `managers/scheduler.py` | 7 | disagg gloo group |
| `nsa/utils.py` | 7 | CP split padding |
| `disaggregation/mooncake/conn.py` | 6 | session 失败处理 |
| `disaggregation/decode.py` | 6 | 超时/retract |
| `configs/model_config.py` | 6 | tf 版本校验放宽 / DeepseekV32 draft |
| `utils/weight_checker.py` | 4 | 跳过 cos_sin_cache |
| `models/deepseek_nextn.py` | 4 | topk_indices 透传 |
| `mem_cache/allocator.py` | 4 | torch fallback |
| `ep_moe/layer.py` | 4 | BF16 MoE 分支 |
| `utils/common.py` | 2 | **SafeUnpickler 白名单加 `"Relax."`** |
| `models/qwen3_vl.py` | 2 | deepstack 顺序 |

**C 类 — Relax 独有的 7 个文件(slime 没有)**
- 真·私有功能(需在 v0.5.12.post1 新源码上重写):
  `deepseek_v2.py`、`deepseek_common/attention_forward_methods/forward_mla.py`、
  `kimi_k25.py`、`glm4_moe.py`、`multimodal/processors/base_processor.py`
- 实为同补丁落在被改名/移动文件上(非新功能,slime 骨架里大概率已含对应路径):
  `rotary_embedding/base.py`(slime: `rotary_embedding.py`)、
  `observability/scheduler_metrics_mixin.py`(slime: `managers/scheduler_metrics_mixin.py`)

### 4.4 新 patch 的组装公式

```
新 sglang.patch = slime v0.3.0 的 v0.5.12.post1 patch（41 文件骨架，已 rebase）
               −  slime 特有、不属于 Relax 的项（如 router 断言）
               +  B 类 21 个文件的 Relax 私有增量
               +  C 类 5 个 Relax 独有功能文件（在新源码结构上重写）
               +  C 类 2 个改名文件的对位（多半已在骨架中）
```
> ⚠️ **以上 ≈26 处是"调研前的上限估算"**(仅按 Relax patch vs slime v0.2.3 的文件级差异得出,尚未核查 v0.5.12.post1 源码)。
> 实际按第 5 节逐项 grep 目标源码后,**绝大多数 B/C 项已被上游原生实现或被 slime 骨架覆盖,真正需人工 port 的塌缩到 4 处**:
> `utils/common.py`(SafeUnpickler `"Relax."`)、`multimodal/processors/base_processor.py`(gpu_id 设备)、
> `models/kimi_k25.py`(post_load_weights)、`server_args.py`(PP+CP 断言放宽)。
> **最终落地结果以第 7 节执行记录为准**,本节(第 4 节)保留为方法论/估算过程参考。

产物:生成新的 `docker/patch/<日期戳>/sglang.patch`,更新 `latest` 软链。

## 5. 必须验证的风险点

1. **BF16 MoE 功能可能丢失**:4.2 "丢弃" 名单里的 `deepep_bf16_kernels.py` / `ep_moe/layer.py` 被 v0.5.12.post1 的
   `moe_runner/deep_gemm.py` 取代。Relax 依赖 BF16 DeepEP 路径,需确认新实现覆盖需求,否则要保留这部分私有 patch。
2. **Relax 私有热更新链路**:`post_process_weights` 全链、`_import_static_state` 的 inference_mode 写回、
   `SafeUnpickler` 的 `"Relax."` 白名单 —— slime 不一定有,必须确保 port 到新版本。
3. **transformers 5.3.0 → 5.6.0**:自身可能引入 breaking change,影响 model-integration 代码。
4. **Relax 自身对 sglang 内部 API 的调用**(`relax/backends/`、`relax/engine/`)跨版本可能 break,需 grep 复核。
5. **基线镜像差异**:Relax 用 `lmsysorg/sglang`(官方),slime 用 `slimerl/sglang`(含 slime 私有改动);
   slime 的 router 断言等不可照搬。

## 6. 建议执行顺序

1. 确认 `lmsysorg/sglang:v0.5.12.post1-cu129` 可 pull(✅ 用户已确认存在)。
2. 改 `Dockerfile`:基线 tag、删 overlay、TMS 自动探测、tilelang 保持 cu128(第 3 节)。
3. 以 slime v0.3.0 的 v0.5.12.post1 patch 为骨架,叠加 B/C 类 Relax 私有改动(第 4 节),生成新 patch。
4. 打镜像 → 跑最小 rollout + 权重同步冒烟,重点验证第 5 节的 1/2 两项。
5. `requirements.txt` / Megatron-Bridge 等依赖改动单独与维护者确认。

---
*附:本文所有文件级结论由 `/root/data/slime` 各 tag 与 `docker/patch/latest/sglang.patch` 的比对脚本生成,
比对时已剔除 patch `@@` 行号偏移,只比真实 `+`/`-` 改动内容。*

---

## 7. 执行记录与决策日志(2026-06-30)

按用户指示"按倾向先推进 + 记录决策,待实跑验证后再决定是否删除"执行。

### 7.1 已完成的改动

**Dockerfile**(`docker/Dockerfile`):
- `:4` 基线镜像 → `lmsysorg/sglang:v0.5.12.post1-cu129`
- 删除 `update-transformers-v5` rsync overlay 整段
- torch_memory_saver:`TMS_CUDA_MAJOR` 改为从 `torch.version.cuda` 自动探测(跟进 slime 机制);commit 保持 `afc13785`(redai-infra HEAD,**比 slime 用的 `a193d9dd` 更新**,不降级)
- (CUDA 已是 12.9,无工具链跳变;tilelang 保持 cu128)

**requirements.txt**:
- `transformers==5.3.0` → `transformers`(不固定版本,与 slime 一致,由基线镜像提供 5.6.0,避免 `pip install -r` 把 5.6.0 降级回 5.3.0)

**启动脚本**:不改。NSA indexer rope 仍用 `INDEXER_ROPE_NEOX_STYLE` env 控制——
经核实 **slime v0.3.0 自己的启动脚本(`scripts/run-glm5-744B-A40B.sh`)也是用 env(`INDEXER_ROPE_NEOX_STYLE=0`)**,
CLI flag `--disable-indexer-rope-neox-style` 只在 slime patch 代码里存在、slime 实践中并不用。
故保留 Relax 的 env 方式才是真正与 slime 一致(patch 实现两者都支持,无需改)。

**新 sglang.patch**(`docker/patch/latest/sglang.patch`,43 文件 / 2912 行):
以 slime v0.3.0 的 v0.5.12.post1 patch 为骨架(41 文件),叠加下列 Relax 私有改动后,
在干净 v0.5.12.post1 源码树上 `git apply --check` **通过(exit 0)**,改动文件均通过 Python 语法检查。

构建源码树:`/root/data/sglang`(tag `v0.5.12.post1`)。

### 7.2 保留的 Relax 私有改动(已 port 进新 patch,共 4 处)

> 对照第 4 节调研前的 ≈26 处上限估算:逐项核查 v0.5.12.post1 源码后,大部分已上游/已被 slime 骨架覆盖,最终只需人工 port 下列 4 处。

| 文件 | 内容 | 适配说明 |
|---|---|---|
| `utils/common.py` | SafeUnpickler 加 `"Relax."` 白名单 | slime 仅有 `"slime."`,追加一行 |
| `multimodal/processors/base_processor.py` | 多模态设备 `cuda:{gpu_id}` | 新源码仍硬编码 `"cuda"`,按 `base_gpu_id` 改 |
| `models/kimi_k25.py` | `post_load_weights` MLA absorb 委托 | 新源码无此方法,补回(hasattr 保护) |
| `server_args.py` | PP+CP 断言放宽(PD prefill NSA) | 替换 `assert pp_size==1` |

说明:`indexer rope` 与 `update_weight_delta_*` 均直接沿用 slime 骨架实现,与 slime 一致。
- indexer rope:slime 版实现**同时支持** `INDEXER_ROPE_NEOX_STYLE` env 与 `--disable-indexer-rope-neox-style` flag;
  Relax 与 slime v0.3.0 一样,**实践中用 env 控制**(启动脚本里设 `INDEXER_ROPE_NEOX_STYLE=0`,未改)。
- `update_weight_delta_*`:Relax 源码未用到,惰性保留便于后续升级。

### 7.3 判定为"已上游/已覆盖/对齐 slime"而丢弃的 Relax 私有改动

> ⚠️ 这些在新 patch 中**不再包含**,需实跑验证上游实现确实覆盖需求后,方可认定删除安全。

| 原 Relax 私有项 | 丢弃依据 |
|---|---|
| **`disable_draft_cuda_graph`**(server_args 字段+参数 + `eagle_worker.py` 用法) | 用户决策:Relax 源码/脚本**未使用**,与 slime v0.3.0 一致一并删除 |
| **多节点 prefill PD TCP + master-node 端口守卫**(`server_args.py` PortArgs) | 用户决策丢弃(原 `configure_ipv6` 已移除、PortArgs 重构为 NetworkAddress,移植风险高);如多节点 colocate 出现端口/IPC stall 再单独补回 |
| Relax 旧的 `nsa_indexer.py` env-only RoPE neox 实现 | 被 slime 版实现取代(slime 版同时支持 env 与 CLI flag);**控制方式仍保留 `INDEXER_ROPE_NEOX_STYLE` env**(脚本未改,与 slime v0.3.0 实际用法一致) |
| NSA topk skip/复用(`deepseek_v2.py`/`forward_mla.py`) | v0.5.12.post1 原生含 `skip_topk`/`next_skip_topk`/`index_topk_pattern` |
| **BF16 DeepEP MoE**(`deepep_bf16_kernels.py` + `ep_moe` BF16 forward) | 上游 `moe_runner/deep_gemm.py` 原生 `_run_bf16_contiguous_gemm`/`_run_masked_bf16_gemm` |
| kimi eagle3 接口(`set_eagle3_layers_to_capture`/`get/set_embed_and_head`) | 上游原生(且签名更完整,Relax 简版被取代) |
| glm4_moe rope 兼容 | 上游重构为 `get_rope_config(config)` |
| `post_process_weights` 全链 / `_import_static_state` inference_mode / disagg 队列释放 | slime 骨架已含 |
| weight_checker cos_sin / qwen3_vl deepstack / glm4v_moe encoder_only 等 | 已覆盖 |

### 7.4 必须实跑验证的项(优先级从高到低)

1. **BF16 MoE**:用 BF16 MoE 模型跑 rollout,确认上游 deep_gemm BF16 路径替代了 Relax 自定义 kernel。
2. **server_args 多节点 PortArgs 块**:多节点 PD-prefill colocate 任务验证(启动期失败,反馈快)。
3. **kimi `post_load_weights`**:Kimi K2.5 colocate train→rollout,验证 MLA absorb `w_kc`/`w_vc` 正常。
4. **transformers 5.3.0 → 5.6.0**:验证 model-integration 代码无 breaking。

### 7.5 依赖类改动状态

- ✅ **`requirements.txt`:`transformers==5.3.0` → `transformers`(不固定)** — 已改,与 slime 一致。
- ✅ **torch_memory_saver**:`TMS_CUDA_MAJOR` 自动探测;commit `afc13785`(已是 redai-infra HEAD,比 slime `a193d9dd` 新)。
- `huggingface_hub==1.7.2`:保留(与 transformers 5.6.0 兼容);v0.5.12.post1 未固定 hf_hub。
- `sglang-router>=0.2.3`:复核与新版兼容性(待定)。
- ✅ **`pip install --ignore-installed PyJWT`**:slime v0.3.0 在 `pip install -r requirements.txt` 前加的一行。
  作用是规避 pip 卸载报错——当某依赖要升级/重装 PyJWT,但环境里已有的 PyJWT 是以 distutils/系统方式装的(缺 `RECORD`),
  pip 无法卸载会报 `Cannot uninstall 'PyJWT'`。`--ignore-installed` 让 pip 直接装新的、不尝试卸载旧的。
  **决策(2026-06-30):先不加**;待首次构建若真撞到该报错再补。
  **更新(2026-07-01):首次构建确认撞到 `Cannot uninstall 'PyJWT'`,已在 `docker/Dockerfile` 的 requirements 安装前补上 `pip install --ignore-installed PyJWT`,与 slime 逐字一致。**
- Megatron-Bridge commit:slime v0.3.0 换了源/commit,Relax 用 NeMo 官方源,是否同步待定。

### 7.6 验证状态

- ✅ 新 patch 干净应用于 v0.5.12.post1(`git apply --check` exit 0)
- ✅ 改动文件 Python 语法检查通过
- ❌ 未打镜像、未实跑(需 GPU 构建环境)——7.4 各项均待镜像构建后验证
