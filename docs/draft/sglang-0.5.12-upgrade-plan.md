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
| 8 | sgl-router 装 **slime 私有 wheel** + 断言 `'slime' in __version__` | ⚠️ **原判"不要照搬"有误**——该 wheel(`zhuzilin/sgl-router` v0.3.2-9daabcd)含 **r3(routing replay)透传 patch**,官方 `sglang-router` 无;用官方会导致 R3 死锁,见 §8.7 |
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
5. **R3 rollout routing replay**(`--use-rollout-routing-replay`):❌ **已验证失败** —— 官方 sgl-router 不透传 routed_experts,step 0 死锁,根因与修复见 §8.7。

### 7.5 依赖类改动状态

- ✅ **`requirements.txt`:`transformers==5.3.0` → `transformers`(不固定)** — 已改,与 slime 一致。
- ✅ **torch_memory_saver**:`TMS_CUDA_MAJOR` 自动探测;commit `afc13785`(已是 redai-infra HEAD,比 slime `a193d9dd` 新)。
- `huggingface_hub==1.7.2`:保留(与 transformers 5.6.0 兼容);v0.5.12.post1 未固定 hf_hub。
- ⚠️ **`sglang-router`**:官方 `sglang-router`(实测 0.3.2)**不支持 routing replay 透传** —— 请求经 router 转发后,响应 `meta_info` 丢失 `routed_experts`,导致 R3 死锁(见 §8.7)。
  修复:改装 slime 私有 fork wheel `zhuzilin/sgl-router` v0.3.2-9daabcd(版本号 `0.3.2+slime`,含 `support r3` patch)。**此即 §2 item 8 "用官方 router" 判断需推翻的点。**
- ✅ **`pip install --ignore-installed PyJWT`**:slime v0.3.0 在 `pip install -r requirements.txt` 前加的一行。
  作用是规避 pip 卸载报错——当某依赖要升级/重装 PyJWT,但环境里已有的 PyJWT 是以 distutils/系统方式装的(缺 `RECORD`),
  pip 无法卸载会报 `Cannot uninstall 'PyJWT'`。`--ignore-installed` 让 pip 直接装新的、不尝试卸载旧的。
  **决策(2026-06-30):先不加**;待首次构建若真撞到该报错再补。
  **更新(2026-07-01):首次构建确认撞到 `Cannot uninstall 'PyJWT'`,已在 `docker/Dockerfile` 的 requirements 安装前补上 `pip install --ignore-installed PyJWT`,与 slime 逐字一致。**
- Megatron-Bridge commit:slime v0.3.0 换了源/commit,Relax 用 NeMo 官方源,是否同步待定。

### 7.6 验证状态

- ✅ 新 patch 干净应用于 v0.5.12.post1(`git apply --check` exit 0)
- ✅ 改动文件 Python 语法检查通过
- ⚠️ 已打镜像并首次实跑冒烟(Kimi-K2.6, 2026-07-02~03),暴露并修复了 5 个 bug,见 **第 8 节**;
  7.4 中 BF16 MoE / transformers 5.6.0 等仍待更全面验证。
- ❌ 后续单独开 R3 冒烟(2026-07-03 晚)发现第 6 个问题:`--use-rollout-routing-replay` step 0 死锁,根因=官方 sgl-router 无 routed_experts 透传,**修复待落地**,见 **§8.7**。

---

## 8. 首次实跑冒烟发现的 bug 与修复(2026-07-02 ~ 07-03)

镜像按第 7 节改动打好后,以 **Kimi-K2.6**(MLA / INT4 QAT / colocate GRPO / TP2·PP1·CP2·EP8 / 8×GPU 单机)
减层冒烟为载体首次实跑,依次暴露并修复了下列问题。代码类改动见 commit `256b030`;FA3 两处崩溃的完整排查见同目录
`fa3-window-size-left-crash-postmortem.md` 与 `kimi-k2.6-cp-split-with-sizes-crash-investigation.md`。

### 8.1 colocate 显存挤兑 OOM(NCCL calloc 失败)

- **现象**:step 0 在 `_update_router_expert_bias` 的 all_reduce 处
  `NCCL WARN Cuda failure 'out of memory' / Failed to CUDA calloc 10485760 bytes`。并非训练权重装不下——2 层模型 offload 后仅 ~2GB。
- **根因**:colocate 下 SGLang 静态 KV 池(`--sglang-mem-fraction-static 0.7` ≈ 55GB/卡)+ Kimi 超大 MoE(EP=8 多 expert)激活
  + seq 16k 动态 batch 叠加把卡榨干,NCCL 连 10MB 通信 buffer 都拿不到(报的是 NCCL calloc,而非 torch OOM)。
- **修复/缓解**:上下文 16384 → 4096(`--rollout-max-response-len` / `--eval-max-response-len` / `--max-tokens-per-gpu`,见 run 脚本);
  可选降 `--sglang-mem-fraction-static`、设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

### 8.2 减层 checkpoint 与 `--num-layers` 不一致 → 训推 diff 0.7

- **现象**:`train_rollout_logprob_abs_diff ≈ 0.757`(全量层历史值 ~0.04),稳定不发散。
- **根因**:脚本 `--num-layers 3` 但 checkpoint `text_config.num_hidden_layers=2`;日志 `model_provider` 打印
  `Override provider.num_layers: 2 -> 3`,bridge 只填 2 层权重、**第 3 层随机初始化** → 训练前向被污染,与干净 2 层的 SGLang 推理系统性偏离。
- **信号/坑**:diff 稳定(非 NaN、不发散)= 结构性系统偏差;唯一日志线索是那条 **INFO 级** `Override provider.num_layers`,
  混在十几条同款 Override 里、无告警、还被 `[repeated 7x]` 折叠,极易漏。
- **修复**:`NLAYERS=2` 与 checkpoint 对齐;`scripts/models/kimi-k2.6.sh` 把 `NLAYERS` 改为可 env 覆盖(`${NLAYERS:=61}`)。
  **教训:减层冒烟必须让 `--num-layers` == checkpoint 的 `num_hidden_layers`。**

### 8.3 CP `split_with_sizes` 崩溃(is_vl_model padding 记账口径不一致)

- **现象**:step 0 结尾指标上传阶段 `RuntimeError: split_with_sizes expects split_sizes to sum exactly to ...`;仅 `CP>1` + `qkv_format=thd` 触发。
- **根因**:Kimi-K2.6 是纯文本 MoE,但 checkpoint 带 `preprocessor_config.json` → `actor.py:154` 误判 `is_vl_model=True`
  → 打包/前向走 bridge VL+CP+thd unsplit 路径、按 pad 后长度切 token 级字段;而记账侧(`log_rollout_data` / loss / `stream_dataloader`)
  调 `maybe_padded_total_lengths` 时 **漏传 `is_vl_model`** → 按未 pad 公式重算 → 每样本差 2 → 崩。
- **修复**:5 处 `maybe_padded_total_lengths` 补 `is_vl_model`(actor.py / data.py×2 / loss.py / stream_dataloader.py)。
  **与 sglang 升级无关**,dev 上同样存在,本次因启用 `--context-parallel-size 2` 才暴露。

### 8.4 FA3 `window_size_left` TypeError(TE 2.14.1)

- **现象**:CP flash 路径 `TypeError: _flash_attn_forward() got an unexpected keyword argument 'window_size_left'`。
- **根因**:容器混装的残缺 FA3 beta(旧 `window_size` 元组签名)被 TE 2.14.1 选中,而 TE 2.14.1 假设较新 FA3 API(带 `window_size_left`)。详见 postmortem。
- **修复**:`docker/Dockerfile` FA3 hopper commit `fbf24f67` → `0f82fead`(2026-03-18,TE 2.14.1 发布前最后一次 hopper 接口改动,含 `window_size_left`)。

### 8.5 FA3 `schema_.has_value()` / `basic_string::_M_create`(custom op 重复注册 → use-after-free)

- **现象**:换上 0f82fead 后,`flash_attn_fwd → _flash_attn_forward` 处报三种不同错——
  `schema_.has_value() ... Tried to access the schema for .`(空 op 名)/ `basic_string::_M_create` / `UnicodeDecodeError`,典型 use-after-free。参数全合法,abi3/非-abi3 都崩。CP 路径(`cp_p2p_fwd_flash_attn`)与**非 CP 路径**(`flash_attn_varlen_func`,`qwen3-4b-GRPO-gpu8` CP=1 复现)都触发——即任何 `--attention-backend flash` 的前向都会中招。
- **根因**:site-packages 里存在**两份内容完全相同**、都注册同名 torch custom op `flash_attn_3::_flash_attn_forward` 的文件:copy-hack 的包内 `flash_attn_3/flash_attn_interface.py`(TE import 时绑定)与 `setup.py install` 落到 egg 根的**游离顶层** `flash_attn_interface.py`。第二个 importer 触发 torch `get_library_allowing_overwrite`
  → `_destroy()` 掉 TE 已注册的 Library → TE 旧句柄再 dispatch 时读到已释放/空 schema。
- **订正(2026-07-06,`qwen3-4b-GRPO-gpu8` 干净重打镜像实测)**:原判「干净镜像不含游离顶层模块、纯属容器污染、Dockerfile 无需改动」**已被推翻**。`0f82fead` 的 hopper `setup.py` 用 `py_modules=["flash_attn_interface"]`,**每次干净构建都会把顶层 `flash_attn_interface.py` 种进 egg 根**(实测 `import flash_attn_interface` 落在 `flash_attn_3-3.0.0-…egg/flash_attn_interface.py`,与包内那份字节相同、各注册 3 处 custom op);叠加 copy-hack → 干净镜像天然自带重复注册。**故 Dockerfile 必须加 strip 步骤**(已落地,见 `docker/Dockerfile` FA3 块)。
- **修复**:**只删**游离顶层 `flash_attn_interface.py`——`find $python_path -maxdepth 1 -name flash_attn_interface.py -delete` + `find $python_path -maxdepth 2 -path '*.egg/flash_attn_interface.py' -delete`(此二式不误伤 FA2 的 `flash_attn/flash_attn_interface.py` 与 copy-hack 的 `flash_attn_3/flash_attn_interface.py`);**`flash_attn_config.py` 必须保留**——包内 `flash_attn_3/flash_attn_interface.py` 在 `round_up_headdim()` 里 `from flash_attn_config import CONFIG`,删掉会在前向 `ModuleNotFoundError`(原「删 flash_attn_interface.py / flash_attn_config.py」说法有误,已订正)。verify:`assert find_spec('flash_attn_interface') is None` 且 `import flash_attn_config` 成功。

### 8.6 小结

| # | bug | 落地位置 | 是否由 0.5.12 升级引入 |
|---|---|---|---|
| 8.1 | colocate OOM (NCCL calloc) | run 脚本上下文降到 4k | 否(colocate 通用) |
| 8.2 | 层数不一致 → diff 0.7 | `NLAYERS=2` + 可 env 覆盖 | 否(减层冒烟配置) |
| 8.3 | CP `split_with_sizes` | `is_vl_model`×5(commit `256b030`) | 否(dev 已存在,CP=2 暴露) |
| 8.4 | FA3 `window_size_left` | Dockerfile FA3 → `0f82fead` | 是(TE 2.14.1 栈相关) |
| 8.5 | FA3 重复注册崩溃 | Dockerfile FA3 块加 strip(删顶层 `flash_attn_interface.py`,留 `flash_attn_config.py`) | **是**(`0f82fead` setup.py 布局,干净镜像即复现) |
| 8.7 | R3 routing replay 死锁 (Bug A) | 换 slime 私有 router wheel(见 §8.7,已落地) | **是**(官方 sgl-router 无 r3 透传) |
| 8.8 | R3 CP fan-out 缺失 0-d crash (Bug B) | `_broadcast_routed_experts` CP→PP→TP(见 §8.8) | 否(relax 潜伏,R3+CP>1 暴露) |
| 8.9 | R3 `is_vl_model` CP 对齐错位 diff 6× (Bug C) | `fill_routing_replay` 按 `tp*cp*2` 对齐(见 §8.9) | 否(§8.3 同源遗漏,R3+CP>1 暴露) |

> 结论:8.1-8.5 中 **8.4 与 8.5** 均源于本次 sglang/镜像升级栈——8.4 是 TE 2.14.1 需要更新的 FA3,8.5 是该更新 commit(`0f82fead`)`setup.py` 布局的副作用(干净镜像即复现,需 Dockerfile strip;2026-07-06 订正,原判「容器污染、不改 Dockerfile」有误);
> 8.1/8.2 是 colocate 与减层冒烟的通用配置问题;8.3 是 dev 早已潜伏、被 CP=2 暴露的通用 bug。
> **8.7** 是后续单独开 R3 冒烟(2026-07-03 晚)才发现的,**直接由本次升级的 router 依赖决策引入**(用官方 sgl-router 而非 slime 私有 fork);修好后又串联暴露 **8.8/8.9**(均为 relax 侧潜伏、与升级无关,`R3 + CP>1` 才触发)。R3 在 CP>1 端到端跑通至此达成。

---

## 8.7 R3(rollout routing replay)在官方 sgl-router 下 step 0 死锁(2026-07-03)

在 Kimi-K2.6 冒烟脚本(`run-kimi-k2.6-2layers-8xgpu-int4.sh`)开启 `--use-rollout-routing-replay` 后,任务在 **step 0 死锁**(非崩溃)。

### 现象
- Actor 每分钟 ~2000 次空轮询 `GET_META`/`GET_CONSUMPTION`(TransferQueueController 统计),始终拿不到可消费 batch。
- Rollout 侧无限 `waiting for data system to catch up`。
- Rollout 实际已生成完 256/256、`PUT_DATA` 一次;但 `NOTIFY_DATA_UPDATE` 仅 1 次,消费方永远等不齐字段。

### 根因链(逐层实证)
1. 开 R3 后 Actor 的 `build_data_fields`(`relax/engine/sft/runtime.py:92`)把 `rollout_routed_experts` 列为**必需字段**。
2. 但生产侧写进 TransferQueue 的 TensorDict **没有这一列**(日志中 `Transferring batch rollout_batch: TensorDict(...)` 只有 tokens/loss_masks/rollout_log_probs/... 9 列)。
3. `convert_samples_to_train_data` 仅当 `samples[0].rollout_routed_experts is not None` 才写该列(`relax/utils/utils.py:147`)。
4. sample 只在 sglang 响应 `meta_info` 含 `routed_experts` 时才被赋值(`relax/engine/rollout/sglang_rollout.py:394`)。
5. → **sglang 响应里没有 `routed_experts`。**

### 真凶 = 官方 sgl-router(实测 0.3.2)转发时丢字段
用 `tools/repro_routed_experts/` 隔离复现(2 层 Kimi INT4):

| 探测 | routed_experts |
|---|---|
| 引擎直连 `/generate`,text | **PRESENT**(base64 len≈1452) |
| 引擎直连 `/generate`,input_ids | **PRESENT** |
| **经 sgl-router**,text | **ABSENT** |
| **经 sgl-router**,input_ids(==relax 实际路径) | **ABSENT** |

排除项:dp-attention(tp2/dp2/ep2 直连正常)、EP、请求 flag 传递(离线构造 `GenerateReqInput(..., return_routed_experts=True)` 字段保留)、sglang 引擎侧 capturer(`HostCache[routed_experts]` 正确分配、`get_topk` 恒返回张量)。
relax rollout 恰恰 POST 到 sgl-router(`sglang_rollout.py:251`,router 由 `distributed/ray/rollout.py:_start_router` 启动),**router 在转发请求/回传响应时丢掉了 routed_experts** → `sample.rollout_routed_experts=None` → 列缺失 → Actor 死等。

### 为什么 slime 能用而 relax 不能
- slime Dockerfile 装的是**私有 fork wheel**:`zhuzilin/sgl-router` `v0.3.2-9daabcd`(`sglang_router-0.3.2-...whl`),并断言 `'slime' in sglang_router.__version__`。
- 该 fork 提交史明确含 R3 patch:`4596bd4 support r3` / `9b66be3 Support r3 with pd` / PR#7 / PR#12(`9daabcd`)/ `f70eff0 set version to 0.3.2+slime`。
- 即 **routed_experts 的 router 透传是 slime fork 新增的,官方 sgl-router 0.3.2 没有。** relax 当初(§2 item 8)决定"用官方、不照搬 slime 私有 wheel",正是本 bug 的直接原因。

### 社区佐证
- sglang [#12075](https://github.com/sgl-project/sglang/issues/12075):RL routing replay 从 MoE 取 router 输出的功能诉求(closed,指向引擎侧 [PR #9499](https://github.com/sgl-project/sglang/pull/9499),已进 0.5.12)。
- [#8791](https://github.com/sgl-project/sglang/issues/8791):router 只 merge **显式处理过**的 meta_info 字段(如 logprobs),其余丢弃。
- [#9621](https://github.com/sgl-project/sglang/issues/9621):router **丢弃 extra 参数**,直连 worker 才生效。

### 修复(已落地)
把 `docker/Dockerfile` 的 sgl-router 换成 slime 私有 fork wheel `zhuzilin/sgl-router` v0.3.2-9daabcd(`0.3.2+slime`)。**修好 A 后 routed_experts 流到训练侧,又串联暴露出 B、C 两个 relax 侧潜伏 bug(见 §8.8/§8.9)——R3 在 CP>1 端到端跑通实为"三连击"。** 尚未细究 router 是"请求进"还是"响应出"丢字段——换 fork 即可绕过,无需细分。

> 复现脚本:`tools/repro_routed_experts/serve_and_probe.sh`(单机直连)、`serve_router_probe.sh`(引擎+router 对照)。

---

## 8.8 R3 Bug B:CP>1 下 routed_experts 广播缺 CP fan-out(2026-07-06)

- **现象**:修好 A 后 CP2 实跑,训练侧 `stream_dataloader.py` `for i in range(len(routed_experts_offsets)-1)` 报 `TypeError: len() of a 0-d tensor`。
- **根因**:`_broadcast_routed_experts` 只做 PP→TP 广播,假设 (TP0,PP0) 的所有 CP partner 都持有源数据;但 `should_fetch` 的 CP=0 guard 使只有 (TP0,PP0,CP0) 真正持有 → CP≠0 rank 收 0-d offsets。`GRPO + R3 + CP>1` 才触发,SFT 不碰。
- **修复**:`stream_dataloader.py` `_broadcast_routed_experts` 重构为 CP→PP→TP 三段广播(补短路 `cp_trivial`)。**与 sglang 升级无关,是 relax 早已潜伏、被 R3+CP>1 暴露的通用 bug。** 详见 investigation doc §3.4。

## 8.9 R3 Bug C:`is_vl_model` 下 `fill_routing_replay` CP 对齐用错(2026-07-06)

- **现象**:修好 A、B 后 CP2 开 R3,`train_rollout_logprob_abs_diff` 从不开 R3 的 0.028 涨到 **0.17**(6×,所有训推一致性指标同步恶化);CP=1 开 R3 正常(0.012)。**非崩溃,静默数值错。**
- **根因**:与 **§8.3 同源** —— Kimi-K2.6 是 VL 模型(`is_vl_model=True`,检测正确),CP>1+thd 走 bridge unsplit 路径,bridge 按**每样本 `tp*cp*2`** 对齐做 CP 切分。§8.3(commit `256b030`/`2edaade`)已把 `maybe_padded_total_lengths(is_vl_model)` 补进 loss/metric 侧 5 处,**却漏了 routing-replay 这第 6 处**:`fill_routing_replay` 仍用 `slice_with_cp` 的 `2*cp` 对齐 + 全局 `tp*dpm` pad → 每样本 chunk 边界与 bridge 不一致 → routed_experts 挂错 token、SP rank>0 整体平移。`_align_top_indices` 静默截断只遮丑。
- **定位**:env `R3_DEBUG_ALIGN` 探针 —— 广播指纹探针未触发(证 §8.8 广播 OK),`_align_top_indices` firing 探针狂刷 `recorded=1152 vs scores=1064`。
- **修复**:`actor.py` `fill_routing_replay` 按 `maybe_padded_total_lengths` 分支——VL 路径每样本 pad 到 `tp*cp*2`、去掉全局 pad(与 bridge 一致),非-VL 维持原逻辑。**与 sglang 升级无关。** 实跑验证 diff 0.17→**0.013**、`R3DBG-C`=0。详见 investigation doc §3.5;防复发建议(抽共享 CP 切分 helper)见 §3.6。
