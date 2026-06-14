# CycleNet 增强模块由串联改为并联的修改说明书

## 1. 修改目标

当前 `models/CycleNet.py` 中，增强模块在去周期后的 residual 上按顺序串联：

```text
residual x
  -> MRT
  -> FrequencyFilter / SGF / other freq variants
  -> MLP backbone
```

该方式的问题是 MRT 会先改变 residual，再交给频域模块处理。这样会引入两个风险：

1. MRT 的低通池化可能改变频域模块本应处理的中频结构。
2. 实验上 `MRT+Freq` 并没有稳定超过单模块，说明串联不一定能体现二者互补。

并联改法的目标是让 MRT 和 FrequencyFilter 从同一个原始 residual 出发，各自提取互补信息，再把两条分支的 residual delta 融合：

```text
                         -> MRT branch --------\
residual x_base ---------                       +-> fused residual -> MLP backbone
                         -> Freq branch -------/
```

推荐论文叙事：

```text
MRT captures low-frequency multi-resolution trends.
FrequencyFilter captures frequency-domain residual structures.
The two branches operate on the same cycle-removed residual and are fused additively.
```

## 2. 推荐保留串联模式

不要直接删除现有串联逻辑。建议新增参数控制融合方式：

```bash
--fusion_mode serial      # 默认，保持现有结果可复现
--fusion_mode parallel    # 新并联结构
```

原因：

1. 已有日志和 checkpoint 都基于串联/单模块逻辑，保留旧模式方便复现实验。
2. 并联结构需要和串联结构做公平消融。
3. 如果并联只在部分数据集有效，论文里可以报告两种融合方式对比。

## 3. 核心设计

### 3.1 不直接相加完整输出

不推荐：

```python
x = x_mrt + x_freq
```

因为 `x_mrt` 和 `x_freq` 都包含原始 residual，本质会把 `x_base` 重复加两次。

推荐融合 residual delta：

```python
x_base = x

x_mrt = run_mrt_branch(x_base)
x_freq = run_freq_branch(x_base)

delta_mrt = x_mrt - x_base
delta_freq = x_freq - x_base

x = x_base + delta_mrt + delta_freq
```

这个写法有三个优点：

1. 保持初始恒等性：MRT 和 Freq 内部都有 `res_scale=0`，所以初始时 `delta_mrt=0`、`delta_freq=0`。
2. 避免重复注入 residual 本体。
3. 明确表达两个模块提供的是 residual correction。

### 3.2 可选增加分支融合系数

推荐先做最小版本：

```python
x = x_base + delta_mrt + delta_freq
```

如果并联后训练不稳定，再增加两个可学习标量：

```python
self.mrt_branch_scale = nn.Parameter(torch.ones(1))
self.freq_branch_scale = nn.Parameter(torch.ones(1))

x = x_base + self.mrt_branch_scale * delta_mrt + self.freq_branch_scale * delta_freq
```

不建议一开始把分支 scale 初始化为 0。因为每个模块内部已经有 `res_scale=0`，再加一层 0 初始化会让模块学习更慢。

### 3.3 频域分支范围

主论文建议只把 `FrequencyFilter V1` 放入并联主结构：

```text
parallel main = MRT branch + FrequencyFilter V1 branch
```

SGF、V2、V3、V4 建议继续作为消融，不放进主结构。理由：

1. V1 最简单，审稿人最容易接受。
2. SGF 当前没有稳定优于 V1，且 prior 生成脚本需要先修 train-only。
3. V2/V3/V4 会让主方法显得过度搜索。

## 4. 代码修改范围

### 4.1 `run.py`

新增参数：

```python
parser.add_argument(
    '--fusion_mode',
    type=str,
    default='serial',
    choices=['serial', 'parallel'],
    help='Fusion mode for enhancement modules: serial or parallel'
)
```

建议放在增强模块参数附近，即 `--mrt_layers`、`--freq_layers` 后面。

实验 setting 后缀建议加入：

```python
fusion_mode = getattr(args, 'fusion_mode', 'serial')
if fusion_mode == 'parallel':
    module_suffix += '_parallel'
```

注意：`is_training=0` 分支当前只处理了 `mrt_layers` 和 `lowrank`，没有完整处理 `freq_layers`、`sgf_layers`、`freq_loss_alpha`。如果要用测试模式加载并联 checkpoint，需要同步修复 `else` 分支的 `module_suffix` 生成逻辑。

### 4.2 `models/CycleNet.py`

在 `__init__` 中保存模式：

```python
self.fusion_mode = getattr(configs, 'fusion_mode', 'serial')
assert self.fusion_mode in ['serial', 'parallel']
```

在 `forward` 中，把当前串联增强逻辑拆成两个路径。

当前逻辑：

```python
for mrt in self.mrt_blocks:
    x = mrt(x)

for freq in self.freq_blocks:
    x = freq(x)

for freq_v2 in self.freq_v2_blocks:
    x = freq_v2(x)

for freq_v3 in self.freq_v3_blocks:
    x = freq_v3(x)

for freq_v4 in self.freq_v4_blocks:
    x = freq_v4(x)

for sgf in self.sgf_blocks:
    x = sgf(x)
```

推荐改成：

```python
if self.fusion_mode == 'serial':
    for mrt in self.mrt_blocks:
        x = mrt(x)
    for freq in self.freq_blocks:
        x = freq(x)
    for freq_v2 in self.freq_v2_blocks:
        x = freq_v2(x)
    for freq_v3 in self.freq_v3_blocks:
        x = freq_v3(x)
    for freq_v4 in self.freq_v4_blocks:
        x = freq_v4(x)
    for sgf in self.sgf_blocks:
        x = sgf(x)

elif self.fusion_mode == 'parallel':
    x_base = x

    x_mrt = x_base
    for mrt in self.mrt_blocks:
        x_mrt = mrt(x_mrt)

    x_freq = x_base
    for freq in self.freq_blocks:
        x_freq = freq(x_freq)
    for freq_v2 in self.freq_v2_blocks:
        x_freq = freq_v2(x_freq)
    for freq_v3 in self.freq_v3_blocks:
        x_freq = freq_v3(x_freq)
    for freq_v4 in self.freq_v4_blocks:
        x_freq = freq_v4(x_freq)
    for sgf in self.sgf_blocks:
        x_freq = sgf(x_freq)

    x = x_base + (x_mrt - x_base) + (x_freq - x_base)
```

这个版本保持所有旧模块兼容，但论文主实验建议只启用：

```bash
--mrt_layers 1 --freq_layers 1 --fusion_mode parallel
```

不要同时启用 `freq_layers` 和 `sgf_layers`，否则频域分支内部仍然是串联多个频域模块，贡献会混杂。

## 5. 建议新增辅助函数

如果想让 `forward` 更清晰，可以在 `Model` 中加两个私有函数：

```python
def _run_mrt_branch(self, x):
    for mrt in self.mrt_blocks:
        x = mrt(x)
    return x

def _run_freq_branch(self, x):
    for freq in self.freq_blocks:
        x = freq(x)
    for freq_v2 in self.freq_v2_blocks:
        x = freq_v2(x)
    for freq_v3 in self.freq_v3_blocks:
        x = freq_v3(x)
    for freq_v4 in self.freq_v4_blocks:
        x = freq_v4(x)
    for sgf in self.sgf_blocks:
        x = sgf(x)
    return x
```

然后 `forward` 写成：

```python
if self.fusion_mode == 'serial':
    x = self._run_freq_branch(self._run_mrt_branch(x))
else:
    x_base = x
    x_mrt = self._run_mrt_branch(x_base)
    x_freq = self._run_freq_branch(x_base)
    x = x_base + (x_mrt - x_base) + (x_freq - x_base)
```

这个写法更适合长期维护。

## 6. 初始等价性检查

并联实现后，必须做一个快速 sanity check：

```python
model_base = CycleNet with mrt_layers=0, freq_layers=0
model_parallel = CycleNet with mrt_layers=1, freq_layers=1, fusion_mode=parallel
```

由于新增模块内部 `res_scale=0`，理论上并联模型初始输出应和 baseline 几乎一致，差异只来自参数初始化中 baseline backbone/cycleQueue 是否完全相同。

建议用固定 seed 构造同一输入，检查：

```text
parallel module delta norm should be 0 at initialization
```

更直接的检查方式是在 `forward` 中临时打印：

```python
print((x_mrt - x_base).abs().max(), (x_freq - x_base).abs().max())
```

初始应接近 0。

## 7. 实验矩阵

### 7.1 最小验证

先在 Solar 和 Weather 上跑：

```text
Baseline
MRT only
Freq only
MRT+Freq serial
MRT+Freq parallel
```

预测长度：

```text
96, 192, 336, 720
```

推荐命名：

```text
logs/parallel_solar/solar_mrt1_freq1_parallel_96.log
logs/parallel_weather/weather_mrt1_freq1_parallel_96.log
```

### 7.2 论文主表

如果最小验证有效，再扩展到：

```text
ETTh1, ETTh2, ETTm1, ETTm2, Weather, Solar, Electricity, Traffic
```

每个数据集至少比较：

```text
CycleNet
CycleNet + MRT
CycleNet + Freq
CycleNet + MRT + Freq serial
CycleNet + MRT + Freq parallel
```

FreDF 单独作为训练策略比较：

```text
Freq parallel without FreDF
Freq parallel with FreDF
```

不要只报告 `parallel + FreDF`，否则无法证明结构收益来自并联融合。

## 8. 预期结果与判定标准

并联结构成立的证据不是“某一个数据集涨点”，而是以下三点：

1. `parallel` 平均优于 `serial`。
2. `parallel` 至少不明显损害单模块强项，例如 Solar 上不能弱于 MRT/Freq 太多。
3. `parallel` 的退步次数少于 `serial`，尤其在 336/720 长预测上更稳定。

建议论文表述：

```text
Parallel fusion reduces interference between temporal trend extraction and spectral filtering.
```

避免过强表述：

```text
Parallel fusion always improves performance.
```

## 9. 风险点

### 9.1 并联可能仍然不如单模块

如果 `parallel` 仍然不稳定，说明 MRT 和 Freq 的目标频段虽不同，但模型容量或优化过程存在竞争。此时可以改成带门控融合：

```python
gate = torch.sigmoid(self.parallel_gate)  # shape [1] or [C,1]
x = x_base + gate * delta_mrt + (1 - gate) * delta_freq
```

但不建议第一版就上 gate，因为会增加审稿人质疑点。

### 9.2 FreDF 可能掩盖结构贡献

FreDF 在 ETT 系列上已经有明显收益，所以并联结构实验应先不用 FreDF。确认结构有效后，再叠加 FreDF。

### 9.3 SGF prior 需要修正

SGF 的 prior 必须只从 train split 计算。如果继续使用完整数据生成 prior，消融实验可能被质疑数据泄漏。

## 10. 推荐实施顺序

1. 在 `run.py` 新增 `--fusion_mode` 和 setting 后缀。
2. 在 `models/CycleNet.py` 新增 `self.fusion_mode`。
3. 把增强模块 forward 拆成 `_run_mrt_branch` 和 `_run_freq_branch`。
4. 实现 `serial` 和 `parallel` 两套路径。
5. 跑 Solar/Weather 的 5 组最小消融。
6. 如果并联优于串联，再扩展到全数据集。
7. 最后再测试 `parallel + FreDF`，作为训练策略叠加结果。

## 11. 推荐命令模板

Solar:

```bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_${pred_len} \
  --model CycleNet \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len ${pred_len} \
  --enc_in 137 \
  --cycle 144 \
  --model_type mlp \
  --use_revin 0 \
  --mrt_layers 1 \
  --freq_layers 1 \
  --fusion_mode parallel \
  --train_epochs 30 \
  --patience 15 \
  --itr 1 \
  --batch_size 64 \
  --learning_rate 0.01 \
  --random_seed 2024 \
  > logs/parallel_solar/solar_mrt1_freq1_parallel_${pred_len}.log 2>&1
```

Weather:

```bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_${pred_len} \
  --model CycleNet \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len ${pred_len} \
  --enc_in 21 \
  --cycle 144 \
  --model_type mlp \
  --use_revin 1 \
  --mrt_layers 1 \
  --freq_layers 1 \
  --fusion_mode parallel \
  --train_epochs 30 \
  --patience 15 \
  --itr 1 \
  --batch_size 256 \
  --learning_rate 0.005 \
  --random_seed 2024 \
  > logs/parallel_weather/weather_mrt1_freq1_parallel_${pred_len}.log 2>&1
```

## 12. 最终建议

第一版并联不要引入复杂 gate，也不要引入 SGF。只做：

```text
x = x_base + (MRT(x_base) - x_base) + (Freq(x_base) - x_base)
```

这版最干净，能直接回答论文问题：时域多尺度趋势增强和频域残差滤波是否互补。如果这版都不能稳定超过串联，那么问题不在融合形式，而在两个模块的适用数据分布不同，需要转向“数据集自适应选择模块”的叙事。
