# Adapter MIMO-OFDM 中文说明

这个仓库目前完成的是论文路线的第一步：先把 MIMO-OFDM 信道估计的传统基线和数据生成流程跑通。现在还没有加入 CNN、Adapter、在线微调，目的是先保证后续深度学习实验的数据来源和传统算法对比是可靠的。

## 是否下载了新包

没有下载任何新包。

当前代码只使用你环境里已经存在的 Python 包：

```text
numpy
scipy
matplotlib
```

我已用下面命令检查过这些包可以导入：

```powershell
python -c "import numpy, scipy, matplotlib; print('deps ok')"
```

## 当前实现内容

当前代码实现了：

- 2x2 MIMO-OFDM 频域仿真
- Rayleigh / Rician 多径信道
- LTE 风格 EPA / EVA / ETU 功率延迟谱
- 3GPP TR 38.901 风格 TDL-A / TDL-B / TDL-C 功率延迟谱
- 正交导频观测
- LS 信道估计
- LS 后的频域线性插值
- Oracle LMMSE 频域估计
- NMSE 曲线绘制
- 后续 CNN/Adapter 可直接使用的数据集导出

## 目录结构

```text
Adapter_MIMO-OFDM/
├── README.md
├── README_zh.md
├── docs/
│   └── phase1_baselines.md
├── scripts/
│   ├── run_baselines.py
│   └── generate_dataset.py
└── src/
    └── adapter_mimo_ofdm/
        ├── __init__.py
        └── sim.py
```

各文件作用：

```text
src/adapter_mimo_ofdm/sim.py
```

核心仿真代码。信道生成、导频生成、LS、LMMSE、NMSE、数据导出都在这里。

```text
scripts/run_baselines.py
```

运行传统算法验证，输出 SNR-NMSE 表格、CSV 文件和曲线图。

```text
scripts/generate_dataset.py
```

生成后续训练神经网络需要的数据集，比如 `h_ls_ri -> h_true_ri`。

```text
docs/phase1_baselines.md
```

第一阶段实验说明，包括公式、baseline 定义和已经跑出的结果。

## 整体数据流

当前流程是：

```text
选择信道 profile
        ↓
生成多径信道 H_true
        ↓
在少量导频子载波上观测 Y_p
        ↓
得到导频位置 LS 估计 H_LS,p
        ↓
方法 1：线性插值得到全频域 H_LS
        ↓
方法 2：Oracle LMMSE 得到 H_LMMSE
        ↓
和 H_true 比较，计算 NMSE
```

后续 CNN/Adapter 会从这里接上：

```text
输入：h_ls_ri
标签：h_true_ri
```

## 核心仿真公式

MIMO-OFDM 每个子载波上的系统模型：

```text
Y[k] = H[k] X[k] + N[k]
```

其中：

- `X[k]` 是第 `k` 个子载波上的发送符号或导频；
- `Y[k]` 是接收信号；
- `H[k]` 是 MIMO 信道矩阵；
- `N[k]` 是复高斯噪声。

频域信道由 tapped-delay profile 生成：

```text
H[k] = sum_l h_l exp(-j 2 pi f_k tau_l)
```

其中：

- `h_l` 是第 `l` 条路径的复信道系数；
- `tau_l` 是该路径延迟；
- `f_k` 是子载波频率；
- 每条路径的平均功率由 EPA/EVA/ETU 或 TDL-A/B/C profile 给出。

NMSE 计算：

```text
NMSE = ||H_true - H_hat||^2 / ||H_true||^2
```

## 代码详细说明

### `SimConfig`

位置：[src/adapter_mimo_ofdm/sim.py](src/adapter_mimo_ofdm/sim.py)

这是仿真参数配置类，常用字段如下：

```text
n_frames             仿真帧数
n_tx                 发射天线数，默认 2
n_rx                 接收天线数，默认 2
n_subcarriers        OFDM 子载波数，默认 64
channel_profile      信道 profile，默认 tdl-a
subcarrier_spacing_hz 子载波间隔，默认 15 kHz
delay_spread_ns      TDL profile 的延迟扩展，默认 300 ns
pilot_ratio          导频比例，默认 1/4
snr_db               SNR 列表
channel_model        rayleigh 或 rician
rician_k             Rician K 因子
seed                 随机种子
```

### `resolve_delay_profile`

作用：把 profile 名称转换成“路径延迟 + 路径功率”。

支持：

```text
exponential
epa
eva
etu
tdl-a
tdl-b
tdl-c
```

EPA/EVA/ETU 使用固定 ns 延迟；TDL-A/B/C 使用归一化延迟，再乘以 `delay_spread_ns`。

### `generate_frequency_channel`

作用：生成真实频域 MIMO 信道 `H_true`。

输出形状：

```text
[batch, subcarrier, rx, tx]
```

默认 2x2 MIMO、64 子载波时：

```text
[样本数, 64, 2, 2]
```

这一步做的事情：

1. 根据 profile 生成每条路径的平均功率；
2. 对每个收发天线对生成复高斯路径系数；
3. Rayleigh 只使用 NLOS 随机路径；
4. Rician 会额外加入第 0 tap 的 LOS 分量；
5. 用频域公式把时延路径叠加成每个子载波上的 `H[k]`。

### `pilot_indices`

作用：根据导频比例选择导频子载波。

例如：

```text
n_subcarriers = 64
pilot_ratio = 0.25
```

则大约每 4 个子载波放一个导频，共 16 个导频子载波。

### `observe_orthogonal_pilots`

作用：模拟导频接收。

当前采用正交导频设计。对 2 发 2 收 MIMO 来说，可以理解为使用 2 个正交导频时隙，让每个发射天线的信道可以分开估计。

代码中简化为：

```text
H_LS,p = H_true,p + noise
```

这等价于导频矩阵是单位阵或正交矩阵时的 LS 观测结果。

### `interpolate_pilots_linear`

作用：LS 只能在导频子载波上得到估计，因此需要插值得到全部 64 个子载波的信道。

当前使用频域周期线性插值：

```text
H_LS,pilot → H_LS,all
```

这就是图里的 `LS + linear interp.`。

### `frequency_covariance`

作用：根据 profile 计算频域信道相关矩阵。

公式：

```text
R[k,m] = sum_l p_l exp(-j 2 pi (k-m) Delta_f tau_l)
```

其中：

- `p_l` 是第 `l` 条路径功率；
- `tau_l` 是路径延迟；
- `Delta_f` 是子载波间隔。

这个矩阵是 LMMSE 的关键。

### `lmmse_frequency_estimate`

作用：做频域 LMMSE 信道估计。

当前实现的是 oracle LMMSE，因为它知道真实 profile 和噪声方差：

```text
H_LMMSE = R_hp (R_pp + sigma_n^2 I)^(-1) H_LS,p
```

它适合作为传统算法强基线，但论文里要说明这是比较理想的 LMMSE 设置。

### `simulate_baselines`

作用：完整跑一组 SNR-NMSE 实验。

它会循环每个 SNR：

```text
生成 H_true
生成导频观测
计算 LS + 插值
计算 Oracle LMMSE
计算 NMSE
保存结果
```

### `complex_mimo_to_ri_channels`

作用：把复数 MIMO 信道转换成神经网络常用的实数张量。

原始复数形状：

```text
[batch, subcarrier, rx, tx]
```

转换后：

```text
[batch, 2*rx*tx, subcarrier]
```

默认 2x2 MIMO 时：

```text
[batch, 8, 64]
```

其中 8 个通道来自：

```text
2 个实虚部 × 2 个接收天线 × 2 个发射天线
```

这个就是后续 1D CNN 的输入格式。

### `generate_dataset`

作用：生成 `.npz` 数据集。

保存字段包括：

```text
h_true        真实复数信道，[samples, 64, 2, 2]
h_ls          LS 插值后的复数信道，[samples, 64, 2, 2]
h_pilot_ls    只在导频位置的 LS 信道，[samples, pilot_num, 2, 2]
h_true_ri     神经网络标签，[samples, 8, 64]
h_ls_ri       神经网络输入，[samples, 8, 64]
pilot_indices 导频子载波位置
snr_db         每个样本对应的 SNR
delays_sec     profile 路径延迟
powers_linear  profile 路径功率
```

## 如何运行

### 1. 运行传统基线

```powershell
python scripts/run_baselines.py --frames 2000 --profile tdl-a --pilot-ratio 0.25
```

输出：

```text
outputs/baselines/rayleigh_tdl-a_p0.25_n2000.csv
outputs/baselines/rayleigh_tdl-a_p0.25_n2000.png
```

### 2. 生成训练数据

```powershell
python scripts/generate_dataset.py --samples 5000 --profile tdl-a --out data/rayleigh_tdl_a_p025_train_5k.npz
```

### 3. 换成 LTE EVA profile

```powershell
python scripts/run_baselines.py --frames 2000 --profile eva --pilot-ratio 0.25
```

### 4. 换成 Rician 信道

```powershell
python scripts/run_baselines.py --frames 2000 --profile tdl-a --channel rician --rician-k 5
```

### 5. 改导频比例

```powershell
python scripts/run_baselines.py --frames 2000 --profile tdl-a --pilot-ratio 0.125
```

`0.125` 对应导频比例 `1/8`。

## 已经跑出的第一组结果

命令：

```powershell
python scripts/run_baselines.py --frames 2000 --profile tdl-a --pilot-ratio 0.25
```

结果：

```text
SNR(dB) | LS+linear NMSE(dB) | Oracle LMMSE NMSE(dB)
0       | -1.526             | -9.477
5       | -6.303             | -13.289
10      | -10.778            | -17.533
15      | -14.305            | -21.882
20      | -16.564            | -26.099
25      | -17.540            | -30.420
30      | -18.006            | -35.044
```

解释：

- LS 在高 SNR 处出现平台，是因为导频较稀疏，线性插值本身有误差；
- Oracle LMMSE 知道真实 PDP 和噪声方差，因此性能明显更强；
- 后续 CNN/Adapter 的目标，是在少导频条件下超过 LS，并尽量接近 LMMSE。

## 与顶刊/高水平论文实验的对齐方式

当前对齐点：

- 使用 MIMO-OFDM 系统模型；
- 使用标准化 tapped-delay-line power-delay profile；
- 使用 SNR-NMSE 曲线作为基础指标；
- 使用 LS 和 LMMSE 作为传统 baseline；
- 保留导频比例实验入口；
- 数据保存为复数信道和实虚拆分信道，便于接 CNN/Adapter。

后面继续增强时，可以再加：

- BER 曲线；
- Doppler / time-varying channel；
- 3GPP CDL channel；
- 更严格的 pilot pattern；
- CNN、DnCNN、ReEsNet、Transformer 类深度学习基线；
- Online Adapter 与 full fine-tuning 对比。

## 下一步

下一步建议做 Offline CNN：

```text
输入：h_ls_ri
输出：h_hat_ri
监督：h_true_ri
损失：MSE(h_hat_ri, h_true_ri)
```

先得到：

```text
LS vs Offline CNN vs Oracle LMMSE
```

这张 SNR-NMSE 图跑通以后，再加 Adapter 在线微调。

## 二维时频数据集

为了对齐 ChannelNet / DeepPilotDesign 这类论文代码，新增了二维时频网格数据生成脚本：

```powershell
python scripts/generate_grid_dataset.py --samples 5000 --profile tdl-a --num-pilots 48 --out data/grid_tdl_a_72x14_p48_5k.npz
```

默认设置：

```text
OFDM symbols = 14
Subcarriers = 72
Pilots = 8/16/24/36/48
MIMO = 2x2
Input = h_ls_grid_ri, shape [N, 8, 14, 72]
Label = h_true_grid_ri, shape [N, 8, 14, 72]
```

服务器上生成论文规模数据。当前默认先跑较少但更稳定的 `16 pilots`：

```bash
bash scripts/server_generate_grid_datasets.sh
```

如果要复现实验里原来的 `48 pilots`，可以显式指定：

```bash
PILOTS=48 bash scripts/server_generate_grid_datasets.sh
```

这些参数都可以在命令前覆盖：

```bash
PILOTS=24 TRAIN_SAMPLES=32000 VAL_SAMPLES=4000 TEST_SAMPLES=4000 bash scripts/server_generate_grid_datasets.sh
PILOTS=16 SNR_MIN=5 SNR_MAX=25 MAX_DOPPLER_HZ=120 bash scripts/server_generate_grid_datasets.sh
```

默认会生成：

```text
Train: 32000, Rayleigh TDL-A
Val: 4000, Rayleigh TDL-A
Test: 4000, Rayleigh TDL-A
Shift tests: TDL-B, TDL-C, Rician TDL-A
File names include p8/p16/p24/p36/p48, for example grid_p16_tdl_a_train_32000.npz
```

## 开源论文 CNN baseline

已加入三个论文对齐的 CNN baseline：

```text
srcnn      ChannelNet / DeepPilotDesign 中的 SRCNN 超分辨率模块
channelnet SRCNN + DnCNN，对齐 ChannelNet 的 SR + IR pipeline
reesnet    残差 CNN / ReEsNet 风格，对比更强的 residual estimator
```

服务器上训练这三个 baseline。默认读取 `16 pilots` 数据并保存验证集最优 checkpoint：

```bash
bash scripts/server_train_open_cnn_baselines.sh
```

训练方式按论文代码习惯拆开：

```text
SRCNN: 单独训练，学习从 LS 插值图恢复信道图
ChannelNet: 先加载 SRCNN best，再固定 SRCNN 训练 DnCNN 去噪模块
ReEsNet: 残差 CNN 端到端训练，作为强 residual estimator 对照
```

默认轮数：

```text
SRCNN_EPOCHS = 120
DNCNN_EPOCHS = 120
REESNET_EPOCHS = 120
CHANNELNET_FINE_TUNE_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 0
```

默认模型超参：

```text
SRCNN:
  9x9 Conv(8->64) + ReLU
  1x1 Conv(64->32) + ReLU
  5x5 Conv(32->8)
  lr = 1e-3

ChannelNet:
  stage 1: load SRCNN best checkpoint
  stage 2: freeze SRCNN, train DnCNN depth=8, hidden=64
  stage 3: joint fine-tune 20 epochs with lr=1e-4
  lr = 1e-3

ReEsNet:
  residual blocks = 8
  hidden channels = 64
  lr = 5e-4

Common:
  batch_size = 128
  optimizer = AdamW
  weight_decay = 1e-5
  scheduler = CosineAnnealingLR
  checkpoint = validation NMSE best
```

如果要加一个很短的 ChannelNet 联合微调阶段：

```bash
CHANNELNET_FINE_TUNE_EPOCHS=20 bash scripts/server_train_open_cnn_baselines.sh
```

如果只想跑 80 轮，并保持 20 轮早停：

```bash
PILOTS=16 EPOCHS=80 EARLY_STOPPING_PATIENCE=20 bash scripts/server_run_pilot_experiment.sh
```

如果要训练其他导频数量：

```bash
PILOTS=48 SRCNN_EPOCHS=100 DNCNN_EPOCHS=100 REESNET_EPOCHS=100 bash scripts/server_train_open_cnn_baselines.sh
```

也可以细调三个模型：

```bash
PILOTS=16 \
SRCNN_EPOCHS=120 SRCNN_LR=1e-3 \
DNCNN_EPOCHS=120 DNCNN_LR=1e-3 CHANNELNET_DEPTH=8 CHANNELNET_HIDDEN=64 \
REESNET_EPOCHS=120 REESNET_LR=5e-4 REESNET_DEPTH=8 REESNET_HIDDEN=64 \
BATCH_SIZE=128 NUM_WORKERS=2 \
bash scripts/server_train_open_cnn_baselines.sh
```

一键跑完整导频实验：

```bash
PILOTS=16 EPOCHS=120 bash scripts/server_run_pilot_experiment.sh
```

或者单独训练：

```bash
python scripts/train_grid_cnn.py --model srcnn --train data/grid/grid_p16_tdl_a_train_32000.npz --val data/grid/grid_p16_tdl_a_val_4000.npz --test data/grid/grid_p16_tdl_a_test_4000.npz
python scripts/train_channelnet_pipeline.py --train data/grid/grid_p16_tdl_a_train_32000.npz --val data/grid/grid_p16_tdl_a_val_4000.npz --test data/grid/grid_p16_tdl_a_test_4000.npz --srcnn-checkpoint outputs/p16/cnn_srcnn/srcnn_grid_best.pt
python scripts/train_grid_cnn.py --model reesnet --train data/grid/grid_p16_tdl_a_train_32000.npz --val data/grid/grid_p16_tdl_a_val_4000.npz --test data/grid/grid_p16_tdl_a_test_4000.npz
```

训练完成后，对比 LS、LMMSE、SRCNN、ChannelNet、ReEsNet 并画图：

```bash
bash scripts/server_compare_grid_methods.sh
```

其他导频数量同样用 `PILOTS` 指定：

```bash
PILOTS=48 bash scripts/server_compare_grid_methods.sh
```

输出：

```text
outputs/p16/comparison/grid_method_comparison.csv
outputs/p16/comparison/grid_method_comparison.json
outputs/p16/comparison/grid_method_comparison.png
```

该脚本默认比较：

```text
LS + 2D interpolation
Empirical 2D LMMSE (用训练集估计协方差，非 oracle)
Oracle 2D LMMSE
Mismatched 2D LMMSE (默认假设 TDL-A)
SRCNN
ChannelNet
ReEsNet
```
