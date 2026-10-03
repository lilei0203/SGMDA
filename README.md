# SGMDA — HMDD v3.2 五折交叉验证 Python 项目

这是依据用户提供的 SGMDA 论文编写的**新研究实现**。它实现了数据准备、每折重建特征、模型训练、RF 分类和重复五折评价；不是从原实验代码恢复的复现包，也不保证得到论文中的数值。当前按用户要求停止进一步测试，尚未完成真实数据上的完整五折训练。

**最重要的实现选择：常规随机森林无法反向传播。** 默认版本增加一个仅在神经训练时使用的可微线性评分层，为 Highway 提供真实梯度；最终预测仍使用 RF。这个评分层是本项目补足训练机制的改动，不是原稿已明确描述的组件。另提供不增加评分层的 `fixed_highway` 模式，但该模式的 Highway 是固定随机变换，不能称为学习到的特征交互。详见 [实现说明](docs/IMPLEMENTATION.md)。

## 数据已经放在哪里

- `data/raw/mdformer/`：公开论文 MDformer 的四个原始数据文件及下载清单。
- `data/processed/hmdd32.npz`：已准备的数据，内部方向统一为 **疾病 × miRNA**。
- `data/processed/hmdd32.provenance.json`：来源、固定提交号、SHA-256 和规模核验。

规模：**591 个疾病、853 个 miRNA、12,446 个正关联、491,677 个未标注对**。这是发表论文使用的 HMDD v3.2 派生基准子集，不是 HMDD v3.2 全部原始记录。实体名称和顺序保留来源文件，不擅自合并 precursor/mature miRNA 名称。

数据来自 [MDformer 作者仓库](https://github.com/Linda908/MDformer)，固定提交为 `07223704e2effd1de43689e840ce5365b4317a36`。只采用关联矩阵、实体名称和疾病语义相似性；不使用来源仓库中根据完整关联矩阵计算的 GIP、miRNA 功能相似性或图邻接矩阵。

## 快速运行

在本目录执行；推荐 Python 3.10 或更新版本。项目使用现代 PyTorch，不要求论文中的 PyTorch 1.1 / Ubuntu 16.04。

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 压缩包已带准备好的数据，通常无需再次下载/准备。
# 如果没有 data/processed/hmdd32.npz，执行以下两行：
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe -m sgmda prepare

# 正式配置：20 次独立运行，每次 5 折，每折 100 个 epoch
.\.venv\Scripts\python.exe -m sgmda run --config configs/default.json --output results/sgmda_20runs
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
python -m sgmda prepare
python -m sgmda run --config configs/default.json --output results/sgmda_20runs
```

现有 Codex 工作区已经具备依赖，可在本项目目录使用父目录解释器：

```powershell
..\.venv\Scripts\python.exe -m sgmda run --config configs/default.json --output results/sgmda_20runs
```

以上都是供后续执行的命令；本次交付不启动训练。

## 常用选项

```powershell
# 先跑一次完整五折（不是 20 次结果）
python -m sgmda run --output results/one_run --runs 1

# 有兼容 CUDA 的 PyTorch 时使用 GPU；RF 仍在 CPU 上训练
python -m sgmda run --output results/gpu_run --device cuda --threads 4

# 保存每折神经网络与 RF 模型（100 个 RF 可能占较多空间）
python -m sgmda run --output results/with_models --save-models

# 断点续跑：要求配置、数据、代码完全相同
python -m sgmda run --output results/sgmda_20runs --resume

# 无辅助评分层：仅 GAE 重建与对比损失训练，Highway 固定
python -m sgmda run --config configs/fixed_highway.json --output results/fixed_highway
```

`configs/smoke.json` 提供 2 epoch / 16 维 / 20 棵树的小配置，仅用于以后检查程序流程，不能冒充正式实验。输出路径应使用新的目录；旧的未完成尝试不应作为实验结果。

保存模型后，可以为指定疾病/miRNA 对打分：

```powershell
python scripts/predict_fold.py --experiment results/with_models --run 1 --fold 1 --pairs pairs.csv --output predictions.csv
```

输入 `pairs.csv` 的列名为 `disease,mirna`，名称须与数据实体表完全一致。输出会标明该样本是否参与所选折的训练；对训练样本打分不能当作外部验证。此脚本使用单折模型，不是全数据重训练或跨折集成。

## 每折具体做什么

1. 每次运行从原始未标注集合无放回抽取 12,446 对，标为采样负类，与全部正样本组合。
2. 使用固定随机种子的 `StratifiedKFold` 随机分层分成五折。同一运行保存一份样本及折分配，供基线复用。
3. 每折只把训练正关联写入 `A_train`；测试正关联为零，训练负样本与测试样本不重叠。
4. 从 `A_train` 重算 SVD、GIP（包括带宽）、MISIM 风格的 miRNA 功能相似性；融合疾病语义相似性并重建图。
5. 当前折重新初始化两个 GAE、Highway，以及默认模式的训练评分层。SVD 是固定 buffer，不参与梯度更新。
6. 默认用 GAE 图重建 BCE + 固定线性视图指向 GAE 视图的对比损失 + 训练评分层的关联 BCE 联合更新神经部分。
7. 神经训练完成后生成固定的训练与测试 pair features。独立训练 RF，再对测试样本预测概率。
8. 保存真实标签、概率、分类结果、训练损失和梯度诊断；评价阶段才使用测试标签。

全体已知实体可作为图节点，因此这是**pair-wise、transductive 5-CV**，不是实体冷启动评估。未知对是采样负类，不代表实验确认的生物学负关联。

## 指标如何汇总

- 每折报告 AUC、AP、梯形 PR-AUC、ACC、Precision、Recall、F1。
- 一个完整运行内，每个样本恰好得到一次 out-of-fold 预测；**拼接这五折 OOF 预测，再计算一组运行级指标**。
- 对 20 个完整运行的指标计算均值和样本标准差 `ddof=1`，不是对 100 折直接计算标准差。
- 主 AUPR 字段是 `AUPR_AP`，明确采用 `average_precision_score`；`PR_AUC_trapezoid` 同时另存，避免混淆。
- 分类阈值预先固定为 `score >= 0.5`；不使用外层测试数据选择阈值或 epoch。
- Precision/Recall/F1 针对正类，零分母时返回 0。
- `BCE_evaluation_only` 是最终 RF 概率的评价损失，不用于任何反向传播。
- 只有一次运行时标准差写为 `null` / “not estimable”，不填造假设数值。
- 不生成没有实际基线运行支持的显著性 P 值，也不内置论文目标分数。

## 输出结构

```text
results/your_run/
  manifest.json                 # 配置、数据/代码指纹、依赖版本、评价定义
  run_metrics.csv               # 每次完整五折的一组指标
  fold_metrics.csv              # 每折指标
  summary.json                  # 跨运行均值、真实 SD、完成状态
  RESULTS.md                    # 完成后可读的指标表
  run_01/
    split.npz                   # pairs, labels, fold；可供基线复用
    oof_predictions.csv         # 带实体名称的全部 OOF 预测
    metrics.json
    fold_1/
      training_associations.npz # 可检查测试关联已移除
      training_log.csv         # 各损失、Highway/GAE 梯度范数
      audit.json               # 特征指纹、样本计数、SVD 固定、Highway 参数变化
      predictions.npz
      complete.json            # 完成标记及关键产物校验值
      neural.pt                # 可选
      random_forest.joblib      # 可选
```

发生中断时，仅在五折全部完成后才报告该运行的指标。支持恢复已完成折，校验源代码、配置、数据和关键预测产物，拒绝混用不一致的结果。读取 pickle/joblib 模型时只使用可信的本地模型文件。

## 文件结构

| 文件 | 作用 |
|---|---|
| `sgmda/data.py` | 数据验证、名称映射、平衡采样、五折划分 |
| `sgmda/features.py` | 训练折内 SVD、GIP、MISIM、图构建 |
| `sgmda/model.py` | GAE、对比损失、Highway、训练评分层 |
| `sgmda/train.py` | 神经训练、梯度诊断、RF 拟合与预测 |
| `sgmda/metrics.py` | 指标及重复实验标准差 |
| `sgmda/experiment.py` | 完整实验、保存、审计、续跑 |
| `sgmda/cli.py` | 命令行接口 |
| `scripts/download_data.py` | 下载固定版本数据 |
| `scripts/predict_fold.py` | 使用保存的单折模型对指定样本打分 |
| `scripts/package_project.py` | 打包代码与数据，不带临时训练结果 |
| `tests/test_protocol.py` | 后续可执行的泄露、梯度、指标、续跑测试 |

验证范围和当前限制见 [VALIDATION_STATUS.md](docs/VALIDATION_STATUS.md)。数据出处和方法参考见 [SOURCES.md](docs/SOURCES.md)。默认超参数不是从已公布指标反向调出的最优参数，后续比较方法时应共享样本划分，并仅在训练部分内部调参。

