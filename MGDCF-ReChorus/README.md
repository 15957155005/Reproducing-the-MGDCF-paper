# MGDCF 复现（基于 ReChorus）

> 机器学习理论课期末大作业：复现论文 **MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering**（TKDE, arXiv:2204.02338），
> 在 [ReChorus](https://github.com/THUwangcy/ReChorus) 2.0 框架中实现该算法，并与框架内同类算法（BPRMF / NeuMF / LightGCN）在两个框架数据集（Grocery_and_Gourmet_Food、MovieLens-1M）上对比。

## 目录结构

```
MGDCF-ReChorus/
├── data/                         # 数据集（Top-k 推荐格式）
│   ├── Grocery_and_Gourmet_Food/ # 框架内置，可直接使用
│   └── MovieLens_1M/             # 由 prepare_ml1m.py 生成
├── src/                          # ReChorus 框架
│   └── models/general/
│       ├── MGDCF.py              # ★ 本作业新增：Hetero-MGDCF（MGDN + InfoBPR）
│       ├── BPRMF.py / NeuMF.py / LightGCN.py   # 基线
├── scripts/
│   ├── run_experiments.py        # 批量运行基线 + 本文模型，汇总 results/summary.csv
│   ├── run_grid.py               # 扩散步数 K / 是否加自环 的网格搜索
│   ├── run_sensitivity.py        # α / β / K 敏感性实验，汇总 results/sensitivity.csv
│   ├── plot_results.py           # 生成报告用图片
│   └── make_report.py            # 依据实验结果生成中文报告（docx）
├── results/                      # 实验结果与图片
├── log/                          # 训练日志
└── README.md
```

## 环境

```
Python 3.10+（实测 3.13）
torch 2.x（CPU 或 CUDA 均可）
numpy / pandas / scipy / scikit-learn / tqdm
matplotlib（仅画图用）
```

安装：

```bash
pip install torch pandas scipy scikit-learn tqdm matplotlib
```

## 数据准备

- `Grocery_and_Gourmet_Food`：仓库已包含（12.2 万条训练交互）。
- `MovieLens-1M`：按框架自带 `data/MovieLens_1M/MovieLens-1M.ipynb` 的规则处理（评分≥4 视为正反馈、5-core 过滤、按天 8:1:1 划分、每条例样本采样 99 个负例）：

```bash
cd data/MovieLens_1M
python prepare_ml1m.py     # 首次运行会自动下载 ml-1m.zip
```

处理后数据集统计：

| 数据集 | 用户数 | 物品数 | 训练交互 | 测试样本 | 负采样 |
|---|---|---|---|---|---|
| Grocery_and_Gourmet_Food | 12,801 | 12,801 | 121,892 | 14,681 | 99 |
| MovieLens-1M | 6,032 | 3,125 | 568,731 | 2,877 | 99 |

> 具体数值以 `prepare_ml1m.py` 输出与 `log/*.txt` 中的 `# user / # item` 为准。

## 复现的模型

| 配置名 | 实现 | 与论文的对应关系 |
|---|---|---|
| `BPRMF` | 框架自带 BPR-MF | MF（BPR 单负样本） |
| `NeuMF` | 框架自带 NeuMF | NCF |
| `LightGCN` | 框架自带 LightGCN（`n_layers=3`） | 马尔可夫图扩散的特例（α=β=1） |
| `APPNP` | `MGDCF.py`，`--loss_type BPR` | MGDN + BPR，即 APPNP（α=0.1, β=0.9, K=4） |
| `MF-InfoBPR` | `MGDCF.py`，`--n_layers 0` | 式(27)：K=0 时 MGDN 退化为 MF |
| `Hetero-MGDCF` | `MGDCF.py`，`--loss_type InfoBPR` | 论文主模型：Hetero-MGDN + InfoBPR |

### 与论文公式的对应（`src/models/general/MGDCF.py`）

| 论文 | 代码 |
|---|---|
| 式(3) Â = D̃^{-1/2}(A+I)D̃^{-1/2} | `MGDCFBase.build_adjmat(..., self_loop=True)` |
| 式(6)(7) H⁽ᵏ⁾ = βÂH⁽ᵏ⁻¹⁾ + αH⁽⁰⁾ | `MGDCFBase._propagate()` |
| 式(8) Z = H⁽ᴷ⁾/Γ | `return h / self.gamma` |
| 式(22) InfoBPR 多负样本 softmax 损失 | `MGDCFBase.loss()`，`--num_neg` |
| 式(27) K=0 时退化为 MF | `--n_layers 0` |

## 运行实验

```bash
# 全部配置 × 两个数据集（默认 CPU 4 线程/任务，6 个任务并行）
python scripts/run_experiments.py --datasets Grocery_and_Gourmet_Food MovieLens_1M --jobs 6 --epoch 100

# 只跑某个配置
python scripts/run_experiments.py --datasets Grocery_and_Gourmet_Food --only Hetero-MGDCF

# 扩散步数 K / 自环 的网格搜索（用于确定每个数据集的最优 K）
python scripts/run_grid.py --dataset MovieLens_1M --jobs 5 --epoch 100

# 超参数敏感性（α / β / K）
python scripts/run_sensitivity.py --dataset Grocery_and_Gourmet_Food --jobs 6 --epoch 60

# 只重新解析日志生成 summary
python scripts/run_experiments.py --summary

# 画图
python scripts/plot_results.py
```

单条命令示例（等价于上表某一配置）：

```bash
cd src
python main.py --model_name MGDCF --dataset Grocery_and_Gourmet_Food --path ../data/ \
  --emb_size 64 --n_layers 4 --alpha 0.1 --beta 0.9 --loss_type InfoBPR --num_neg 8 \
  --lr 1e-3 --l2 0 --batch_size 2048 --num_workers 0 \
  --topk 20 --metric NDCG,HR --main_metric NDCG@20 --early_stop 10 --epoch 100
```

## 实现说明与对框架的改动

1. **新增文件**：`src/models/general/MGDCF.py`（模型注册方式与框架一致，`models/general/__init__.py` 自动加载）。
2. **`LightGCN.py` 小改**：原实现将稀疏邻接矩阵硬编码为 `.cuda()`，在纯 CPU 环境会报错；改为按其 `device` 放置，并换成 CSR 稀疏张量。模型逻辑未变。
3. **CPU 上的稀疏乘法优化**：PyTorch 对稀疏矩阵乘法的自动求导在 CPU 上非常慢（ML-1M 上单次传播约 0.65 s）。我们在 `src/utils/layers.py` 中新增了 `SparseMM`（显式传入转置矩阵，使前向/反向都是普通 CSR×稠密乘法），MGDCF 与 LightGCN 均改用它，单 epoch 时间从约 190 s 降到约 13 s，数值结果完全一致。
4. **InfoBPR 的负样本数**：框架的 `--num_neg` 控制每条正样本的负例个数，InfoBPR 取 `--num_neg 8`；`num_neg=1` 时 InfoBPR 严格等于 BPR。
5. **L2 正则**：论文对输出表示 `Z` 施加 `Ψ_L2·||Z||²`；为了与框架内基线口径一致，本实现统一使用优化器的 weight decay（`--l2`）作为正则项（主实验取 0，与论文"改进来自优化而非正则"的设定一致）。
6. **评测协议**：沿用 ReChorus 的 Top-k 协议（每条测试样本 1 正 + 99 个采样负例，NDCG@20 / HR@20）。因此数值与论文中"全量排序"结果不可直接比较，但与框架内基线在同一协议下可比。

## 主要结果

测试集 NDCG@20 / HR@20（单种子 seed=0，99 负例采样评测）：

| 模型 | Grocery NDCG@20 | Grocery HR@20 | ML-1M NDCG@20 | ML-1M HR@20 |
|---|---|---|---|---|
| BPRMF | 0.3012 | 0.5620 | 0.3512 | 0.7317 |
| NeuMF | 0.2887 | 0.5487 | 0.3628 | 0.7344 |
| LightGCN | 0.3299 | 0.6164 | 0.3550 | 0.7331 |
| APPNP（MGDN + BPR） | 0.3314 | 0.6151 | 0.3508 | 0.7313 |
| MF + InfoBPR | 0.2895 | 0.5464 | **0.3659** | **0.7511** |
| **Hetero-MGDCF** | **0.3333** | **0.6186** | 0.3492 | 0.7216 |

结论要点：

- Grocery（较稀疏的图）上 Hetero-MGDCF 全面最优，比最强基线 LightGCN 提升约 1.0%；
- ML-1M（稠密图）上增益主要来自 InfoBPR（MF：0.3512 → 0.3659，+4.2%），
  图传播在该数据上不再带来额外收益，与论文表 2 中 MF-InfoBPR 与 Hetero-MGDCF 几乎持平的现象一致；
- 扩散步数 K 是最敏感的超参数：两个数据集均以 K=2 最优，K=4 时明显过平滑。

完整日志见 `log/`，汇总见 `results/summary.csv` 与 `results/sensitivity.csv`，图片见 `results/figures/`。
