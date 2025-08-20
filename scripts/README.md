# Graph-Mamba 批量实验脚本

本目录包含两个核心脚本，用于Graph-Mamba项目的大规模基准测试：

1. **`generate_cluster_cmds.py`** - 生成批量集群计算命令
2. **`aggregate_results.py`** - 聚合和分析实验结果

## 🚀 快速开始

### 1. 生成批量命令

```bash
# 在项目根目录运行
python scripts/generate_cluster_cmds.py

# 输出: commands.list (包含42条独立可执行的命令)
```

### 2. 聚合实验结果

```bash
# 聚合results/目录下的所有实验结果
python scripts/aggregate_results.py

# 输出到 outputs/ 目录:
# - benchmark_summary.csv: 详细结果记录
# - benchmark_pivot.csv: 透视表（模型×数据集）
```

## 📋 详细说明

### generate_cluster_cmds.py

**功能**: 自动扫描配置文件并生成42条集群计算命令（3个模型 × 14个数据集）

**支持的模型**:
- GPS
- Mamba  
- Exphormer_LRGB

**支持的数据集** (共14个):
- **分子回归**: `zinc`, `aqsol`, `qm9`
- **TU图分类**: `dd`, `proteins`, `colors3`, `mutagenicity`, `coildel`, `dblp`, `twitter`, `synthetic`
- **OGB数据集**: `molhiv`
- **LRGB数据集**: `peptides-func`, `peptides-struct`

**输出格式**:
```
python main.py --cfg configs/Benchmark/GPS/zinc-exported-GPS.yaml --repeat 1
python main.py --cfg configs/Benchmark/Mamba/zinc-exported-Mamba.yaml --repeat 1
python main.py --cfg configs/Benchmark/Exphormer_LRGB/zinc-exported-EX.yaml --repeat 1
# ... 共42条命令
```

**集群提交示例** (SLURM):
```bash
# 1. 生成命令列表
python scripts/generate_cluster_cmds.py

# 2. 创建SLURM作业脚本
cat > submit_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=graph_mamba_benchmark
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=1-42
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# 读取第N行命令并执行
command=$(sed -n "${SLURM_ARRAY_TASK_ID}p" commands.list)
echo "执行命令: $command"
eval $command
EOF

# 3. 提交作业数组
mkdir -p logs
sbatch submit_job.sh
```

### aggregate_results.py

**功能**: 从`results/`目录聚合所有实验结果，生成CSV分析文件

**数据来源优先级**:
1. `results/{experiment}/agg/val/best.json` (验证集最佳结果)
2. `results/{experiment}/agg/test/best.json` (测试集最佳结果，如果存在)
3. `results/{experiment}/{seed}/test/stats.json` (最后一轮测试结果)

**参数选项**:
```bash
# 默认设置
python scripts/aggregate_results.py

# 自定义输出目录
python scripts/aggregate_results.py --output-dir my_analysis

# 自定义结果目录
python scripts/aggregate_results.py --results-dir custom_results
```

**输出文件**:

#### 1. benchmark_summary.csv
包含所有实验的详细记录，每行一个实验结果：

| 列名 | 说明 |
|------|------|
| model | 模型名 (GPS, Mamba, Exphormer_LRGB, GatedGCN) |
| dataset | 数据集名 (zinc, molhiv等) |
| split | 数据集分割 (val, test) |
| epoch | 训练轮数 |
| primary_metric | 主要评估指标 |
| primary_value | 主要指标数值 |
| mae, accuracy, auc, ap | 具体指标值 |
| runtime | 每轮训练时间 |
| params | 模型参数数量 |

#### 2. benchmark_pivot.csv
透视表格式，便于对比分析：
- **行**: 数据集 + 主要指标 (如 `zinc_mae`, `molhiv_auc`)
- **列**: 模型名
- **值**: 对应的性能指标

**主要指标映射**:
```python
指标映射 = {
    # 回归任务 - MAE
    'zinc': 'mae', 'aqsol': 'mae', 'qm9': 'mae', 'peptides-struct': 'mae',
    
    # 分类任务 - Accuracy  
    'dd': 'accuracy', 'proteins': 'accuracy', 'colors3': 'accuracy',
    'mutagenicity': 'accuracy', 'coildel': 'accuracy', 'dblp': 'accuracy',
    'twitter': 'accuracy', 'synthetic': 'accuracy',
    
    # OGB任务 - AUC
    'molhiv': 'auc',
    
    # 多标签任务 - AP
    'peptides-func': 'ap'
}
```

## 📊 使用示例

### 完整工作流程

```bash
# 1. 生成命令并提交集群作业
python scripts/generate_cluster_cmds.py
# 提交到集群...

# 2. 等待所有作业完成后，聚合结果
python scripts/aggregate_results.py

# 3. 分析结果
python -c "
import pandas as pd
import matplotlib.pyplot as plt

# 读取结果
df = pd.read_csv('outputs/benchmark_summary.csv')

# 显示各模型在每个数据集上的表现
pivot = pd.read_csv('outputs/benchmark_pivot.csv', index_col=0)
print('模型性能对比:')
print(pivot)

# 可视化 (需要 matplotlib)
pivot.plot(kind='bar', figsize=(12, 8))
plt.title('Graph-Mamba Benchmark Results')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('benchmark_comparison.png')
print('图表已保存到 benchmark_comparison.png')
"
```

### 结果分析示例

```bash
# 查看汇总统计
python -c "
import pandas as pd
df = pd.read_csv('outputs/benchmark_summary.csv')

print('📊 实验汇总:')
print(f'总实验数: {len(df)}')
print(f'模型数量: {df[\"model\"].nunique()}')
print(f'数据集数量: {df[\"dataset\"].nunique()}')

print('\n🤖 各模型结果数量:')
print(df['model'].value_counts())

print('\n📈 按指标类型分组:')
print(df.groupby('primary_metric')['primary_value'].agg(['count', 'mean', 'std']))
"
```

## ⚠️ 注意事项

### 配置文件要求
- 所有配置文件必须存在于 `configs/Benchmark/{Model}/` 目录
- 文件命名格式: `{dataset}-exported-{Model}.yaml`
- Exphormer_LRGB 模型使用 `EX` 简写

### 结果文件要求  
- 实验结果保存在 `results/{experiment_name}/` 目录
- 支持的结果文件格式:
  - `agg/val/best.json` (首选)
  - `agg/test/best.json`  
  - `{seed}/test/stats.json`

### 依赖库
```bash
pip install pandas  # 用于结果聚合脚本
```

## 🛠️ 故障排除

### 命令生成失败
- **问题**: "找不到 main.py"
- **解决**: 确保在项目根目录运行脚本

- **问题**: "配置文件缺失"  
- **解决**: 检查配置文件是否存在，文件名是否正确

### 结果聚合失败
- **问题**: "没有找到任何有效的实验结果"
- **解决**: 检查 `results/` 目录是否存在结果文件

- **问题**: pandas导入错误
- **解决**: `pip install pandas`

### 透视表数据较少
这通常是因为多数实验只有验证集结果没有测试集结果。透视表默认优先显示测试集结果。

## 🔄 扩展和定制

### 添加新模型
1. 在 `generate_cluster_cmds.py` 的 `models` 列表中添加新模型名
2. 确保对应的配置文件存在于 `configs/Benchmark/{NewModel}/`

### 添加新数据集
1. 在 `generate_cluster_cmds.py` 的 `datasets` 列表中添加新数据集名
2. 在 `aggregate_results.py` 的 `get_dataset_metric_mapping()` 中添加对应的主要指标
3. 确保相应的配置文件存在

### 自定义指标
修改 `aggregate_results.py` 中的指标提取和映射逻辑以支持新的评估指标。

---

**创建时间**: 2025-01-20  
**版本**: 1.0  
**作者**: AI Assistant based on Graph-Mamba project requirements
