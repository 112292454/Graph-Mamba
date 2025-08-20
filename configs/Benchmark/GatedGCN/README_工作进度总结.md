# GatedGCN模型在导出数据集上的测试进度总结

## 项目概述

本项目旨在将TokenizerGraph项目中预处理好的14个数据集集成到Graph-Mamba框架中，并使用GatedGCN模型作为首个baseline进行全面测试。

## 已完成工作

### 1. 数据集集成 ✅
成功集成了以下14个数据集到Graph-Mamba框架：

#### 分子回归数据集 (3个)
- **QM9**: 量子化学分子属性预测 - ✅ 测试通过
  - **支持模式**: 单属性回归 (如homo, gap) 或 16维多回归 (all)
  - **测试结果**: HOMO属性 MAE=0.1863, R²=0.8057
- **ZINC**: 分子溶解度预测 (单属性回归) - ✅ 测试通过  
- **AQSOL**: 分子水溶性预测 (单属性回归) - ✅ 测试通过

#### TU图分类数据集 (8个)
- **DD**: 蛋白质结构分类 (二分类) - ✅ 测试通过
- **PROTEINS**: 蛋白质功能分类 (二分类) - ✅ 测试通过
- **COLORS3**: 图着色问题 (11分类) - ✅ 测试通过
- **MUTAGENICITY**: 化合物致突变性 (二分类) - ✅ 测试通过  
- **COILDEL**: 蛋白质螺旋结构 (100分类) - ✅ 测试通过
- **DBLP**: 学术网络分类 (二分类) - ✅ 测试通过
- **TWITTER**: 社交网络分类 (二分类) - ✅ 测试通过
- **SYNTHETIC**: 人工合成图 (二分类) - ✅ 测试通过

#### OGB数据集 (1个)  
- **MOLHIV**: HIV抑制剂预测 (二分类，ROC-AUC评估) - ✅ 测试通过

#### LRGB数据集 (2个)
- **PEPTIDES-FUNC**: 多标签功能预测 (10维多标签分类) - ✅ 测试通过
- **PEPTIDES-STRUCT**: 多目标结构预测 (11维多标签回归) - ✅ 测试通过

### 2. GatedGCN配置文件创建 ✅
为所有14个数据集创建了专门的GatedGCN配置文件，位于`configs/GatedGCN/`目录下：

- 统一使用20个epoch进行快速测试
- 正确配置了编码器类型（LinearNode/LinearEdge）
- 设置了合适的评估指标（accuracy/mae/auc/ap）
- 调优了batch size以适应不同数据集

### 3. 关键技术问题解决 ✅

#### 问题1: 编码器配置错误
- **问题**: 导出数据使用token特征，需要LinearNode/LinearEdge编码器
- **解决**: 统一配置所有数据集使用正确的编码器

#### 问题2: 数据集名称不匹配  
- **问题**: TUDataset中部分数据集名称不匹配（如DBLP_v1 vs dblp）
- **解决**: 修正配置文件中的数据集名称

#### 问题3: MOLHIV评估指标错误
- **问题**: 配置中使用rocauc指标，但系统只计算auc
- **解决**: 将metric_best改为auc

#### 问题4: AQSOL数据集格式错误
- **问题**: 格式配置为AQSOL，应为PyG-AQSOL
- **解决**: 修正数据集格式配置

#### 问题5: Peptides多标签数据的PyG collate问题 ⭐⭐⭐
- **问题**: PyG的collate函数将多维标签展平，导致维度不匹配错误
- **根本原因**: 
  ```
  # 期望: [batch_size, label_dim] -> [128, 10]
  # 实际: [batch_size * label_dim] -> [1280]
  ```
- **解决方案**: 在ExportedDataset类中添加后处理逻辑
  ```python
  # 修复多维标签被展平的问题
  first_label = pyg_data_list[0].y
  if first_label.dim() > 0 and len(first_label) > 1:
      # 重新reshape并更新slices
      num_samples = len(pyg_data_list)
      label_dim = len(first_label)
      self.data.y = self.data.y.view(num_samples, label_dim)
      self.slices['y'] = torch.arange(0, num_samples + 1)
  ```

#### 问题6: QM9数据集的多属性支持 ⭐⭐
- **需求**: 支持QM9的16个量子化学属性的灵活选择
- **实现方案**: 
  1. **单属性回归**: 通过数据集名称指定 (如 `QM9-homo`, `QM9-gap`)
  2. **多属性回归**: 使用 `QM9-all` 进行16维回归，包含z-score标准化
  3. **默认行为**: 不指定时默认使用 `homo` 属性
- **技术实现**:
  ```python
  # 根据数据集名称解析属性
  if self.qm9_target == 'all':
      # 16维多回归 + z-score标准化
      qm9_properties = ['mu', 'alpha', 'homo', 'lumo', 'gap', ...]
      y = torch.tensor([label[k] for k in qm9_properties])
      # 标准化处理
      self.data.y = (self.data.y - self.data.y.mean()) / self.data.y.std()
  else:
      # 单属性回归
      y = torch.tensor([label[self.qm9_target]])
  ```
- **配置示例**:
  - `QM9-homo`: HOMO轨道能量回归
  - `QM9-gap`: HOMO-LUMO能隙回归  
  - `QM9-all`: 16个属性的多任务回归

### 4. 测试结果汇总 ✅

#### 成功运行的数据集 (14/14)
| 数据集 | 类型 | 指标 | 状态 | 备注 |
|-------|------|------|------|------|
| DD | 二分类 | accuracy | ✅ | 已验证可正常训练 |
| PROTEINS | 二分类 | accuracy | ✅ | 完整训练完成 |
| ZINC | 回归 | mae | ✅ | - |
| AQSOL | 回归 | mae | ✅ | - |
| COLORS3 | 多分类 | accuracy | ✅ | - |
| MUTAGENICITY | 二分类 | accuracy | ✅ | - |
| COILDEL | 多分类 | accuracy | ✅ | - |
| DBLP | 二分类 | accuracy | ✅ | - |
| TWITTER | 二分类 | accuracy | ✅ | - |
| SYNTHETIC | 二分类 | accuracy | ✅ | - |
| MOLHIV | 二分类 | auc | ✅ | **AUC = 0.6989** |
| PEPTIDES-FUNC | 多标签分类 | ap | ✅ | 解决了collate问题 |
| PEPTIDES-STRUCT | 多标签回归 | mae | ✅ | - |
| QM9-HOMO | 单属性回归 | mae | ✅ | **MAE=0.1863, R²=0.8057** |

#### 已完全解决 ✅
所有14个数据集均测试通过，支持率：**14/14 (100%)**

## 下一步计划

### 阶段1: Baseline模型测试
按照GatedGCN的成功经验，逐个测试其他baseline模型：

1. **GPS** (`configs/Benchmark/GPS/`)
2. **Mamba** (`configs/Benchmark/Mamba/`)  
3. **Exphormer_LRGB** (`configs/Benchmark/Exphormer_LRGB/`)

**测试策略**:
- 复制GatedGCN的配置模板
- 调整模型特定参数
- 使用20 epochs进行快速验证
- 解决遇到的技术问题

### 阶段2: 大规模正式测试
所有模型验证通过后：
- 将epoch恢复到合理数值（100-500）
- 进行多次重复实验
- 收集完整的性能指标
- 绘制对比表格

### 阶段3: 结果分析与报告
- 模型性能对比分析
- 数据集特性分析
- 技术总结报告

## 重要注意事项

### Peptides数据集处理差异 ⚠️
- 我们的导出版本对Peptides数据集做了特殊处理（token化）
- 效果可能不如原版，但保证了一致性
- 在后续测试其他模型时，建议保留两个版本进行对比

### 技术经验总结
1. **编码器选择**: 导出数据统一使用LinearNode/LinearEdge
2. **标签维度**: 多维标签需要特别处理PyG的collate行为
3. **评估指标**: 确保配置文件中的指标与系统实际计算的一致
4. **数据集名称**: 严格按照数据集支持的名称进行配置

---
---
**文档创建时间**: 2025-08-20  
**GatedGCN测试完成度**: 14/14 数据集 (100%) ✅  
**下一目标**: 完成4个baseline模型的全数据集测试

## QM9数据集使用示例

### 1. 单属性回归任务
```yaml
# configs/GatedGCN/qm9-homo-exported-GatedGCN.yaml
dataset:
  format: Exported-QM9
  name: QM9-homo  # 使用HOMO属性
```

### 2. 多属性回归任务
```yaml
# configs/GatedGCN/qm9-all-exported-GatedGCN.yaml  
dataset:
  format: Exported-QM9
  name: QM9-all  # 使用所有16个属性
share:
  dim_out: 16  # 输出维度设为16
```

### 3. 支持的QM9属性
- **分子性质**: mu(偶极矩), alpha(极化率), homo, lumo, gap
- **热力学性质**: r2, zpve, u0, u298, h298, g298, cv  
- **原子化性质**: u0_atom, u298_atom, h298_atom, g298_atom

## 批量集群提交与结果聚合（下一步）

### 1) 生成批量命令（用于集群调度）
- 目标: 生成 42 条独立可运行的指令（3 模型 × 14 数据集，每行一条），便于提交到集群队列。
- 指令格式（示例）:
```
python main.py --cfg configs/Benchmark/GPS/zinc-exported-GPS.yaml --repeat 1
python main.py --cfg configs/Benchmark/Mamba/zinc-exported-Mamba.yaml --repeat 1
python main.py --cfg configs/Benchmark/Exphormer_LRGB/zinc-exported-EX.yaml --repeat 1
# ... 其余数据集同理，共 14 × 3 行
```
- 计划提供脚本: `scripts/generate_cluster_cmds.py`
  - 读取 Benchmark 目录结构，自动生成 42 行命令，输出到 `commands.list`。

### 2) 结果聚合与分析产出 CSV
- 目标: 从 `results/<run-name>/<seed>/` 或 `results/<run-name>/agg/` 聚合不同模型、不同数据集的关键指标，形成便于绘图与表格对比的 CSV。
- 行列设计:
  - 行: 一条运行记录（或聚合记录）
  - 列: `model`, `dataset`, `metric`, `split`(val/test), `value`, `epoch`, `runtime`
- 指标映射建议:
  - 回归任务: `mae`
  - 二/多分类: `accuracy`
  - OGB-MOLHIV: `auc`
  - 多标签: `ap`
- 计划提供脚本: `scripts/aggregate_results.py`
  - 自动扫描 `results/`，解析 `agg/` 或最后一轮 `val/test` 日志，输出 `benchmark_summary.csv`
  - 可选: 生成按“横轴=模型、纵轴=数据集、单元格=指标”的透视表 CSV

### 3) 统一规范
- 所有命令要求 GPU 环境；非 CUDA 环境直接报错退出。
- 下游正式实验将把 epoch 恢复至 100~500，并提供多次重复的统计结果。

