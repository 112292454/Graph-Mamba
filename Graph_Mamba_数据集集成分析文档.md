# Graph-Mamba 数据集集成分析文档

## 数据集集成机制概述

Graph-Mamba项目采用分层的数据集集成机制，支持多种数据集格式和来源。通过统一的数据加载接口，可以方便地集成新的数据集。

## 现有数据集支持

### 1. 按格式分类

#### A. PyG原生数据集 (`format: PyG-{DatasetClass}`)
- **GNNBenchmarkDataset**: MNIST, CIFAR10
- **TUDataset**: DD, NCI1, ENZYMES, PROTEINS, IMDB系列, COLLAB
- **ZINC**: subset/full版本的分子数据集
- **Amazon**: photo, computers (节点分类)
- **Coauthor**: physics, cs (节点分类)
- **Planetoid**: 引用网络数据集
- **WikipediaNetwork**: 维基百科网络数据

#### B. OGB数据集 (`format: OGB`)
- **图属性预测**: ogbg-* 系列
- **节点分类**: ogbn-arxiv, ogbn-proteins
- **链接预测**: ogbl-* 系列
- **特殊数据集**: PCQM4Mv2系列

#### C. 自定义数据集 (`format: PyG-{CustomClass}`)
- **超像素数据集**: VOCSuperpixels, COCOSuperpixels
- **分子数据集**: Peptides (functional/structural), AQSOL
- **恶意软件检测**: MalNetTiny
- **接触预测**: PCQM4Mv2Contact

### 2. 数据集配置示例

```yaml
# TUDataset示例 (ENZYMES)
dataset:
  format: PyG-TUDataset
  name: ENZYMES
  task: graph
  task_type: classification

# ZINC数据集示例
dataset:
  format: PyG-ZINC  
  name: subset  # 或 full
  task: graph
  task_type: regression

# OGB数据集示例
dataset:
  format: OGB
  name: ogbg-molhiv
  task: graph
  task_type: classification_binary
```

## 数据集集成架构

### 核心组件层次
```
数据集集成架构
├── 配置层 (.yaml files)
│   ├── dataset.format: 指定数据集格式
│   ├── dataset.name: 指定具体数据集名称
│   └── 其他数据集特定配置
├── 路由层 (master_loader.py)
│   ├── load_dataset_master(): 主路由函数
│   ├── 格式识别与分发
│   └── 预处理流程调度
├── 预处理层 (preformat_* functions)
│   ├── 数据集特定预处理
│   ├── 特征工程
│   └── 数据分割
└── 标准化层
    ├── 统一数据接口
    ├── 位置编码集成
    └── 增强处理
```

### 关键文件说明

#### 1. 主加载器 (`graphgps/loader/master_loader.py`)
```python
@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir):
    """统一数据集加载入口"""
    if format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        if pyg_dataset_id == 'TUDataset':
            dataset = preformat_TUDataset(dataset_dir, name)
        elif pyg_dataset_id == 'ZINC':
            dataset = preformat_ZINC(dataset_dir, name)
        # ... 其他PyG数据集
    elif format == 'OGB':
        # OGB数据集处理
    # ... 后续标准化处理
```

#### 2. 数据集特定预处理函数
每个数据集类型都有对应的预处理函数，例如：
- `preformat_TUDataset()`: TU数据集预处理
- `preformat_ZINC()`: ZINC数据集预处理  
- `preformat_Peptides()`: 肽数据集预处理

#### 3. 自定义数据集类
位于 `graphgps/loader/dataset/` 目录，实现特定数据集的加载逻辑：
- `peptides_functional.py`: 功能性肽数据集
- `coco_superpixels.py`: COCO超像素数据集
- `malnet_tiny.py`: MalNet恶意软件数据集

## 数据处理流程

### 标准数据处理管道
```
原始数据
    ↓
[1] 数据集加载 (dataset-specific loader)
    ↓  
[2] 基础预处理 (preformat_* function)
    ├── 特征类型转换
    ├── 数据清洗
    └── 基础增强
    ↓
[3] 位置编码计算 (compute_posenc_stats)
    ├── Laplacian PE
    ├── Random Walk SE  
    ├── SignNet
    └── 其他PE方法
    ↓
[4] 图结构增强 (可选)
    ├── 扩展边 (expander edges)
    ├── 距离特征
    └── 有效阻抗
    ↓
[5] 数据分割 (prepare_splits)
    ├── 训练/验证/测试分割
    └── 分割索引设置
    ↓
标准化的PyG数据集
```

### TUDataset集成案例分析

#### 1. 配置文件设置
```yaml
dataset:
  format: PyG-TUDataset
  name: ENZYMES  # 或其他TU数据集名称
  task: graph
  task_type: classification
```

#### 2. 预处理函数实现
```python
def preformat_TUDataset(dataset_dir, name):
    """TUDataset预处理"""
    # 1. 确定预变换函数
    if name in ['DD', 'NCI1', 'ENZYMES', 'PROTEINS']:
        func = None  # 无需特殊预处理
    elif name.startswith('IMDB-') or name == "COLLAB":
        func = T.Constant()  # 添加常数特征
    else:
        raise ValueError(f"不支持的TUDataset: {name}")
    
    # 2. 加载数据集
    dataset = TUDataset(dataset_dir, name, pre_transform=func)
    return dataset
```

#### 3. 支持的TUDataset
- **有原始特征**: DD, NCI1, ENZYMES, PROTEINS
- **需要特征补充**: IMDB系列, COLLAB (添加常数特征)

### ZINC数据集集成案例

#### 1. 配置示例
```yaml
dataset:
  format: PyG-ZINC
  name: subset  # 或 full
  task: graph  
  task_type: regression
```

#### 2. 预处理实现
```python
def preformat_ZINC(dataset_dir, name):
    """ZINC数据集预处理"""
    if name not in ['subset', 'full']:
        raise ValueError(f"ZINC数据集版本错误: {name}")
    
    # 合并训练、验证、测试分割
    dataset = join_dataset_splits([
        ZINC(root=dataset_dir, subset=(name == 'subset'), split=split)
        for split in ['train', 'val', 'test']
    ])
    return dataset
```

## 集成新数据集的完整指南

### 方法一: 直接使用PyG原生数据集 ⭐ (推荐，最简单)

**适用场景**: 数据集已被PyTorch Geometric原生支持

**步骤**:
1. **检查支持性**: 确认数据集在PyG中可用
2. **添加路由**: 在`master_loader.py`中添加处理分支
3. **编写预处理**: 创建对应的`preformat_*`函数
4. **配置文件**: 创建yaml配置文件

**难度评估**: ⭐ (非常简单)

**示例**: 集成新的TUDataset
```python
# 在master_loader.py的PyG-TUDataset分支中添加
elif pyg_dataset_id == 'TUDataset':
    dataset = preformat_TUDataset(dataset_dir, name)

def preformat_TUDataset(dataset_dir, name):
    # 添加新数据集支持
    if name == 'NEW_DATASET':
        func = T.OneHotDegree()  # 根据需要选择预处理
    # ... 现有代码
    dataset = TUDataset(dataset_dir, name, pre_transform=func)
    return dataset
```

### 方法二: 扩展已有数据集类 ⭐⭐

**适用场景**: 数据集格式与现有支持的类似

**步骤**:
1. **识别相似数据集**: 找到格式相近的已支持数据集
2. **修改预处理**: 扩展对应的预处理函数
3. **测试兼容性**: 确保数据格式匹配
4. **配置调优**: 调整超参数配置

**难度评估**: ⭐⭐ (简单)

### 方法三: 创建自定义数据集类 ⭐⭐⭐

**适用场景**: 数据集有特殊格式或处理需求

**完整实现步骤**:

#### 步骤1: 创建数据集类
```python
# graphgps/loader/dataset/your_dataset.py
import os.path as osp
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.io import read_txt_array

class YourDataset(InMemoryDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None):
        self.split = split
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def processed_file_names(self):
        return [f'{self.split}.pt']
    
    def process(self):
        # 实现数据处理逻辑
        data_list = []
        # ... 数据加载和预处理代码
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
```

#### 步骤2: 添加预处理函数
```python
# 在master_loader.py中添加
def preformat_YourDataset(dataset_dir, name):
    """你的数据集预处理"""
    dataset = join_dataset_splits([
        YourDataset(root=dataset_dir, split=split)
        for split in ['train', 'val', 'test']
    ])
    return dataset
```

#### 步骤3: 集成到主加载器
```python
# 在master_loader.py的load_dataset_master函数中添加
elif pyg_dataset_id == 'YourDataset':
    dataset = preformat_YourDataset(dataset_dir, name)
```

#### 步骤4: 创建配置文件
```yaml
# configs/Mamba/your-dataset-EX.yaml
out_dir: results
metric_best: accuracy  # 根据任务类型调整
dataset:
  format: PyG-YourDataset
  name: your_dataset_name
  task: graph  # 或 node
  task_type: classification  # 或 regression
  # 其他配置...
```

**难度评估**: ⭐⭐⭐ (中等)

### 方法四: 集成外部数据集 ⭐⭐⭐⭐

**适用场景**: 使用DGL或其他框架的数据集

**主要挑战**:
1. **数据格式转换**: DGL → PyG格式转换
2. **依赖管理**: 处理不同框架的依赖冲突  
3. **特征对齐**: 确保节点/边特征格式兼容
4. **分割策略**: 适配不同的数据分割方式

**转换函数示例**:
```python
def dgl_to_pyg(dgl_graph):
    """DGL图转PyG格式"""
    import dgl
    from torch_geometric.data import Data
    
    # 获取边列表
    src, dst = dgl_graph.edges()
    edge_index = torch.stack([src, dst], dim=0)
    
    # 转换节点特征
    x = dgl_graph.ndata.get('feat', None)
    
    # 转换边特征
    edge_attr = dgl_graph.edata.get('feat', None)
    
    # 获取标签
    y = dgl_graph.ndata.get('label', None)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
```

**难度评估**: ⭐⭐⭐⭐ (较难)

## 具体集成示例

### 示例1: 集成新的TUDataset (MUTAG)

#### 1. 修改预处理函数
```python
def preformat_TUDataset(dataset_dir, name):
    if name in ['DD', 'NCI1', 'ENZYMES', 'PROTEINS', 'MUTAG']:  # 添加MUTAG
        func = None
    elif name.startswith('IMDB-') or name == "COLLAB":
        func = T.Constant()
    else:
        ValueError(f"Loading dataset '{name}' from TUDataset is not supported.")
    dataset = TUDataset(dataset_dir, name, pre_transform=func)
    return dataset
```

#### 2. 创建配置文件
```yaml
# configs/Mamba/mutag-EX.yaml
out_dir: results
metric_best: accuracy
dataset:
  format: PyG-TUDataset
  name: MUTAG
  task: graph
  task_type: classification
  node_encoder: True
  node_encoder_name: Atom
  edge_encoder: True  
  edge_encoder_name: Bond
# ... 其他配置
```

### 示例2: 集成自己的ZINC变体

#### 1. 创建预处理函数
```python
def preformat_ZINC_Custom(dataset_dir, name):
    """自定义ZINC数据集变体"""
    if name == 'custom_subset':
        # 自定义子集逻辑
        dataset = ZINC(root=dataset_dir, subset=True, split='train')
        # 可以添加自定义过滤逻辑
        # dataset = filter_dataset(dataset, criteria)
    else:
        raise ValueError(f"Unsupported ZINC variant: {name}")
    return dataset
```

#### 2. 集成到主加载器
```python
elif pyg_dataset_id == 'ZINC_Custom':
    dataset = preformat_ZINC_Custom(dataset_dir, name)
```

## 配置系统深度解析

### 关键配置参数

#### 数据集基础配置
```yaml
dataset:
  format: PyG-TUDataset          # 数据集格式标识
  name: ENZYMES                  # 具体数据集名称
  task: graph                    # 任务类型: graph/node/edge
  task_type: classification      # 具体任务: classification/regression
  transductive: False            # 是否为直推式学习
  split_mode: standard          # 数据分割模式
```

#### 编码器配置
```yaml
dataset:
  node_encoder: True            # 是否启用节点编码器
  node_encoder_name: Atom       # 节点编码器类型
  node_encoder_bn: False        # 节点编码器批归一化
  edge_encoder: True            # 是否启用边编码器
  edge_encoder_name: Bond       # 边编码器类型
  edge_encoder_bn: False        # 边编码器批归一化
```

#### 位置编码配置
```yaml
posenc_LapPE:                   # Laplacian位置编码
  enable: True
  eigen:
    laplacian_norm: sym         # 拉普拉斯归一化: none/sym/rw
    eigvec_norm: L2            # 特征向量归一化
    max_freqs: 10              # 最大特征值数量
  model: DeepSet               # 编码模型: DeepSet/Transformer
  dim_pe: 16                   # 编码维度
  layers: 2                    # 编码层数
```

### 编码器类型说明

#### 节点编码器选项
- **Atom**: 原子类型编码 (分子数据)
- **LapPE**: Laplacian位置编码
- **TypeDictNode**: 类型字典编码
- **LinearNode**: 线性变换编码

#### 边编码器选项  
- **Bond**: 化学键编码 (分子数据)
- **LinearEdge**: 线性边编码
- **DummyEdge**: 虚拟边编码
- **TypeDictEdge**: 类型字典边编码

## 性能优化建议

### 1. 数据预处理优化
- **缓存机制**: 利用PyG的processed文件缓存
- **批量处理**: 使用DataLoader的num_workers并行加载
- **内存管理**: 对大数据集使用InMemoryDataset vs Dataset选择

### 2. 位置编码优化
- **选择性启用**: 根据任务需求选择合适的PE方法
- **维度调优**: 平衡表达能力和计算开销
- **预计算**: 利用位置编码的预计算缓存

### 3. 配置优化
- **批量大小**: 根据GPU内存和数据集大小调整
- **学习率**: 针对不同数据集规模调整学习策略
- **正则化**: 根据数据集复杂度调整dropout和权重衰减

## 常见问题与解决方案

### 1. 数据格式问题
**问题**: 节点/边特征维度不匹配
**解决**: 检查编码器配置，确保dim_inner匹配

### 2. 内存问题
**问题**: 大数据集导致内存不足
**解决**: 
- 减小batch_size
- 使用梯度累积 (batch_accumulation)
- 启用分桶处理 (Bucket变体)

### 3. 收敛问题
**问题**: 模型在新数据集上收敛困难
**解决**:
- 调整学习率和调度策略
- 检查数据预处理的正确性
- 调整模型架构参数 (layers, dim_hidden)

### 4. 性能问题
**问题**: 训练速度过慢
**解决**:
- 使用较小的max_freqs (位置编码)
- 减少GPS层数
- 启用混合精度训练

## 集成难度评估总结

| 集成方式 | 难度等级 | 开发时间 | 主要工作 | 风险等级 |
|---------|---------|----------|----------|----------|
| 现有PyG数据集扩展 | ⭐ | 1-2小时 | 修改预处理函数 | 低 |
| 相似格式数据集 | ⭐⭐ | 半天 | 适配预处理逻辑 | 低 |
| 自定义数据集类 | ⭐⭐⭐ | 1-2天 | 完整实现数据加载 | 中 |
| 跨框架数据集 | ⭐⭐⭐⭐ | 3-5天 | 格式转换+兼容性 | 高 |
| 复杂多模态数据 | ⭐⭐⭐⭐⭐ | 1-2周 | 架构调整+新功能 | 很高 |

## 最佳实践建议

### 1. 开发流程
1. **先测试最小集成**: 用现有配置测试基本功能
2. **逐步优化**: 先确保能运行，再优化性能
3. **充分验证**: 在小数据集上验证正确性
4. **文档记录**: 记录配置和特殊处理逻辑

### 2. 代码质量
1. **错误处理**: 添加完整的异常处理和错误信息
2. **类型检查**: 使用类型注解和验证
3. **测试覆盖**: 编写单元测试确保功能正确
4. **向后兼容**: 确保不影响现有数据集

### 3. 性能监控
1. **基准测试**: 与类似数据集对比性能
2. **内存监控**: 跟踪内存使用情况
3. **速度分析**: 识别性能瓶颈
4. **可视化**: 使用wandb监控训练过程

---

**文档版本**: 1.0  
**最后更新**: 2024年最新版本  
**适用范围**: Graph-Mamba项目新数据集集成指导