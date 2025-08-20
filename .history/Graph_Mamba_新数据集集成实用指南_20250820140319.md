# Graph-Mamba 新数据集集成实用指南

## 快速开始指南

### 集成DGL TUDataset到Graph-Mamba

假设你想集成DGL中的一个TUDataset数据集到Graph-Mamba中，以下是完整的操作步骤：

#### 方案一：直接使用PyG的TUDataset (推荐⭐)

**步骤1：确认数据集支持**
```python
# 检查数据集是否在PyG TUDataset中可用
from torch_geometric.datasets import TUDataset
dataset = TUDataset(root='test', name='MUTAG')  # 测试是否可以加载
```

**步骤2：修改预处理函数**
```python
# 在 graphgps/loader/master_loader.py 的 preformat_TUDataset 函数中添加
def preformat_TUDataset(dataset_dir, name):
    if name in ['DD', 'NCI1', 'ENZYMES', 'PROTEINS', 'MUTAG']:  # 添加你的数据集
        func = None
    elif name.startswith('IMDB-') or name == "COLLAB":
        func = T.Constant()
    elif name in ['REDDIT-BINARY', 'REDDIT-MULTI-5K']:  # 新增社交网络数据集
        func = T.OneHotDegree(max_degree=100)  # 根据数据集特点选择合适的预处理
    else:
        raise ValueError(f"Loading dataset '{name}' from TUDataset is not supported.")
    dataset = TUDataset(dataset_dir, name, pre_transform=func)
    return dataset
```

**步骤3：创建配置文件**
```yaml
# configs/Mamba/mutag-EX.yaml
out_dir: results
metric_best: accuracy
wandb:
  use: True
  project: Graph-Mamba-NewDatasets
dataset:
  format: PyG-TUDataset
  name: MUTAG
  task: graph
  task_type: classification
  transductive: False
  node_encoder: True
  node_encoder_name: Atom  # 分子数据集使用Atom编码
  edge_encoder: True
  edge_encoder_name: Bond  # 分子数据集使用Bond编码
posenc_LapPE:
  enable: True
  eigen:
    max_freqs: 8
  dim_pe: 16
train:
  mode: custom
  batch_size: 32
  eval_period: 1
model:
  type: GPSModel
  loss_fun: cross_entropy
gt:
  layer_type: CustomGatedGCN+Mamba_Hybrid_Degree_Noise
  layers: 4
  n_heads: 4
  dim_hidden: 64
gnn:
  dim_inner: 64
optim:
  optimizer: adamW
  base_lr: 0.001
  max_epoch: 100
seed: 42
```

**步骤4：运行测试**
```bash
cd /home/gzy/py/Graph-Mamba
python main.py --cfg configs/Mamba/mutag-EX.yaml wandb.use False
```

#### 方案二：集成自定义ZINC变体数据集

**步骤1：创建自定义预处理函数**
```python
# 在 graphgps/loader/master_loader.py 中添加
def preformat_ZINC_Custom(dataset_dir, name):
    """自定义ZINC数据集变体"""
    if name == 'zinc_filtered':
        # 加载完整ZINC数据集
        train_dataset = ZINC(root=dataset_dir, subset=False, split='train')
        val_dataset = ZINC(root=dataset_dir, subset=False, split='val')  
        test_dataset = ZINC(root=dataset_dir, subset=False, split='test')
        
        # 自定义过滤逻辑 (例如：只保留节点数在10-50之间的分子)
        def filter_by_size(dataset, min_nodes=10, max_nodes=50):
            filtered_data = []
            for data in dataset:
                if min_nodes <= data.num_nodes <= max_nodes:
                    filtered_data.append(data)
            return filtered_data
        
        train_filtered = filter_by_size(train_dataset)
        val_filtered = filter_by_size(val_dataset)
        test_filtered = filter_by_size(test_dataset)
        
        # 重新构建数据集
        from torch_geometric.data import InMemoryDataset
        class FilteredZINC(InMemoryDataset):
            def __init__(self, data_list):
                super().__init__()
                self.data, self.slices = self.collate(data_list)
        
        # 合并所有数据
        all_data = train_filtered + val_filtered + test_filtered
        dataset = FilteredZINC(all_data)
        
        # 设置分割索引
        n1, n2, n3 = len(train_filtered), len(val_filtered), len(test_filtered)
        dataset.split_idxs = [
            list(range(n1)),
            list(range(n1, n1 + n2)),
            list(range(n1 + n2, n1 + n2 + n3))
        ]
        
    else:
        raise ValueError(f"Unsupported ZINC variant: {name}")
    return dataset

# 在load_dataset_master函数中添加路由
elif pyg_dataset_id == 'ZINC_Custom':
    dataset = preformat_ZINC_Custom(dataset_dir, name)
```

**步骤2：创建配置文件**
```yaml
# configs/Mamba/zinc-custom-EX.yaml
dataset:
  format: PyG-ZINC_Custom
  name: zinc_filtered
  task: graph
  task_type: regression
# ... 其他配置类似标准ZINC
```

### 集成外部框架数据集 (DGL示例)

#### 完整DGL到PyG转换示例

**步骤1：创建转换工具**
```python
# graphgps/loader/dgl_converter.py
import torch
import dgl
from torch_geometric.data import Data, InMemoryDataset

def dgl_graph_to_pyg(dgl_graph):
    """将DGL图转换为PyG格式"""
    # 获取边
    src, dst = dgl_graph.edges()
    edge_index = torch.stack([src, dst], dim=0)
    
    # 节点特征
    x = dgl_graph.ndata.get('feat', None)
    if x is None:
        x = torch.ones(dgl_graph.num_nodes(), 1)  # 默认特征
    
    # 边特征  
    edge_attr = dgl_graph.edata.get('feat', None)
    
    # 图标签
    y = dgl_graph.ndata.get('label', None)
    if y is None:
        y = dgl_graph.graph_attrs.get('label', torch.tensor(0))
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

class DGLToPyGDataset(InMemoryDataset):
    """DGL数据集转PyG数据集的通用类"""
    def __init__(self, dgl_dataset, root, transform=None, pre_transform=None):
        self.dgl_dataset = dgl_dataset
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        data_list = []
        for i in range(len(self.dgl_dataset)):
            dgl_graph, label = self.dgl_dataset[i]
            pyg_data = dgl_graph_to_pyg(dgl_graph)
            if isinstance(label, torch.Tensor):
                pyg_data.y = label
            data_list.append(pyg_data)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
```

**步骤2：集成到主加载器**
```python
# 在 graphgps/loader/master_loader.py 中添加
def preformat_DGL_Dataset(dataset_dir, name):
    """DGL数据集预处理"""
    try:
        import dgl
        from dgl.data import TUDataset as DGL_TUDataset
        from graphgps.loader.dgl_converter import DGLToPyGDataset
    except ImportError:
        raise ImportError("DGL not installed. Install with: pip install dgl")
    
    if name.startswith('DGL_'):
        dgl_name = name[4:]  # 去掉'DGL_'前缀
        dgl_dataset = DGL_TUDataset(dgl_name)
        dataset = DGLToPyGDataset(dgl_dataset, dataset_dir)
        
        # 手动设置分割 (DGL数据集通常没有预定义分割)
        n = len(dataset)
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)
        
        import random
        random.seed(42)
        indices = list(range(n))
        random.shuffle(indices)
        
        dataset.split_idxs = [
            indices[:train_size],
            indices[train_size:train_size+val_size],
            indices[train_size+val_size:]
        ]
    else:
        raise ValueError(f"Unsupported DGL dataset: {name}")
    
    return dataset

# 在load_dataset_master中添加
elif pyg_dataset_id == 'DGL_Dataset':
    dataset = preformat_DGL_Dataset(dataset_dir, name)
```

## 性能调优指南

### 1. 内存优化

**问题**: 数据集太大导致内存不足
```yaml
# 解决方案配置
train:
  batch_size: 16  # 减小批量大小
optim:
  batch_accumulation: 4  # 梯度累积模拟大批量
```

**使用分桶处理大图**:
```yaml
gt:
  layer_type: CustomGatedGCN+Mamba_Hybrid_Degree_Noise_Bucket  # 使用分桶变体
```

### 2. 速度优化

**减少位置编码计算开销**:
```yaml
posenc_LapPE:
  enable: True
  eigen:
    max_freqs: 8  # 减少特征值数量 (默认16)
  dim_pe: 8     # 减少编码维度 (默认16)
```

**调整模型复杂度**:
```yaml
gt:
  layers: 3        # 减少层数 (默认4-6)
  dim_hidden: 48   # 减少隐藏维度 (默认64-96)
```

### 3. 收敛性优化

**学习率调度**:
```yaml
optim:
  optimizer: adamW
  base_lr: 0.001
  scheduler: cosine_with_warmup
  num_warmup_epochs: 10  # 增加预热期
  weight_decay: 0.01     # 适当的权重衰减
```

**数据增强策略**:
```yaml
# 对于小数据集，可以启用数据增强
prep:
  add_reverse_edges: True  # 添加反向边
  add_self_loops: True     # 添加自环
```

## 常见错误与解决方案

### 1. 维度不匹配错误
```python
# 错误信息: RuntimeError: mat1 and mat2 shapes cannot be multiplied
# 原因: gt.dim_hidden != gnn.dim_inner
# 解决:
gt:
  dim_hidden: 64
gnn:
  dim_inner: 64  # 必须匹配
```

### 2. 数据加载错误
```python
# 错误信息: AttributeError: 'Data' object has no attribute 'x'
# 原因: 节点没有特征
# 解决: 在预处理中添加默认特征
def preformat_your_dataset(dataset_dir, name):
    dataset = YourDataset(dataset_dir, name)
    
    # 为没有特征的节点添加度特征
    def add_node_features(data):
        if data.x is None:
            from torch_geometric.utils import degree
            deg = degree(data.edge_index[0], data.num_nodes)
            data.x = deg.unsqueeze(-1).float()
        return data
    
    pre_transform_in_memory(dataset, add_node_features)
    return dataset
```

### 3. 标签格式错误
```python
# 错误信息: Expected target size (batch_size,), got torch.Size([batch_size, 1])
# 原因: 标签维度不正确
# 解决: 调整标签格式
def fix_labels(data):
    if data.y.dim() > 1 and data.y.size(1) == 1:
        data.y = data.y.squeeze(-1)
    return data

pre_transform_in_memory(dataset, fix_labels)
```

### 4. 位置编码错误
```python
# 错误信息: RuntimeError: Laplacian eigendecomposition failed
# 原因: 图不连通或太大
# 解决: 禁用位置编码或使用更鲁棒的PE
posenc_LapPE:
  enable: False  # 临时禁用
# 或者使用
posenc_RWSE:     # 随机游走SE更鲁棒
  enable: True
```

## 完整集成清单

### 开发前检查
- [ ] 确认数据集格式和特征类型
- [ ] 检查是否已有类似数据集的实现
- [ ] 评估计算资源需求
- [ ] 准备测试用小规模数据

### 实现阶段
- [ ] 实现数据加载和预处理
- [ ] 添加到主加载器路由
- [ ] 创建配置文件
- [ ] 编写单元测试

### 验证阶段
- [ ] 小规模数据测试
- [ ] 检查数据格式正确性
- [ ] 验证训练收敛性
- [ ] 性能基准测试

### 优化阶段
- [ ] 超参数调优
- [ ] 性能profiling
- [ ] 内存使用优化
- [ ] 文档编写

## 实际操作示例

### 场景：集成你的自定义分子数据集

假设你有一个CSV格式的分子数据集，包含SMILES字符串和标签：

```python
# 步骤1：创建数据集类
# graphgps/loader/dataset/custom_molecules.py
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import InMemoryDataset, Data

class CustomMoleculeDataset(InMemoryDataset):
    def __init__(self, root, csv_file, smiles_col='smiles', label_col='label', 
                 transform=None, pre_transform=None):
        self.csv_file = csv_file
        self.smiles_col = smiles_col
        self.label_col = label_col
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def smiles_to_graph(self, smiles):
        """将SMILES转换为PyG图"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 原子特征 (原子序数)
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append([atom.GetAtomicNum()])
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # 边索引和边特征
        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])  # 无向图
            bond_type = bond.GetBondTypeAsDouble()
            edge_attrs.extend([[bond_type], [bond_type]])
        
        if len(edge_indices) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def process(self):
        df = pd.read_csv(self.csv_file)
        data_list = []
        
        for idx, row in df.iterrows():
            graph = self.smiles_to_graph(row[self.smiles_col])
            if graph is not None:
                graph.y = torch.tensor([row[self.label_col]], dtype=torch.float)
                data_list.append(graph)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# 步骤2：添加预处理函数
def preformat_CustomMolecules(dataset_dir, csv_path):
    dataset = CustomMoleculeDataset(root=dataset_dir, csv_file=csv_path)
    
    # 随机分割数据
    n = len(dataset)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    
    indices = torch.randperm(n)
    dataset.split_idxs = [
        indices[:train_size].tolist(),
        indices[train_size:train_size+val_size].tolist(),
        indices[train_size+val_size:].tolist()
    ]
    
    return dataset

# 步骤3：集成到主加载器
elif pyg_dataset_id == 'CustomMolecules':
    dataset = preformat_CustomMolecules(dataset_dir, name)  # name是CSV文件路径
```

**配置文件**:
```yaml
# configs/Mamba/custom-molecules-EX.yaml
dataset:
  format: PyG-CustomMolecules
  name: /path/to/your/molecules.csv  # CSV文件路径
  task: graph
  task_type: regression  # 或classification
  node_encoder: True
  node_encoder_name: Atom
  edge_encoder: True
  edge_encoder_name: Bond
# ... 其他标准配置
```

**运行**:
```bash
python main.py --cfg configs/Mamba/custom-molecules-EX.yaml
```

## 总结建议

1. **从简单开始**: 先尝试使用现有的PyG数据集类
2. **逐步扩展**: 基于成功的简单案例逐步增加复杂性
3. **充分测试**: 在小数据集上验证正确性后再处理大数据集
4. **文档记录**: 详细记录每个集成的特殊处理和配置
5. **性能监控**: 持续监控内存使用和训练速度
6. **社区支持**: 参考GraphGPS和Graph-Mamba的GitHub issues获取帮助

通过以上指南，你应该能够成功集成各种新数据集到Graph-Mamba框架中。记住，数据集集成的关键是理解数据格式、正确的预处理和合适的配置调优。

---

**文档版本**: 1.0  
**最后更新**: 2024年最新版本  
**维护者**: Graph-Mamba项目组