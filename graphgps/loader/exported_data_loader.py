"""
导出数据专用加载器
基于TokenizerGraph项目导出的标准化数据格式
确保与原项目100%数据一致性
"""

import os
import pickle
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.graphgym.config import cfg
from typing import Dict, List, Any, Optional


class ExportedDataset(InMemoryDataset):
    """
    加载TokenizerGraph导出的标准化数据格式
    
    数据格式：
    {
        'graphs': List[Dict],      # 图数据列表
        'labels': List[Any],       # 标签列表  
        'splits': Dict[str, np.ndarray],  # 数据划分
        'dataset_info': Dict       # 数据集元信息
    }
    """
    
    def __init__(self, 
                 exported_file_path: str,
                 dataset_name: str,
                 transform=None, 
                 pre_transform=None,
                 qm9_target: str = None):
        """
        Args:
            exported_file_path: 导出的pkl文件路径
            dataset_name: 数据集名称（用于日志）
        """
        self.exported_file_path = exported_file_path
        self.dataset_name = dataset_name
        self.qm9_target = qm9_target
        
        if not os.path.exists(exported_file_path):
            raise FileNotFoundError(f"Exported data file not found: {exported_file_path}")
        
        # 临时root目录，InMemoryDataset需要
        temp_root = os.path.dirname(exported_file_path)
        super().__init__(root=temp_root, transform=transform, pre_transform=pre_transform)
        
        # 加载导出数据
        self._load_exported_data()
        
    def _load_exported_data(self):
        """加载导出的标准化数据"""
        print(f"🔄 Loading exported data: {self.dataset_name}")
        print(f"   File: {self.exported_file_path}")
        
        # 加载pkl文件
        with open(self.exported_file_path, 'rb') as f:
            exported_data = pickle.load(f)
        
        # 验证数据格式
        required_keys = ['graphs', 'labels', 'splits']
        for key in required_keys:
            if key not in exported_data:
                raise ValueError(f"Missing key '{key}' in exported data")
        
        graphs_data = exported_data['graphs']
        labels_data = exported_data['labels']
        splits_data = exported_data['splits']
        dataset_info = exported_data.get('dataset_info', {
            'task_type': 'classification',
            'dataset_name': self.dataset_name
        })
        
        print(f"   📊 Loaded {len(graphs_data)} graphs")
        print(f"   📋 Task type: {dataset_info.get('task_type', 'unknown')}")
        
        # 转换为PyG格式
        pyg_data_list = self._convert_to_pyg_format(graphs_data, labels_data)
        
        # 使用PyG的collate方法处理数据
        self.data, self.slices = self.collate(pyg_data_list)
        
        # 修复多维标签被展平的问题
        first_label = pyg_data_list[0].y
        if first_label.dim() > 0 and len(first_label) > 1:
            # 多维标签：重新reshape并更新slices
            num_samples = len(pyg_data_list)
            label_dim = len(first_label)
            self.data.y = self.data.y.view(num_samples, label_dim)
            # 更新slices信息：每个样本占一行
            self.slices['y'] = torch.arange(0, num_samples + 1)
            
            # QM9多属性回归任务的z-score标准化
            if self.dataset_name == 'qm9' and self.qm9_target == 'all':
                print(f"   📊 QM9多属性回归：对{label_dim}个属性进行z-score标准化")
                # 计算每个属性的均值和标准差
                self.data.y = (self.data.y - self.data.y.mean(dim=0, keepdim=True)) / (self.data.y.std(dim=0, keepdim=True) + 1e-8)
        
        # 设置数据分割
        self._setup_data_splits(splits_data)
        
        # 保存数据集信息
        self.dataset_info = dataset_info
        
        print(f"   ✅ Successfully loaded {self.dataset_name}")
    
    def _convert_to_pyg_format(self, graphs_data: List[Dict], labels_data: List[Any]) -> List[Data]:
        """
        将导出的图数据转换为PyG的Data格式
        
        导出格式：
        {
            'src': np.ndarray,         # 源节点ID
            'dst': np.ndarray,         # 目标节点ID  
            'num_nodes': int,          # 节点总数
            'node_feat': np.ndarray,   # 节点特征 (N, D_node)
            'edge_feat': np.ndarray,   # 边特征 (E, D_edge)
        }
        """
        pyg_data_list = []
        
        for i, (graph_dict, label) in enumerate(zip(graphs_data, labels_data)):
            # 验证图数据格式
            required_graph_keys = ['src', 'dst', 'num_nodes', 'node_feat', 'edge_feat']
            for key in required_graph_keys:
                if key not in graph_dict:
                    raise ValueError(f"Missing key '{key}' in graph {i}")
            
            # 提取数据
            src = graph_dict['src']
            dst = graph_dict['dst'] 
            num_nodes = graph_dict['num_nodes']
            node_feat = graph_dict['node_feat']
            edge_feat = graph_dict['edge_feat']
            
            # 转换为torch tensor
            # 边索引：将src, dst组合为edge_index
            edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long()
            
            # 节点特征：确保2D并转换为float32
            if node_feat.ndim == 1:
                node_feat = node_feat.reshape(-1, 1)
            x = torch.from_numpy(node_feat).float()  # 转换为float32以兼容Linear层
            
            # 边特征：确保2D并转换为float32
            if edge_feat.ndim == 1:
                edge_feat = edge_feat.reshape(-1, 1)
            edge_attr = torch.from_numpy(edge_feat).float()  # 转换为float32以兼容Linear层
            
            # 标签处理 - 支持多种数据类型
            if isinstance(label, np.ndarray):
                y = torch.from_numpy(label)
            elif isinstance(label, dict):
                # QM9数据集的特殊处理
                if self.dataset_name == 'qm9' and self.qm9_target:
                    if self.qm9_target == 'all':
                        # 使用所有属性进行多回归任务
                        qm9_properties = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 
                                        'u298', 'h298', 'g298', 'cv', 'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom']
                        y = torch.tensor([label[k] for k in qm9_properties])
                    else:
                        # 使用指定属性进行单回归任务
                        if self.qm9_target not in label:
                            raise ValueError(f"QM9 target '{self.qm9_target}' not found in label keys: {list(label.keys())}")
                        y = torch.tensor([label[self.qm9_target]])
                else:
                    # 其他dict格式标签（保持原有逻辑）
                    y = torch.tensor([label[k] for k in sorted(label.keys())])
            elif isinstance(label, (int, float, np.integer, np.floating)):
                # 支持基本数据类型：int, float等
                y = torch.tensor([label])
            elif isinstance(label, (list, tuple)):
                # 支持列表和元组
                y = torch.tensor(label)
            else:
                # 尝试直接转换为tensor
                try:
                    y = torch.tensor([label])
                except (ValueError, TypeError):
                    raise ValueError(f"Unsupported label type: {type(label)}. Expected np.ndarray, dict, int, float, list, or tuple.")
            
            # 确保标签是正确的数据类型
            if y.dtype == torch.int64:
                pass  # 分类任务保持int64
            else:
                y = y.float()  # 回归任务转为float
            
            # 创建PyG Data对象
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                num_nodes=num_nodes
            )
            
            pyg_data_list.append(data)
        
        return pyg_data_list
    
    def _setup_data_splits(self, splits_data: Dict[str, np.ndarray]):
        """设置数据分割"""
        train_idx = splits_data['train'].tolist()
        val_idx = splits_data['val'].tolist() 
        test_idx = splits_data['test'].tolist()
        
        self.split_idxs = [train_idx, val_idx, test_idx]
        
        # 添加GraphGym期望的索引字段
        import torch
        self.data.train_graph_index = torch.tensor(train_idx, dtype=torch.long)
        self.data.val_graph_index = torch.tensor(val_idx, dtype=torch.long)
        self.data.test_graph_index = torch.tensor(test_idx, dtype=torch.long)
        
        print(f"   📋 Data splits:")
        print(f"      Train: {len(train_idx)} samples")
        print(f"      Val:   {len(val_idx)} samples")
        print(f"      Test:  {len(test_idx)} samples")
    
    @property
    def processed_file_names(self):
        return []  # 数据已经导出处理好了
    
    def process(self):
        pass  # 无需额外处理


# 数据集名称到文件名的映射
DATASET_FILE_MAPPING = {
    # 分子数据集（回归）
    'qm9': 'qm9_export.pkl',
    'zinc': 'zinc_export.pkl', 
    'aqsol': 'aqsol_export.pkl',
    'molhiv': 'molhiv_export.pkl',  # 二分类，但放在这里方便管理
    
    # TU数据集（分类）
    'colors3': 'colors3_export.pkl',
    'proteins': 'proteins_export.pkl', 
    'dd': 'dd_export.pkl',
    'mutagenicity': 'mutagenicity_export.pkl',
    'coildel': 'coildel_export.pkl',
    'dblp': 'dblp_export.pkl',
    'twitter': 'twitter_export.pkl',
    'synthetic': 'synthetic_export.pkl',
    
    # LRGB数据集
    'peptides_func': 'peptides_func_export.pkl',
    'peptides_struct': 'peptides_struct_export.pkl',
}


def load_exported_dataset(dataset_name: str, exported_data_dir: str, qm9_target: str = 'homo') -> ExportedDataset:
    """
    加载指定的导出数据集
    
    Args:
        dataset_name: 数据集名称
        exported_data_dir: 导出数据目录路径
        
    Returns:
        ExportedDataset对象
    """
    if dataset_name not in DATASET_FILE_MAPPING:
        available_datasets = list(DATASET_FILE_MAPPING.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available_datasets}")
    
    file_name = DATASET_FILE_MAPPING[dataset_name]
    file_path = os.path.join(exported_data_dir, file_name)
    
    return ExportedDataset(
        exported_file_path=file_path,
        dataset_name=dataset_name,
        qm9_target=qm9_target if dataset_name == 'qm9' else None
    )


# 主加载器要调用的预处理函数
def preformat_exported_qm9(exported_data_dir):
    """加载导出的QM9数据"""
    from torch_geometric.graphgym.config import cfg
    
    # 根据数据集名称解析QM9属性配置
    dataset_name = cfg.dataset.name
    if dataset_name == 'QM9-homo':
        qm9_target = 'homo'
    elif dataset_name == 'QM9-gap':
        qm9_target = 'gap'
    elif dataset_name == 'QM9-all':
        qm9_target = 'all'
    elif dataset_name.startswith('QM9-'):
        # 提取属性名，如 QM9-lumo -> lumo
        qm9_target = dataset_name.split('-', 1)[1]
    else:
        # 默认使用homo属性
        qm9_target = 'homo'
    
    return load_exported_dataset('qm9', exported_data_dir, qm9_target=qm9_target)

def preformat_exported_zinc(exported_data_dir):
    """加载导出的ZINC数据"""
    return load_exported_dataset('zinc', exported_data_dir)

def preformat_exported_aqsol(exported_data_dir):
    """加载导出的AQSOL数据"""
    return load_exported_dataset('aqsol', exported_data_dir)

def preformat_exported_molhiv(exported_data_dir):
    """加载导出的MOLHIV数据"""
    return load_exported_dataset('molhiv', exported_data_dir)

def preformat_exported_colors3(exported_data_dir):
    """加载导出的COLORS-3数据"""
    return load_exported_dataset('colors3', exported_data_dir)

def preformat_exported_proteins(exported_data_dir):
    """加载导出的PROTEINS数据"""
    return load_exported_dataset('proteins', exported_data_dir)

def preformat_exported_dd(exported_data_dir):
    """加载导出的DD数据"""
    return load_exported_dataset('dd', exported_data_dir)

def preformat_exported_mutagenicity(exported_data_dir):
    """加载导出的Mutagenicity数据"""
    return load_exported_dataset('mutagenicity', exported_data_dir)

def preformat_exported_coildel(exported_data_dir):
    """加载导出的COIL-DEL数据"""
    return load_exported_dataset('coildel', exported_data_dir)

def preformat_exported_dblp(exported_data_dir):
    """加载导出的DBLP数据"""
    return load_exported_dataset('dblp', exported_data_dir)

def preformat_exported_twitter(exported_data_dir):
    """加载导出的TWITTER数据"""
    return load_exported_dataset('twitter', exported_data_dir)

def preformat_exported_synthetic(exported_data_dir):
    """加载导出的SYNTHETIC数据"""
    return load_exported_dataset('synthetic', exported_data_dir)

def preformat_exported_peptides_func(exported_data_dir):
    """加载导出的Peptides-func数据"""
    return load_exported_dataset('peptides_func', exported_data_dir)

def preformat_exported_peptides_struct(exported_data_dir):
    """加载导出的Peptides-struct数据"""
    return load_exported_dataset('peptides_struct', exported_data_dir)
