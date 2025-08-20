import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_edge_encoder


@register_edge_encoder('LinearEdge')
class LinearEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        if cfg.dataset.name in ['MNIST', 'CIFAR10']:
            self.in_dim = 1
        elif cfg.dataset.name == "ogbn-proteins":
            self.in_dim = 8
        elif cfg.dataset.name in [
            # 原始支持的数据集
            'DD', 'PROTEINS', 'colors3', 'mutagenicity', 'coildel', 'synthetic', 'dblp', 'twitter',
            # 导出数据集 - 分子数据集
            'qm9', 'subset', 'aqsol', 'AQSOL', 'QM9',  # subset是ZINC的名称，支持大小写
            # QM9数据集的各种属性变体
            'QM9-homo', 'QM9-gap', 'QM9-lumo', 'QM9-all', 'QM9-mu', 'QM9-alpha',
            'QM9-r2', 'QM9-zpve', 'QM9-u0', 'QM9-u298', 'QM9-h298', 'QM9-g298', 'QM9-cv',
            'QM9-u0_atom', 'QM9-u298_atom', 'QM9-h298_atom', 'QM9-g298_atom',
            # 导出数据集 - OGB数据集  
            'ogbg-molhiv', 'peptides-functional', 'peptides-structural',
            # 导出数据集 - TU数据集的不同命名方式
            'dd', 'proteins', 'COLORS3', 'MUTAGENICITY', 'COILDEL', 'SYNTHETIC', 'DBLP', 'TWITTER',
            'COLORS-3', 'COIL-DEL', 'TWITTER-Real-Graph-Partial', 'Mutagenicity',  # 支持更多命名变体
            # 其他可能的命名方式
            'colors-3', 'coil-del'
        ]:  # 支持所有导出数据集的1维边特征
            self.in_dim = 1
        else:
            raise ValueError("Input edge feature dim is required to be hardset "
                             "or refactored to use a cfg option.")
        self.encoder = torch.nn.Linear(self.in_dim, emb_dim)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr.view(-1, self.in_dim))
        return batch
