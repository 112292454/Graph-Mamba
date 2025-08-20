#!/usr/bin/env python3
"""
修复peptides数据集标签格式的脚本
将 list of lists 转换为 numpy arrays
"""

import pickle
import numpy as np

def fix_peptides_labels():
    # 修复peptides-func
    print("修复 peptides-func 标签格式...")
    with open('export_system/exported/peptides_func_export.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print(f"原始标签类型: {type(data['labels'])}")
    print(f"原始标签长度: {len(data['labels'])}")
    print(f"第一个标签: {data['labels'][0]}")
    
    # 转换为numpy数组
    labels_array = np.array(data['labels'], dtype=np.float32)
    print(f"转换后标签形状: {labels_array.shape}")
    print(f"转换后标签类型: {labels_array.dtype}")
    
    data['labels'] = labels_array
    
    # 保存修复后的数据
    with open('export_system/exported/peptides_func_export.pkl', 'wb') as f:
        pickle.dump(data, f)
    print("✅ peptides-func 标签格式修复完成")
    
    # 修复peptides-struct
    print("\n修复 peptides-struct 标签格式...")
    with open('export_system/exported/peptides_struct_export.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print(f"原始标签类型: {type(data['labels'])}")
    print(f"原始标签长度: {len(data['labels'])}")
    print(f"第一个标签: {data['labels'][0]}")
    
    # 转换为numpy数组
    labels_array = np.array(data['labels'], dtype=np.float32)
    print(f"转换后标签形状: {labels_array.shape}")
    print(f"转换后标签类型: {labels_array.dtype}")
    
    data['labels'] = labels_array
    
    # 保存修复后的数据
    with open('export_system/exported/peptides_struct_export.pkl', 'wb') as f:
        pickle.dump(data, f)
    print("✅ peptides-struct 标签格式修复完成")

if __name__ == "__main__":
    fix_peptides_labels()
