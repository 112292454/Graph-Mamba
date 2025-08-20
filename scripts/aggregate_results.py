#!/usr/bin/env python3
"""
聚合实验结果脚本

此脚本从 results/ 目录聚合不同模型、不同数据集的实验结果，
生成便于分析的CSV文件和透视表。

使用方法:
    python scripts/aggregate_results.py [--output-dir outputs]
    
输出:
    - benchmark_summary.csv: 详细结果记录
    - benchmark_pivot.csv: 透视表（横轴=模型，纵轴=数据集×指标）
"""

import os
import json
import sys
import argparse
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
import re

def parse_result_dir_name(result_dir_name: str) -> Tuple[str, str]:
    """
    解析结果目录名，提取数据集名和模型名
    
    Args:
        result_dir_name: 如 'zinc-exported-GatedGCN'
        
    Returns:
        (dataset, model): 如 ('zinc', 'GatedGCN')
    """
    # 处理特殊情况
    if 'exported' in result_dir_name:
        parts = result_dir_name.split('-exported-')
        if len(parts) == 2:
            dataset, model = parts
            return dataset, model
    
    # 处理其他可能的模式
    # 如果没有-exported-模式，尝试其他分割方式
    parts = result_dir_name.split('-')
    if len(parts) >= 2:
        # 假设最后一部分是模型名
        model = parts[-1]
        dataset = '-'.join(parts[:-1])
        return dataset, model
    
    return result_dir_name, "Unknown"

def get_dataset_metric_mapping() -> Dict[str, str]:
    """
    根据数据集类型返回主要评估指标的映射
    """
    return {
        # 回归任务 - 使用MAE
        'zinc': 'mae',
        'aqsol': 'mae', 
        'qm9': 'mae',
        'peptides-struct': 'mae',
        
        # 分类任务 - 使用accuracy
        'dd': 'accuracy',
        'proteins': 'accuracy',
        'colors3': 'accuracy',
        'mutagenicity': 'accuracy',
        'coildel': 'accuracy',
        'dblp': 'accuracy',
        'twitter': 'accuracy',
        'synthetic': 'accuracy',
        
        # OGB任务 - 使用AUC
        'molhiv': 'auc',
        
        # 多标签任务 - 使用AP
        'peptides-func': 'ap'
    }

def load_json_safely(file_path: Path) -> Optional[Dict]:
    """安全地加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  警告: 无法读取 {file_path}: {e}")
        return None

def extract_metrics_from_result(result_dir: Path) -> Dict:
    """
    从结果目录提取关键指标
    
    优先级:
    1. agg/val/best.json (验证集最佳结果)
    2. agg/test/best.json (测试集最佳结果，如果存在)
    3. {seed}/test/stats.json的最后一行 (测试集最终结果)
    
    Args:
        result_dir: 结果目录路径
        
    Returns:
        包含指标的字典
    """
    metrics = {}
    
    # 优先尝试读取agg/val/best.json
    val_best_file = result_dir / "agg" / "val" / "best.json"
    if val_best_file.exists():
        data = load_json_safely(val_best_file)
        if data:
            metrics['val'] = data
    
    # 尝试读取agg/test/best.json  
    test_best_file = result_dir / "agg" / "test" / "best.json"
    if test_best_file.exists():
        data = load_json_safely(test_best_file)
        if data:
            metrics['test'] = data
    
    # 如果没有agg结果，尝试从seed目录读取
    if not metrics:
        # 查找seed目录（通常是数字命名）
        seed_dirs = [d for d in result_dir.iterdir() 
                    if d.is_dir() and d.name.isdigit()]
        
        if seed_dirs:
            # 选择第一个seed目录
            seed_dir = seed_dirs[0]
            
            # 读取test/stats.json
            test_stats_file = seed_dir / "test" / "stats.json"
            if test_stats_file.exists():
                try:
                    # 读取最后一行作为最终结果
                    with open(test_stats_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if last_line:
                                data = json.loads(last_line)
                                metrics['test'] = data
                except Exception as e:
                    print(f"⚠️  警告: 读取 {test_stats_file} 出错: {e}")
    
    return metrics

def aggregate_all_results(results_dir: Path) -> List[Dict]:
    """
    聚合results目录下所有实验结果
    """
    all_results = []
    dataset_metrics = get_dataset_metric_mapping()
    
    print("🔍 扫描结果目录...")
    
    # 遍历results目录下的所有子目录
    for result_dir in results_dir.iterdir():
        if not result_dir.is_dir():
            continue
            
        # 跳过特殊目录
        if result_dir.name.startswith('.') or result_dir.name == 'parallel_test_logs':
            continue
        
        print(f"📊 处理: {result_dir.name}")
        
        # 解析目录名获取数据集和模型
        dataset, model = parse_result_dir_name(result_dir.name)
        
        # 提取实验指标
        metrics = extract_metrics_from_result(result_dir)
        
        if not metrics:
            print(f"   ⚠️  未找到有效结果文件")
            continue
        
        # 确定主要评估指标
        base_dataset = dataset.split('-')[0] if '-' in dataset else dataset
        primary_metric = dataset_metrics.get(base_dataset, 'accuracy')
        
        # 处理每个split的结果
        for split, data in metrics.items():
            if not data:
                continue
                
            # 提取关键信息
            result_record = {
                'model': model,
                'dataset': dataset,
                'split': split,
                'epoch': data.get('epoch', 0),
                'primary_metric': primary_metric,
                'primary_value': data.get(primary_metric, None)
            }
            
            # 添加所有可用的指标
            for metric_name in ['mae', 'accuracy', 'auc', 'ap', 'r2', 'loss']:
                if metric_name in data:
                    result_record[metric_name] = data[metric_name]
            
            # 添加其他有用信息
            result_record['runtime'] = data.get('time_epoch', None)
            result_record['params'] = data.get('params', None)
            
            all_results.append(result_record)
            
            print(f"   ✅ {split}: {primary_metric}={data.get(primary_metric, 'N/A'):.4f}" 
                  if isinstance(data.get(primary_metric), (int, float)) 
                  else f"   ✅ {split}: {primary_metric}={data.get(primary_metric, 'N/A')}")
    
    print(f"\n📈 总计处理了 {len(all_results)} 条结果记录")
    return all_results

def create_pivot_table(df: pd.DataFrame, output_file: Path):
    """
    创建透视表：横轴=模型，纵轴=数据集×指标
    """
    # 只使用test split或val split的结果用于透视表
    pivot_df = df[df['split'].isin(['test', 'val'])].copy()
    
    # 如果同时有test和val，优先使用test
    if 'test' in pivot_df['split'].values:
        pivot_df = pivot_df[pivot_df['split'] == 'test']
    
    # 创建复合行索引: dataset + primary_metric
    pivot_df['dataset_metric'] = pivot_df['dataset'] + '_' + pivot_df['primary_metric']
    
    # 创建透视表
    pivot_table = pivot_df.pivot_table(
        index='dataset_metric',
        columns='model', 
        values='primary_value',
        aggfunc='first'  # 如果有重复，取第一个值
    )
    
    # 保存透视表
    pivot_table.to_csv(output_file)
    print(f"💾 透视表已保存到: {output_file}")
    
    # 显示透视表预览
    print(f"\n📋 透视表预览 (前10行):")
    print(pivot_table.head(10).to_string())
    
    if len(pivot_table) > 10:
        print(f"... (共 {len(pivot_table)} 行)")

def main():
    parser = argparse.ArgumentParser(description='聚合Graph-Mamba实验结果')
    parser.add_argument('--output-dir', default='outputs', 
                        help='输出目录 (默认: outputs)')
    parser.add_argument('--results-dir', default='results',
                        help='结果目录 (默认: results)')
    
    args = parser.parse_args()
    
    # 设置路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_dir = project_root / args.results_dir
    output_dir = project_root / args.output_dir
    
    # 检查results目录是否存在
    if not results_dir.exists():
        print(f"❌ 错误: 结果目录不存在 - {results_dir}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    print(f"🎯 Graph-Mamba 实验结果聚合")
    print(f"📂 结果目录: {results_dir}")
    print(f"📁 输出目录: {output_dir}")
    print()
    
    try:
        # 聚合所有结果
        all_results = aggregate_all_results(results_dir)
        
        if not all_results:
            print("❌ 没有找到任何有效的实验结果")
            sys.exit(1)
        
        # 转换为DataFrame
        df = pd.DataFrame(all_results)
        
        # 保存详细结果
        summary_file = output_dir / "benchmark_summary.csv"
        df.to_csv(summary_file, index=False)
        print(f"\n💾 详细结果已保存到: {summary_file}")
        
        # 显示汇总统计
        print(f"\n📊 结果汇总:")
        print(f"   总记录数: {len(df)}")
        print(f"   模型数量: {df['model'].nunique()}")
        print(f"   数据集数量: {df['dataset'].nunique()}")
        
        model_counts = df['model'].value_counts()
        print(f"\n🤖 各模型结果数量:")
        for model, count in model_counts.items():
            print(f"   {model}: {count} 条")
        
        # 创建透视表
        pivot_file = output_dir / "benchmark_pivot.csv"
        create_pivot_table(df, pivot_file)
        
        print(f"\n🎉 结果聚合完成！")
        print(f"📝 生成的文件:")
        print(f"   - {summary_file}: 详细结果记录")
        print(f"   - {pivot_file}: 透视表（模型×数据集）")
        
    except Exception as e:
        print(f"❌ 脚本执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
