#!/usr/bin/env python3
"""
聚合实验结果脚本

此脚本从 results/ 目录聚合不同模型、不同数据集的实验结果，
生成便于分析的CSV文件和method×dataset透视表。

使用方法:
    python scripts/aggregate_results.py [--output-dir outputs]
    
输出:
    - benchmark_summary.csv: 详细结果记录
    - benchmark_pivot.csv: 透视表（横轴=模型，纵轴=数据集×指标）
    - MAE_best.csv: 回归任务MAE表格（模型×数据集）
    - Acc_best.csv: 分类任务准确率表格（模型×数据集）
"""

import os
import json
import sys
import argparse
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
import re
from collections import defaultdict

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

def get_model_name_mapping() -> Dict[str, str]:
    """
    映射目录名中的模型名到标准模型名
    """
    return {
        'GatedGCN': 'GatedGCN',
        'GPS': 'GraphGPS', 
        'EX': 'Exphormer',
        'Mamba': 'GraphMamba',
        # 可能的其他映射
        'Exphormer': 'Exphormer',
        'GraphGPS': 'GraphGPS',
        'GraphMamba': 'GraphMamba'
    }

def get_model_order() -> List[str]:
    """
    返回模型的排序顺序
    """
    return ['GatedGCN', 'GraphGPS', 'Exphormer', 'GraphMamba']

def get_dataset_classification() -> Tuple[List[str], List[str]]:
    """
    返回回归和分类数据集的列表，按照指定顺序
    """
    regression_datasets = ['qm9', 'zinc', 'aqsol', 'peptides-struct']
    classification_datasets = ['colors3', 'proteins', 'synthetic', 'mutagenicity', 
                             'coildel', 'dblp', 'dd', 'twitter', 'molhiv', 'peptides-func']
    return regression_datasets, classification_datasets

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

def determine_task_type(dataset: str, metric_mapping: Dict[str, str]) -> str:
    """
    根据数据集确定任务类型
    """
    base_dataset = dataset.split('-')[0] if '-' in dataset else dataset
    metric = metric_mapping.get(base_dataset, metric_mapping.get(dataset, 'accuracy'))
    
    if metric == 'mae':
        return 'regression'
    else:
        return 'classification'

def load_json_safely(file_path: Path) -> Optional[Dict]:
    """安全地加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  警告: 无法读取 {file_path}: {e}")
        return None

def extract_best_metric_value(result_dir: Path, dataset: str, metric_mapping: Dict[str, str]) -> Optional[float]:
    """
    从结果目录提取最佳指标值
    
    Args:
        result_dir: 结果目录路径
        dataset: 数据集名称
        metric_mapping: 指标映射
        
    Returns:
        最佳指标值，如果没有找到则返回None
    """
    # 确定目标指标
    base_dataset = dataset.split('-')[0] if '-' in dataset else dataset
    target_metric = metric_mapping.get(base_dataset, metric_mapping.get(dataset, 'accuracy'))
    
    best_value = None
    
    # 优先尝试读取agg/test/best.json (测试集最佳结果)
    test_best_file = result_dir / "agg" / "test" / "best.json"
    if test_best_file.exists():
        data = load_json_safely(test_best_file)
        if data and target_metric in data:
            return data[target_metric]
    
    # 尝试读取agg/val/best.json (验证集最佳结果)
    val_best_file = result_dir / "agg" / "val" / "best.json"
    if val_best_file.exists():
        data = load_json_safely(val_best_file)
        if data and target_metric in data:
            best_value = data[target_metric]
    
    # 如果没有agg结果，尝试从seed目录读取
    if best_value is None:
        # 查找seed目录（通常是数字命名）
        seed_dirs = [d for d in result_dir.iterdir() 
                    if d.is_dir() and d.name.isdigit()]
        
        if seed_dirs:
            # 从所有seed中找最佳值
            seed_values = []
            for seed_dir in seed_dirs:
                # 尝试读取test/stats.json的最后一行
                test_stats_file = seed_dir / "test" / "stats.json"
                if test_stats_file.exists():
                    try:
                        with open(test_stats_file, 'r') as f:
                            lines = f.readlines()
                            if lines:
                                last_line = lines[-1].strip()
                                if last_line:
                                    data = json.loads(last_line)
                                    if target_metric in data:
                                        seed_values.append(data[target_metric])
                    except Exception as e:
                        continue
            
            if seed_values:
                # 根据指标类型选择最佳值
                if target_metric == 'mae':  # 越小越好
                    best_value = min(seed_values)
                else:  # accuracy, auc, ap 越大越好
                    best_value = max(seed_values)
    
    return best_value

def extract_metrics_from_result(result_dir: Path) -> Dict:
    """
    从结果目录提取关键指标 (保留原函数用于详细分析)
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

def collect_best_results_for_tables(results_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    收集所有实验的最佳结果，组织成用于创建method×dataset表格的格式
    
    Returns:
        {task_type: {model: {dataset: best_value}}}
    """
    model_mapping = get_model_name_mapping()
    dataset_metrics = get_dataset_metric_mapping()
    
    # 组织结果: {task_type: {model: {dataset: value}}}
    results = defaultdict(lambda: defaultdict(dict))
    
    print("🔍 扫描结果目录收集最佳结果...")
    
    # 遍历results目录下的所有子目录
    for result_dir in results_dir.iterdir():
        if not result_dir.is_dir():
            continue
            
        # 跳过特殊目录
        if result_dir.name.startswith('.') or result_dir.name == 'parallel_test_logs':
            continue
        
        print(f"📊 处理: {result_dir.name}")
        
        # 解析目录名获取数据集和模型
        dataset, model_raw = parse_result_dir_name(result_dir.name)
        
        # 映射模型名
        model = model_mapping.get(model_raw, model_raw)
        
        # 提取最佳指标值
        best_value = extract_best_metric_value(result_dir, dataset, dataset_metrics)
        
        if best_value is None:
            print(f"   ⚠️  未找到有效结果")
            continue
        
        # 确定任务类型
        task_type = determine_task_type(dataset, dataset_metrics)
        
        # 存储结果
        results[task_type][model][dataset] = best_value
        
        # 确定指标名称用于显示
        base_dataset = dataset.split('-')[0] if '-' in dataset else dataset
        metric_name = dataset_metrics.get(base_dataset, dataset_metrics.get(dataset, 'accuracy'))
        
        print(f"   ✅ {model} - {dataset}: {metric_name}={best_value:.4f}")
    
    return dict(results)

def aggregate_all_results(results_dir: Path) -> List[Dict]:
    """
    聚合results目录下所有实验结果 (保留原函数用于详细分析)
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

def create_method_dataset_tables(results: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path):
    """
    创建method×dataset表格
    
    Args:
        results: {task_type: {model: {dataset: best_value}}} 格式的结果
        output_dir: 输出目录
    """
    regression_datasets, classification_datasets = get_dataset_classification()
    model_order = get_model_order()
    
    for task_type, task_results in results.items():
        if task_type == 'regression':
            datasets = regression_datasets
            table_name = "MAE_best.csv"
            metric_display = "MAE"
        else:  # classification
            datasets = classification_datasets
            table_name = "Acc_best.csv" 
            metric_display = "Accuracy"
        
        print(f"\n📊 创建{metric_display}表格...")
        
        # 创建表格数据
        table_data = []
        
        # 按照指定顺序处理每个模型
        for model in model_order:
            if model not in task_results:
                continue
                
            row = {'Model': model}
            model_results = task_results[model]
            
            for dataset in datasets:
                if dataset in model_results:
                    value = model_results[dataset]
                    row[dataset] = f"{value:.4f}"
                else:
                    row[dataset] = "N/A"
            
            table_data.append(row)
        
        # 如果有数据，创建并保存表格
        if table_data:
            df = pd.DataFrame(table_data)
            output_file = output_dir / table_name
            df.to_csv(output_file, index=False)
            print(f"💾 {metric_display}表格已保存到: {output_file}")
            
            # 显示表格预览
            print(f"\n📋 {metric_display}表格预览:")
            print(df.to_string(index=False))
        else:
            print(f"⚠️  没有找到{task_type}任务的有效结果")

def create_pivot_table(df: pd.DataFrame, output_file: Path):
    """
    创建透视表：横轴=模型，纵轴=数据集×指标 (保留原函数)
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
        # 收集最佳结果用于method×dataset表格
        best_results = collect_best_results_for_tables(results_dir)
        
        if not best_results:
            print("❌ 没有找到任何有效的最佳结果")
            sys.exit(1)
        
        # 创建method×dataset表格
        create_method_dataset_tables(best_results, output_dir)
        
        # 聚合所有详细结果（保留原有功能）
        print(f"\n" + "="*50)
        print("📈 生成详细结果分析...")
        all_results = aggregate_all_results(results_dir)
        
        if all_results:
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
            print(f"   - MAE_best.csv: 回归任务MAE表格（模型×数据集）")
            print(f"   - Acc_best.csv: 分类任务准确率表格（模型×数据集）")
            print(f"   - {summary_file}: 详细结果记录")
            print(f"   - {pivot_file}: 透视表（模型×数据集）")
        else:
            print(f"\n🎉 method×dataset表格生成完成！")
            print(f"📝 生成的文件:")
            print(f"   - MAE_best.csv: 回归任务MAE表格（模型×数据集）")
            print(f"   - Acc_best.csv: 分类任务准确率表格（模型×数据集）")
        
    except Exception as e:
        print(f"❌ 脚本执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
