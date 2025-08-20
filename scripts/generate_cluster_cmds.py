#!/usr/bin/env python3
"""
批量生成集群计算命令脚本

此脚本根据已有的配置文件生成42条独立可运行的命令（3个模型 × 14个数据集），
用于提交到集群进行批量计算。

使用方法:
    python scripts/generate_cluster_cmds.py
    
输出:
    commands.list - 包含42条可独立执行的命令
"""

import os
import sys
from pathlib import Path

def main():
    # 设置基础路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # 检查是否在正确的项目根目录
    if not (project_root / "main.py").exists():
        print("❌ 错误: 找不到 main.py，请确保在 Graph-Mamba 项目根目录运行此脚本")
        sys.exit(1)
    
    # 定义模型和数据集
    models = [
        "GatedGCN",
        "GPS",
        "Mamba", 
        "Exphormer_LRGB"
    ]
    
    # 基于文档确定的14个数据集
    datasets = [
        "dd",           # DD 蛋白质结构分类
        "proteins",     # PROTEINS 蛋白质功能分类
        "zinc",         # ZINC 分子溶解度预测
        "aqsol",        # AQSOL 分子水溶性预测
        "colors3",      # COLORS3 图着色问题
        "mutagenicity", # MUTAGENICITY 化合物致突变性
        "coildel",      # COILDEL 蛋白质螺旋结构
        "dblp",         # DBLP 学术网络分类
        "twitter",      # TWITTER 社交网络分类
        "synthetic",    # SYNTHETIC 人工合成图
        "molhiv",       # MOLHIV HIV抑制剂预测
        "peptides-func", # PEPTIDES-FUNC 多标签功能预测
        "peptides-struct", # PEPTIDES-STRUCT 多目标结构预测
        "qm9"           # QM9 量子化学分子属性预测
    ]
    
    commands = []
    missing_configs = []
    
    print("🔍 检查配置文件并生成命令...")
    print(f"📊 目标: {len(models)} 个模型 × {len(datasets)} 个数据集 = {len(models) * len(datasets)} 条命令")
    print()
    
    # 为每个模型和数据集生成命令
    for model in models:
        model_dir = project_root / f"configs/Benchmark/{model}"
        
        if not model_dir.exists():
            print(f"⚠️  警告: 模型目录不存在 - {model_dir}")
            continue
        
        for dataset in datasets:
            # 根据观察到的命名规律构建配置文件名
            config_file = f"{dataset}-exported-{model}.yaml"
            
            # 对于Exphormer_LRGB，文件名可能使用EX简写
            if model == "Exphormer_LRGB":
                config_file_alt = f"{dataset}-exported-EX.yaml"
                config_path = model_dir / config_file
                config_path_alt = model_dir / config_file_alt
                
                if config_path_alt.exists():
                    config_file = config_file_alt
                    config_path = config_path_alt
            else:
                config_path = model_dir / config_file
            
            # 检查配置文件是否存在
            if config_path.exists():
                relative_config = f"configs/Benchmark/{model}/{config_file}"
                command = f"python main.py --cfg {relative_config} --repeat 3"
                commands.append(command)
                print(f"✅ {model:15} × {dataset:15} -> {config_file}")
            else:
                missing_configs.append(f"{model}/{config_file}")
                print(f"❌ {model:15} × {dataset:15} -> 配置文件缺失: {config_file}")
    
    print()
    print(f"📈 统计结果:")
    print(f"   ✅ 成功生成命令: {len(commands)} 条")
    print(f"   ❌ 缺失配置文件: {len(missing_configs)} 个")
    
    if missing_configs:
        print(f"\n🚨 缺失的配置文件:")
        for config in missing_configs:
            print(f"   - {config}")
    
    # 写入命令文件
    output_file = project_root / "commands.list"
    with open(output_file, 'w', encoding='utf-8') as f:
        for cmd in commands:
            f.write(cmd + '\n')
    
    print(f"\n💾 命令已保存到: {output_file}")
    print(f"📝 文件包含 {len(commands)} 条可独立执行的命令")
    
    if len(commands) > 0:
        print(f"\n📖 使用示例:")
        print(f"   # 查看生成的命令")
        print(f"   cat commands.list")
        print(f"   ")
        print(f"   # 在集群上批量提交（示例）")
        print(f"   # 使用你的集群调度系统，如：")
        print(f"   # sbatch --array=1-{len(commands)} submit_job.sh")
        print(f"   # 其中 submit_job.sh 读取第 $SLURM_ARRAY_TASK_ID 行命令执行")
    
    return len(commands)

if __name__ == "__main__":
    try:
        num_commands = main()
        if num_commands == 42:
            print(f"\n🎉 完美！成功生成了全部 {num_commands} 条命令")
        else:
            print(f"\n⚠️  注意：生成了 {num_commands} 条命令，期望是42条")
            sys.exit(1)
    except Exception as e:
        print(f"❌ 脚本执行出错: {e}")
        sys.exit(1)
