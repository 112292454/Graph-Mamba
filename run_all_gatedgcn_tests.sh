#!/bin/bash

# GatedGCN批量测试脚本
echo "开始批量运行GatedGCN在所有数据集上的测试"
echo "时间: $(date)"
echo "========================================="

# 定义数据集配置文件列表
datasets=(
    # "dd-exported-GatedGCN.yaml"
    # "proteins-exported-GatedGCN.yaml" 
    # "zinc-exported-GatedGCN.yaml"
    # "aqsol-exported-GatedGCN.yaml"
    # "colors3-exported-GatedGCN.yaml"
    # "mutagenicity-exported-GatedGCN.yaml"
    # "coildel-exported-GatedGCN.yaml"
    # "dblp-exported-GatedGCN.yaml"
    # "twitter-exported-GatedGCN.yaml"
    # "synthetic-exported-GatedGCN.yaml"
    # "molhiv-exported-GatedGCN.yaml"
    "peptides-func-exported-GatedGCN.yaml"
    "peptides-struct-exported-GatedGCN.yaml"
)

# 记录结果
success_count=0
failed_count=0
success_list=()
failed_list=()

# 逐个运行
for config in "${datasets[@]}"; do
    echo ""
    echo "========================================="
    echo "正在运行: $config"
    echo "时间: $(date)"
    echo "========================================="
    
    # 运行训练
    python main.py --cfg configs/GatedGCN/$config
    
    # 检查返回码
    if [ $? -eq 0 ]; then
        echo "✅ 成功: $config"
        success_list+=("$config")
        ((success_count++))
    else
        echo "❌ 失败: $config"
        failed_list+=("$config")
        ((failed_count++))
    fi
done

# 打印最终结果
echo ""
echo "========================================="
echo "批量测试完成!"
echo "时间: $(date)"
echo "========================================="
echo "成功: $success_count 个数据集"
echo "失败: $failed_count 个数据集"
echo ""

if [ ${#success_list[@]} -gt 0 ]; then
    echo "✅ 成功的数据集:"
    for dataset in "${success_list[@]}"; do
        echo "  - $dataset"
    done
fi

if [ ${#failed_list[@]} -gt 0 ]; then
    echo ""
    echo "❌ 失败的数据集:"
    for dataset in "${failed_list[@]}"; do
        echo "  - $dataset"
    done
fi

echo ""
echo "详细结果可在results/目录中查看"
echo "========================================="
