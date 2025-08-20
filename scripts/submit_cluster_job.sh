#!/bin/bash
#SBATCH --job-name=graph_mamba_benchmark
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=1-42
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Graph-Mamba 批量基准测试作业脚本
# 
# 使用方法:
# 1. 运行 python scripts/generate_cluster_cmds.py 生成 commands.list
# 2. 创建logs目录: mkdir -p logs  
# 3. 提交作业: sbatch scripts/submit_cluster_job.sh
#
# 注意: 请根据你的集群配置调整SBATCH参数

echo "=== Graph-Mamba 基准测试作业 ==="
echo "作业ID: ${SLURM_JOB_ID}"
echo "数组任务ID: ${SLURM_ARRAY_TASK_ID}"
echo "节点: ${SLURMD_NODENAME}"
echo "开始时间: $(date)"
echo ""

# 检查CUDA环境
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
else
    echo "⚠️  警告: 未找到nvidia-smi，请确保在GPU节点上运行"
fi

# 检查必要文件
if [ ! -f "commands.list" ]; then
    echo "❌ 错误: 未找到commands.list文件"
    echo "请先运行: python scripts/generate_cluster_cmds.py"
    exit 1
fi

# 检查Python环境
if ! python -c "import torch; print('PyTorch版本:', torch.__version__)" 2>/dev/null; then
    echo "❌ 错误: Python环境不正确，无法导入PyTorch"
    exit 1
fi

# 读取对应的命令
command=$(sed -n "${SLURM_ARRAY_TASK_ID}p" commands.list)

if [ -z "$command" ]; then
    echo "❌ 错误: 无法读取第 ${SLURM_ARRAY_TASK_ID} 行命令"
    exit 1
fi

echo "📋 执行命令: $command"
echo "⏰ 执行时间: $(date)"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# 执行命令
eval $command
exit_code=$?

echo ""
echo "⏰ 完成时间: $(date)"
echo "📊 退出状态: $exit_code"

if [ $exit_code -eq 0 ]; then
    echo "✅ 任务 ${SLURM_ARRAY_TASK_ID} 执行成功"
else
    echo "❌ 任务 ${SLURM_ARRAY_TASK_ID} 执行失败 (退出码: $exit_code)"
fi

exit $exit_code
