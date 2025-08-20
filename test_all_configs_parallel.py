#!/usr/bin/env python3
"""
多GPU并行测试所有Benchmark配置文件
支持断点续传，多GPU并行加速
"""

import os
import sys
import subprocess
import yaml
import time
from pathlib import Path
import tempfile
import shutil
import json
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# 每块GPU的并发槽位数（默认1）。可通过环境变量 SLOTS_PER_GPU 调整，例如 SLOTS_PER_GPU=2
SLOTS_PER_GPU = int(os.environ.get('SLOTS_PER_GPU', '1'))

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    print("⚠️  GPUtil未安装，将使用备用GPU检测方法")

def _validate_gpu_id(gpu_id: int) -> bool:
    """通过启动一个最小子进程，验证指定CUDA_VISIBLE_DEVICES下torch是否能看到GPU。"""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    try:
        p = subprocess.run(
            [sys.executable, "-c", "import torch;print(int(torch.cuda.is_available()), torch.cuda.device_count())"],
            capture_output=True, text=True, timeout=5, env=env
        )
        if p.returncode == 0:
            out = (p.stdout or "").strip()
            parts = out.split()
            if len(parts) >= 2:
                is_avail = parts[0] == '1'
                cnt_ok = parts[1].isdigit() and int(parts[1]) >= 1
                return is_avail and cnt_ok
    except Exception:
        pass
    return False

def get_available_gpus():
    """获取可用且可被当前Python进程实际访问的GPU列表（逐个校验）。"""
    candidates = []
    if HAS_GPUTIL:
        try:
            gpus = GPUtil.getGPUs()
            # 先按空闲比例过滤候选
            candidates = [gpu.id for gpu in gpus if gpu.memoryFree / gpu.memoryTotal > 0.3]
        except Exception:
            candidates = []
    if not candidates:
        # 备用：nvidia-smi 列举
        try:
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_lines = [line for line in result.stdout.split('\n') if 'GPU' in line]
                candidates = list(range(len(gpu_lines)))
        except Exception:
            candidates = []

    # 逐个验证
    valid = []
    for gid in candidates or [0]:
        if _validate_gpu_id(gid):
            valid.append(gid)
        else:
            print(f"⚠️  跳过不可用GPU id={gid} (torch不可见)")

    if not valid:
        print("⚠️  未找到可用GPU，将使用CPU模式（CUDA不可见）")
        return []
    return valid

class ProgressManager:
    """进度管理器，支持断点续传，线程安全"""
    def __init__(self, progress_file="test_progress_parallel.json"):
        self.progress_file = progress_file
        self.completed_tasks = set()
        self.failed_tasks = {}
        self.lock = threading.Lock()
        self.load_progress()
    
    def load_progress(self):
        """加载之前的进度"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.completed_tasks = set(data.get('completed', []))
                    self.failed_tasks = data.get('failed', {})
                    print(f"📂 发现进度文件: {len(self.completed_tasks)} 个任务已完成")
            except Exception as e:
                print(f"⚠️  进度文件读取失败: {e}")
        else:
            print("🆕 开始新的并行测试")
    
    def save_progress(self):
        """保存当前进度"""
        with self.lock:
            data = {
                'completed': list(self.completed_tasks),
                'failed': self.failed_tasks
            }
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    def is_completed(self, config_key):
        """检查任务是否已完成"""
        with self.lock:
            return config_key in self.completed_tasks
    
    def mark_completed(self, config_key):
        """标记任务为已完成"""
        with self.lock:
            self.completed_tasks.add(config_key)
            # 如果之前失败过，移除失败记录
            if config_key in self.failed_tasks:
                del self.failed_tasks[config_key]
            # 延迟保存，避免频繁IO卡顿
    
    def mark_failed(self, config_key, error_info):
        """标记任务为失败"""
        with self.lock:
            self.failed_tasks[config_key] = {
                'error': error_info[:200] + "..." if len(error_info) > 200 else error_info,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            # 延迟保存，避免频繁IO卡顿
    
    def get_stats(self):
        """获取统计信息"""
        with self.lock:
            return len(self.completed_tasks), len(self.failed_tasks)

def modify_config_epochs(original_config_path, epochs=2):
    """修改配置文件中的max_epoch参数"""
    with open(original_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改epochs
    if 'optim' not in config:
        config['optim'] = {}
    config['optim']['max_epoch'] = epochs
    
    # 创建临时配置文件
    temp_dir = tempfile.mkdtemp()
    temp_config_path = os.path.join(temp_dir, os.path.basename(original_config_path))
    
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return temp_config_path, temp_dir

def run_test(config_path, config_key, gpu_id=None, timeout=300):
    """运行单个配置文件的测试"""
    try:
        # 修改配置文件epochs
        temp_config_path, temp_dir = modify_config_epochs(config_path, epochs=2)
        
        start_time = time.time()
        
        # 设置环境变量指定GPU
        env = os.environ.copy()
        env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        if gpu_id is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # 调试日志：打印当前任务将使用的CUDA可见设备
        cfg_name = os.path.basename(config_path)
        print(f"[GPU{gpu_id}] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<none>')} | cfg={cfg_name}")

        # 额外快速检查：当前环境下的torch.cuda可用性
        try:
            cuda_check = subprocess.run(
                [sys.executable, "-c", "import torch;\nprint(f'CUDA_AVAIL={torch.cuda.is_available()} NUM={torch.cuda.device_count()}')"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd="/home/gzy/py/Graph-Mamba",
                env=env
            )
            cuda_ok = False
            if cuda_check.returncode == 0:
                msg = cuda_check.stdout.strip()
                print(f"[GPU{gpu_id}] torch.cuda check: {msg}")
                # 解析可用性
                try:
                    parts = msg.replace('CUDA_AVAIL=','').replace('NUM=','').split()
                    if len(parts) >= 2:
                        avail = parts[0] in ('True','1')
                        num = int(parts[1]) if parts[1].isdigit() else 0
                        cuda_ok = (avail and num >= 1)
                except Exception:
                    cuda_ok = False
            else:
                print(f"[GPU{gpu_id}] torch.cuda check failed: {cuda_check.stderr.strip()}")
        except Exception as _e:
            print(f"[GPU{gpu_id}] torch.cuda check exception: {_e}")

        # 强制要求GPU可用
        if not locals().get('cuda_ok', False):
            return {
                'status': 'FAILED',
                'runtime': '0s',
                'result': None,
                'error': f"CUDA is not available for GPU {gpu_id} (cfg={cfg_name}). Abort."
            }
        
        # 运行测试
        cmd = [sys.executable, "main.py", "--cfg", temp_config_path, "--repeat", "1"]

        # 打印与记录将要执行的命令
        cmd_str = ' '.join(cmd)
        print(f"[GPU{gpu_id}] ▶ CMD: {cmd_str}")

        # 日志文件
        logs_dir = os.path.join('results', 'parallel_test_logs')
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        safe_key = config_key.replace('/', '_')
        log_path = os.path.join(logs_dir, f"{timestamp}_{safe_key}.log")
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd="/home/gzy/py/Graph-Mamba",
            env=env
        )
        
        end_time = time.time()
        runtime = end_time - start_time
        
        # 清理临时文件
        shutil.rmtree(temp_dir)
        
        # 将stdout/stderr写入日志文件
        try:
            with open(log_path, 'w', encoding='utf-8') as lf:
                lf.write(f"# CMD: {cmd_str}\n")
                lf.write(f"# CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}\n")
                lf.write("\n# ===== STDOUT =====\n")
                lf.write(result.stdout or '')
                lf.write("\n# ===== STDERR =====\n")
                lf.write(result.stderr or '')
        except Exception:
            pass

        if result.returncode == 0:
            # 从输出中提取最终指标
            lines = result.stdout.split('\n')
            best_line = None
            for line in lines:
                if "Best so far:" in line and "epoch 1" in line:
                    best_line = line.strip()
                    break
            
            return {
                'status': 'SUCCESS',
                'runtime': f"{runtime:.1f}s",
                'result': best_line if best_line else "训练完成",
                'error': None,
                'log': log_path
            }
        else:
            # 返回完整错误信息用于调试
            full_error = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            return {
                'status': 'FAILED', 
                'runtime': f"{runtime:.1f}s",
                'result': None,
                'error': full_error,
                'log': log_path
            }
            
    except subprocess.TimeoutExpired:
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
        return {
            'status': 'TIMEOUT',
            'runtime': f"{timeout}s+",
            'result': None,
            'error': f"超时 (>{timeout}s)"
        }
    except Exception as e:
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
        return {
            'status': 'ERROR',
            'runtime': "0s",
            'result': None,
            'error': str(e)
        }

def _torch_cuda_ok(env):
    """在给定环境变量下快速检测 torch 是否能看到 CUDA 与设备数量。"""
    try:
        p = subprocess.run(
            [sys.executable, "-c", "import torch;\nprint(f'CUDA_AVAIL={torch.cuda.is_available()} NUM={torch.cuda.device_count()}')"],
            capture_output=True, text=True, timeout=10, cwd="/home/gzy/py/Graph-Mamba", env=env
        )
        if p.returncode == 0:
            msg = (p.stdout or '').strip()
            parts = msg.replace('CUDA_AVAIL=','').replace('NUM=','').split()
            if len(parts) >= 2:
                avail = parts[0] in ('True','1')
                num = int(parts[1]) if parts[1].isdigit() else 0
                return avail and num >= 1, msg
        return False, (p.stderr or '').strip()
    except Exception as e:
        return False, str(e)

def launch_task_nonblocking(config_path, config_key, gpu_id, timeout):
    """创建临时cfg并以非阻塞子进程方式启动任务，将输出直接写到当前终端。"""
    temp_config_path, temp_dir = modify_config_epochs(config_path, epochs=2)
    env = os.environ.copy()
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    cfg_name = os.path.basename(config_path)
    print(f"[GPU{gpu_id}] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')} | cfg={cfg_name}")
    ok, cuda_msg = _torch_cuda_ok(env)
    print(f"[GPU{gpu_id}] torch.cuda check: {cuda_msg}")
    if not ok:
        return None, {
            'status': 'FAILED',
            'runtime': '0s',
            'result': None,
            'error': f"CUDA not available for GPU {gpu_id}: {cuda_msg}",
            'log': None
        }, temp_dir

    cmd = [sys.executable, "main.py", "--cfg", temp_config_path, "--repeat", "1"]
    cmd_str = ' '.join(cmd)
    print(f"[GPU{gpu_id}] ▶ CMD: {cmd_str}")

    proc = subprocess.Popen(
        cmd,
        cwd="/home/gzy/py/Graph-Mamba",
        env=env,
        stdout=None,  # 继承父进程stdout
        stderr=None,  # 继承父进程stderr
        text=True
    )
    start_time = time.time()
    return {'proc': proc, 'start': start_time,
            'temp_dir': temp_dir, 'timeout': timeout, 'task': (config_path, config_key)}, None, temp_dir

def find_all_benchmark_configs():
    """找到所有需要测试的配置文件"""
    benchmark_dir = Path("/home/gzy/py/Graph-Mamba/configs/Benchmark")
    
    configs = []
    
    # GPS 配置文件
    gps_dir = benchmark_dir / "GPS"
    for config_file in gps_dir.glob("*-exported-GPS.yaml"):
        configs.append(("GPS", str(config_file)))
    
    # Mamba 配置文件  
    mamba_dir = benchmark_dir / "Mamba"
    for config_file in mamba_dir.glob("*-exported-Mamba.yaml"):
        configs.append(("Mamba", str(config_file)))
        
    # Exphormer_LRGB 配置文件
    exphormer_dir = benchmark_dir / "Exphormer_LRGB"  
    for config_file in exphormer_dir.glob("*-exported-EX.yaml"):
        configs.append(("Exphormer_LRGB", str(config_file)))
    
    return sorted(configs)

def worker_function(gpu_id, tasks, progress_manager, results_dict, worker_id):
    """Worker函数：在指定GPU上处理任务"""
    print(f"🔄 Worker {worker_id} 启动，使用GPU {gpu_id}")
    
    for task in tasks:
        model_name, config_path, dataset_name, config_key = task
        
        try:
            print(f"[GPU{gpu_id}] 🔧 测试 {config_key}")
            
            result = run_test(config_path, config_key, gpu_id=gpu_id, timeout=300)
            
            if result['status'] == 'SUCCESS':
                progress_manager.mark_completed(config_key)
                print(f"[GPU{gpu_id}] ✅ {config_key} 完成 ({result['runtime']}) | log={result.get('log','-')}")
            else:
                progress_manager.mark_failed(config_key, result['error'])
                print(f"[GPU{gpu_id}] ❌ {config_key} 失败 ({result['runtime']}) | log={result.get('log','-')}")
                
            results_dict[config_key] = result
            
        except Exception as e:
            error_msg = str(e)
            progress_manager.mark_failed(config_key, error_msg)
            results_dict[config_key] = {
                'status': 'ERROR',
                'runtime': '0s', 
                'result': None,
                'error': error_msg
            }
            print(f"[GPU{gpu_id}] 💥 {config_key} 异常: {error_msg}")
    
    print(f"🏁 Worker {worker_id} (GPU {gpu_id}) 完成")

def run_all_tests_parallel():
    """使用多GPU并行运行所有测试"""
    # 检测可用GPU
    available_gpus = get_available_gpus()
    num_workers = min(len(available_gpus), 4)  # 最多使用4个GPU
    print(f"🔍 检测到 {len(available_gpus)} 个可用GPU: {available_gpus}")
    if num_workers == 0:
        print("❌ 未检测到可用GPU（或CUDA在当前进程不可见），已终止。")
        return False
    print(f"💻 将使用 {num_workers} 个worker并行执行")
    
    configs = find_all_benchmark_configs()
    progress = ProgressManager()
    results_dict = {}
    total_start = time.time()
    
    completed_count, failed_count = progress.get_stats()
    if completed_count > 0:
        print(f"📊 进度状态: 已完成 {completed_count} 个, 失败 {failed_count} 个")
        if failed_count > 0:
            print("⚠️  之前失败的任务:")
            for task, info in progress.failed_tasks.items():
                print(f"   {task}: {info['error']} ({info['timestamp']})")
        print()
    
    total_tasks = len(configs)
    remaining_tasks = []
    for model_name, config_path in configs:
        dataset_name = os.path.basename(config_path).split('-exported-')[0]
        config_key = f"{model_name}-{dataset_name}"
        if not progress.is_completed(config_key):
            remaining_tasks.append((model_name, config_path, dataset_name, config_key))
    
    print(f"📝 总任务: {total_tasks}, 剩余: {len(remaining_tasks)}")
    if len(remaining_tasks) == 0:
        print("🎉 所有任务都已完成!")
        return True
    print()
    
    # 非阻塞调度：每个GPU可配置多个并发槽
    gpu_slots = {gid: [None for _ in range(SLOTS_PER_GPU)] for gid in available_gpus}
    pending = list(remaining_tasks)
    finished = 0
    total = len(pending)
    total_slots = sum(len(v) for v in gpu_slots.values())
    print(f"🚀 非阻塞多进程调度启动，GPU数={len(gpu_slots)} 槽位总数={total_slots} (SLOTS_PER_GPU={SLOTS_PER_GPU})\n")

    def any_running():
        for slots in gpu_slots.values():
            for slot in slots:
                if slot is not None:
                    return True
        return False

    while pending or any_running():
        # 启动可用槽的新任务
        for gid, slots in gpu_slots.items():
            for idx in range(len(slots)):
                if slots[idx] is None and pending:
                    task = pending.pop(0)
                    model_name, config_path, dataset_name, config_key = task
                    print(f"[GPU{gid}#slot{idx}] 🔧 启动 {config_key}")
                    # 针对耗时数据集提高超时阈值，避免误判失败
                    key_lower = config_key.lower()
                    if 'twitter' in key_lower:
                        tmo = 1800  # Twitter 首个 epoch 较慢
                    elif 'peptides' in key_lower:
                        tmo = 1200
                    elif 'aqsol' in key_lower:
                        tmo = 900
                    else:
                        tmo = 300
                    procRec, early_result, temp_dir = launch_task_nonblocking(config_path, config_key, gid, timeout=tmo)
                    if early_result is not None:
                        results_dict[config_key] = early_result
                        try:
                            progress.mark_failed(config_key, early_result['error'])
                        except Exception as _pe:
                            print(f"[GPU{gid}#slot{idx}] ⚠️ 进度记录失败(启动失败): {config_key} -> {_pe}")
                        print(f"[GPU{gid}#slot{idx}] ❌ {config_key} 启动失败: {early_result['error']}")
                        try:
                            shutil.rmtree(temp_dir)
                        except Exception:
                            pass
                    else:
                        slots[idx] = procRec

        # 轮询已启动进程
        for gid, slots in gpu_slots.items():
            for idx, rec in enumerate(list(slots)):
                if rec is None:
                    continue
                proc = rec['proc']
                ret = proc.poll()
                if ret is None:
                    # 检查超时
                    if time.time() - rec['start'] > rec['timeout']:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        ret = -9
                    else:
                        continue

                config_path, config_key = rec['task']
                runtime = f"{(time.time() - rec['start']):.1f}s"
                if ret == 0:
                    results_dict[config_key] = {
                        'status': 'SUCCESS', 'runtime': runtime, 'result': '训练完成', 'error': None, 'log': None
                    }
                    try:
                        progress.mark_completed(config_key)
                    except Exception as _pe:
                        print(f"[GPU{gid}#slot{idx}] ⚠️ 进度记录失败(完成): {config_key} -> {_pe}")
                    print(f"[GPU{gid}#slot{idx}] ✅ 完成 {config_key} ({runtime})")
                else:
                    results_dict[config_key] = {
                        'status': 'FAILED', 'runtime': runtime, 'result': None,
                        'error': f"Process exited with code {ret}.",
                        'log': None
                    }
                    try:
                        progress.mark_failed(config_key, results_dict[config_key]['error'])
                    except Exception as _pe:
                        print(f"[GPU{gid}#slot{idx}] ⚠️ 进度记录失败(失败): {config_key} -> {_pe}")
                    print(f"[GPU{gid}#slot{idx}] ❌ 失败 {config_key} ({runtime})")

                # 清理临时目录
                try:
                    shutil.rmtree(rec['temp_dir'])
                except Exception:
                    pass

                slots[idx] = None
                finished += 1
                print(f"📊 进度: {finished}/{total}")

        time.sleep(0.3)
    
    total_time = time.time() - total_start
    print(f"\n🏁 并行测试完成! 总用时: {total_time:.1f}s")
    # 尝试最终保存一次进度
    try:
        progress.save_progress()
    except Exception as _e:
        print(f"⚠️ 保存进度失败: {_e}")
    
    # 统计结果
    success_count = sum(1 for result in results_dict.values() if result['status'] == 'SUCCESS')
    failed_count = len(results_dict) - success_count
    
    print(f"📊 最终统计: ✅ {success_count} 成功, ❌ {failed_count} 失败")
    
    if failed_count > 0:
        print("\n❌ 失败的任务:")
        for config_key, result in results_dict.items():
            if result['status'] != 'SUCCESS':
                print(f"  {config_key}: {result['status']}")
    
    return failed_count == 0

def main():
    print("=" * 80)
    print("🚀 开始多GPU并行测试所有Benchmark配置文件 (每个配置训练2个epoch)")
    print("支持断点续传，多GPU并行加速")
    print("=" * 80)
    
    configs = find_all_benchmark_configs()
    print(f"\n找到 {len(configs)} 个配置文件需要测试\n")
    
    success = run_all_tests_parallel()
    
    if success:
        print("\n✨ 测试完成！所有配置文件都能正常运行")
        print("现在可以创建并行训练脚本了！")
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
