"""
量化结果可视化脚本

使用方法：
    1. 在下方 RESULTS 列表中编辑数据
    2. 运行: python plot_quantize_results.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
#                       数据编辑区域
# ============================================================
# 直接在这里修改数据，格式：
# ("模型名称", mAP@50, mAP@50-95, 模型大小MB, 推理时间ms, FPS)

RESULTS = [
    ("FP32", 0.929, 0.618, 8.10, 12.96, 77.2),
    ("FP16", 0.928, 0.615, 7.36, 8.5, 117.6),
    ("INT8-Dynamic", 0.912, 0.595, 2.15, 5.2, 192.3),
    ("Mixed", 0.915, 0.600, 3.20, 6.8, 147.1),
]

# 输出目录
OUTPUT_DIR = Path("./plots")

# 图表标题后缀（可选）
TITLE_SUFFIX = ""
# ============================================================


def create_result(name, map50, map50_95=0, size_mb=0, inference_ms=0, fps=0):
    """创建单个模型结果数据"""
    return {
        'name': name,
        'map50': float(map50),
        'map50_95': float(map50_95),
        'size_mb': float(size_mb),
        'inference_ms': float(inference_ms),
        'fps': float(fps),
        'status': 'success',
    }


# ==================== 绘图函数 ====================

def plot_comparison(results, output_dir, title_suffix=""):
    """绘制综合对比图（6个子图）"""
    valid = [r for r in results if r['status'] == 'success']
    if len(valid) < 2:
        print("有效结果不足2个，跳过绘图")
        return
    
    names = [r['name'] for r in valid]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'模型量化性能对比{title_suffix}', fontsize=16, fontweight='bold', y=0.98)
    
    # 1. mAP@50 对比
    ax1 = fig.add_subplot(2, 3, 1)
    map50 = [r['map50'] for r in valid]
    bars = ax1.bar(names, map50, color=colors[:len(names)], edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('mAP@50', fontsize=12, fontweight='bold')
    ax1.set_title('检测精度对比', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(map50) * 1.15)
    for bar, val in zip(bars, map50):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 模型大小对比
    ax2 = fig.add_subplot(2, 3, 2)
    sizes = [r['size_mb'] for r in valid]
    bars = ax2.bar(names, sizes, color=colors[:len(names)], edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('模型大小 (MB)', fontsize=12, fontweight='bold')
    ax2.set_title('存储空间对比', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. 推理速度对比
    ax3 = fig.add_subplot(2, 3, 3)
    fps = [r['fps'] for r in valid]
    bars = ax3.bar(names, fps, color=colors[:len(names)], edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('FPS', fontsize=12, fontweight='bold')
    ax3.set_title('推理速度对比', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, fps):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.tick_params(axis='x', rotation=15)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. 压缩比
    ax4 = fig.add_subplot(2, 3, 4)
    base_size = valid[0]['size_mb']
    compression = [base_size / r['size_mb'] if r['size_mb'] > 0 else 0 for r in valid]
    bars = ax4.bar(names, compression, color=colors[:len(names)], edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('压缩比', fontsize=12, fontweight='bold')
    ax4.set_title(f'压缩比 (相对于{valid[0]["name"]})', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, compression):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax4.tick_params(axis='x', rotation=15)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. 精度保留率
    ax5 = fig.add_subplot(2, 3, 5)
    base_map = valid[0]['map50']
    retention = [r['map50'] / base_map * 100 for r in valid]
    bars = ax5.bar(names, retention, color=colors[:len(names)], edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('精度保留率 (%)', fontsize=12, fontweight='bold')
    ax5.set_title('精度保留率', fontsize=14, fontweight='bold')
    ax5.axhline(y=95, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='95%阈值')
    ax5.axhline(y=98, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='98%阈值')
    for bar, val in zip(bars, retention):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax5.tick_params(axis='x', rotation=15)
    ax5.legend(loc='lower right')
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. 精度-大小权衡
    ax6 = fig.add_subplot(2, 3, 6)
    for i, r in enumerate(valid):
        ax6.scatter(r['size_mb'], r['map50'], c=colors[i % len(colors)], s=200, 
                   edgecolors='black', linewidths=2, label=r['name'], zorder=3)
        ax6.annotate(r['name'], (r['size_mb'], r['map50']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    ax6.set_xlabel('模型大小 (MB)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('mAP@50', fontsize=12, fontweight='bold')
    ax6.set_title('精度-大小权衡图\n(左上角最优)', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = output_dir / "quantize_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"对比图保存至: {output_path}")


def plot_radar(results, output_dir, title_suffix=""):
    """绘制雷达图"""
    valid = [r for r in results if r['status'] == 'success']
    if len(valid) < 2:
        return
    
    base = valid[0]
    metrics = ['精度', '压缩', '速度', '效率']
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
    
    for i, r in enumerate(valid):
        precision = r['map50'] / base['map50'] * 100 if base['map50'] > 0 else 0
        compression = min(base['size_mb'] / r['size_mb'] * 50, 100) if r['size_mb'] > 0 else 0
        speed = min(r['fps'] / base['fps'] * 50, 100) if base['fps'] > 0 else 0
        efficiency = (precision + compression + speed) / 3
        
        values = [precision, compression, speed, efficiency]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i % len(colors)], label=r['name'])
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
    ax.set_title(f'量化综合性能雷达图{title_suffix}', fontsize=14, fontweight='bold', pad=20)
    
    output_path = output_dir / "quantize_radar.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"雷达图保存至: {output_path}")


def plot_speed_accuracy(results, output_dir):
    """绘制精度-速度权衡图（额外图表）"""
    valid = [r for r in results if r['status'] == 'success']
    if len(valid) < 2:
        return
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, r in enumerate(valid):
        ax.scatter(r['inference_ms'], r['map50'], c=colors[i % len(colors)], s=300, 
                   edgecolors='black', linewidths=2, label=r['name'], zorder=3)
        ax.annotate(r['name'], (r['inference_ms'], r['map50']), 
                    xytext=(8, 8), textcoords='offset points', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('推理时间 (ms)', fontsize=14, fontweight='bold')
    ax.set_ylabel('mAP@50', fontsize=14, fontweight='bold')
    ax.set_title('精度-速度权衡图\n(左上角最优: 高精度+低延迟)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower left', fontsize=10)
    
    plt.tight_layout()
    
    output_path = output_dir / "speed_accuracy_tradeoff.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"精度-速度图保存至: {output_path}")


def print_summary(results):
    """打印数据表格"""
    print("\n" + "=" * 80)
    print(" 模型量化评估结果")
    print("=" * 80)
    
    print(f"\n{'方法':<15} {'mAP@50':<10} {'mAP@50-95':<10} {'大小(MB)':<10} {'推理(ms)':<10} {'FPS':<10}")
    print("-" * 65)
    
    for r in results:
        if r['status'] != 'success':
            print(f"{r['name']:<15} FAILED")
            continue
        print(f"{r['name']:<15} {r['map50']:<10.4f} {r['map50_95']:<10.4f} "
              f"{r['size_mb']:<10.2f} {r['inference_ms']:<10.2f} {r['fps']:<10.1f}")
    
    print("-" * 65)
    
    valid = [r for r in results if r['status'] == 'success']
    if len(valid) >= 2:
        base = valid[0]
        print(f"\n相对于 {base['name']}:")
        for r in valid[1:]:
            comp = base['size_mb'] / r['size_mb'] if r['size_mb'] > 0 else 0
            speed = r['fps'] / base['fps'] if base['fps'] > 0 else 0
            prec = r['map50'] / base['map50'] * 100 if base['map50'] > 0 else 0
            print(f"  {r['name']:<15}: 压缩 {comp:.2f}x, 加速 {speed:.2f}x, 精度保留 {prec:.1f}%")


# ==================== 主流程 ====================

def main():
    if not RESULTS:
        print("请在脚本顶部的 RESULTS 列表中添加数据！")
        return
    
    # 转换数据格式
    results = [create_result(*r) for r in RESULTS]
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 打印数据表格
    print_summary(results)
    
    # 保存数据到JSON
    json_path = OUTPUT_DIR / "input_data.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n数据已保存至: {json_path}")
    
    # 绘图
    title_suffix = f" ({TITLE_SUFFIX})" if TITLE_SUFFIX else ""
    
    plot_comparison(results, OUTPUT_DIR, title_suffix)
    plot_radar(results, OUTPUT_DIR, title_suffix)
    plot_speed_accuracy(results, OUTPUT_DIR)
    
    print(f"\n所有图表已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
