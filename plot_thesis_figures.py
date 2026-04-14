"""
论文科研图片生成脚本 — YOLO11-CSPPC-ECA-SPPELAN 改进模型

生成图片列表：
    fig_01_overall_architecture.*   整体架构图（数据流向+各层组件连接）
    fig_02_csppc_module.*           CSPPC 模块详细结构图
    fig_03_eca_module.*             ECA 注意力机制详细图
    fig_04_sppelan_module.*         SPPELAN 结构详图
    fig_05_enhanced_p3.*            Enhanced P3 小目标增强模块
    fig_06_hierarchy.*              模型层次结构图（Input→Backbone→Neck→Head）

使用方法：
    python plot_thesis_figures.py

输出目录：../Paper/figures/model_structure/
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "Paper" / "figures" / "model_structure"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300


# ==================== 绘图工具函数 ====================

COLORS = {
    'input': '#E8F4FD',
    'input_edge': '#3498DB',
    'conv': '#D5F5E3',
    'conv_edge': '#27AE60',
    'csp': '#FCF3CF',
    'csp_edge': '#F39C12',
    'attention': '#FADBD8',
    'attention_edge': '#E74C3C',
    'pooling': '#E8DAEF',
    'pooling_edge': '#8E44AD',
    'concat': '#D6EAF8',
    'concat_edge': '#2980B9',
    'upsample': '#D5DBDB',
    'upsample_edge': '#7F8C8D',
    'detect': '#FADBD8',
    'detect_edge': '#C0392B',
    'backbone': '#D5F5E3',
    'neck': '#FCF3CF',
    'head': '#FADBD8',
    'arrow': '#2C3E50',
    'text': '#2C3E50',
    'highlight': '#FF6B6B',
    'innovate': '#FFE66D',
}


def draw_rounded_box(ax, x, y, w, h, label, fc, ec, fontsize=9, lw=1.5,
                     sublabel=None, radius=0.15):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle=f"round,pad=0.02,rounding_size={radius}",
                          facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=COLORS['text'], wrap=True)
    if sublabel:
        ax.text(x, y - h / 2 - 0.15, sublabel, ha='center', va='top',
                fontsize=7, color='#7F8C8D', style='italic')


def draw_arrow(ax, x1, y1, x2, y2, style='->', color=COLORS['arrow'],
               lw=1.2, connectionstyle="arc3,rad=0"):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=connectionstyle))


def draw_path_arrow(ax, points, color=COLORS['arrow'], lw=1.2, style='->'):
    """Draw an orthogonal polyline arrow to avoid overlapping labels."""
    for (x1, y1), (x2, y2) in zip(points[:-2], points[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw)
    (x1, y1), (x2, y2) = points[-2], points[-1]
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def add_caption(fig, caption, y_pos=-0.02):
    fig.text(0.5, y_pos, caption, ha='center', va='top', fontsize=10,
             style='italic', color='#555555')


def save_figure(fig, stem):
    """Save each figure as both high-resolution PNG and vector SVG."""
    png_path = OUTPUT_DIR / f"{stem}.png"
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
    print(f"已保存: {png_path}")
    print(f"已保存: {svg_path}")


# ==================== 图1: 整体架构图 ====================

def plot_overall_architecture():
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(-1, 20)
    ax.set_ylim(-1, 13)
    ax.axis('off')
    ax.set_title('改进YOLO模型整体架构图\n(CSPPC-ECA-SPPELAN)',
                 fontsize=16, fontweight='bold', pad=15)

    bh_x = 1.5
    neck_x = 9.5
    head_x = 17.2

    ax.add_patch(FancyBboxPatch((bh_x - 1.3, -0.5), 7, 12.5,
                                 boxstyle="round,pad=0.03", facecolor='#E8F8F5',
                                 edgecolor='#1ABC9C', linewidth=2, alpha=0.3))
    ax.text(bh_x + 2.2, 11.8, 'Backbone 特征提取网络', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#117864')

    ax.add_patch(FancyBboxPatch((neck_x - 3.5, -0.5), 7.5, 12.5,
                                 boxstyle="round,pad=0.03", facecolor='#FEF9E7',
                                 edgecolor='#F39C12', linewidth=2, alpha=0.3))
    ax.text(neck_x + 0.25, 11.8, 'Neck 特征融合网络', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#9A7D0A')

    ax.add_patch(FancyBboxPatch((head_x - 1.5, 4.5), 3.5, 3,
                                 boxstyle="round,pad=0.03", facecolor='#FDEDEC',
                                 edgecolor='#E74C3C', linewidth=2, alpha=0.3))
    ax.text(head_x + 0.25, 7.3, 'Detect\n检测头', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#922B21')

    # ---- Input ----
    draw_rounded_box(ax, bh_x, 10.5, 2.2, 0.8, 'Input Image\n640×640×3',
                     COLORS['input'], COLORS['input_edge'], fontsize=9)

    # ---- Backbone layers ----
    by = [9.3, 8.3, 7.3, 6.3, 5.3, 4.3, 3.3]
    bnames = ['Stem\nConv 3×3 s=2', 'Conv 3×3 s=2', 'C3k2 ×2',
              'Conv 3×3 s=2', 'C3k2 ×2', 'Conv 3×3 s=2', 'C3k2 ×2']
    bsubs = ['P1/2', 'P2/4', '', 'P3/8', '', 'P4/16', '']

    for i, (y, name, sub) in enumerate(zip(by, bnames, bsubs)):
        fc = COLORS['conv'] if i % 2 == 0 else COLORS['csp']
        ec = COLORS['conv_edge'] if i % 2 == 0 else COLORS['csp_edge']
        draw_rounded_box(ax, bh_x, y, 2.2, 0.75, name, fc, ec, fontsize=8,
                         sublabel=sub if sub else None)

    for i in range(len(by) - 1):
        draw_arrow(ax, bh_x, by[i] - 0.38, bh_x, by[i + 1] + 0.38)

    # SPPELAN & C2PSA
    draw_rounded_box(ax, bh_x + 2.5, 3.3, 2.2, 0.8, 'SPPELAN',
                     COLORS['pooling'], COLORS['pooling_edge'], fontsize=9,
                     sublabel='全局上下文')
    draw_arrow(ax, bh_x + 1.1, 3.3, bh_x + 2.5 - 1.1, 3.3)

    draw_rounded_box(ax, bh_x + 2.5, 2.3, 2.2, 0.8, 'C2PSA',
                     COLORS['attention'], COLORS['attention_edge'], fontsize=9,
                     sublabel='空间注意力')
    draw_arrow(ax, bh_x + 2.5, 3.3 - 0.4, bh_x + 2.5, 2.3 + 0.4)

    # Backbone outputs
    p3_out_y, p4_out_y, p5_out_y = 6.3, 4.3, 2.3
    ax.plot([bh_x + 1.1, bh_x + 3.5], [p3_out_y, p3_out_y], '--',
            color=COLORS['highlight'], lw=1.5, alpha=0.7)
    ax.plot([bh_x + 1.1, bh_x + 3.5], [p4_out_y, p4_out_y], '--',
            color=COLORS['highlight'], lw=1.5, alpha=0.7)
    ax.plot([bh_x + 2.5 + 1.1, bh_x + 4.7], [p5_out_y, p5_out_y], '--',
            color=COLORS['highlight'], lw=1.5, alpha=0.7)

    ax.text(bh_x + 3.5, p3_out_y + 0.15, 'P3/8', fontsize=8, color=COLORS['highlight'],
            fontweight='bold')
    ax.text(bh_x + 3.5, p4_out_y + 0.15, 'P4/16', fontsize=8, color=COLORS['highlight'],
            fontweight='bold')
    ax.text(bh_x + 4.7, p5_out_y + 0.15, 'P5/32', fontsize=8, color=COLORS['highlight'],
            fontweight='bold')

    # ---- Neck FPN (top-down) ----
    ny_fpn = [9.3, 7.5, 5.7]
    nnames_fpn = ['Upsample\n×2', 'Concat\n+CSPPC', 'Upsample\n×2']
    nsubs_fpn = ['', 'P4 输出', '']

    for i, (y, name, sub) in enumerate(zip(ny_fpn, nnames_fpn, nsubs_fpn)):
        if 'CSPPC' in name:
            draw_rounded_box(ax, neck_x, y, 2.4, 0.85, name,
                             COLORS['innovate'], COLORS['csp_edge'], fontsize=9,
                             sublabel=sub, lw=2.5)
        elif 'Concat' in name:
            draw_rounded_box(ax, neck_x, y, 2.4, 0.85, name,
                             COLORS['concat'], COLORS['concat_edge'], fontsize=9,
                             sublabel=sub)
        else:
            draw_rounded_box(ax, neck_x, y, 2.4, 0.75, name,
                             COLORS['upsample'], COLORS['upsample_edge'], fontsize=8,
                             sublabel=sub)

    # FPN arrows
    draw_arrow(ax, neck_x, ny_fpn[0] - 0.38, neck_x, ny_fpn[1] + 0.43)
    draw_arrow(ax, neck_x, ny_fpn[1] - 0.43, neck_x, ny_fpn[2] + 0.38)

    # P5 -> FPN top
    draw_path_arrow(ax, [(bh_x + 4.7, p5_out_y), (7.2, p5_out_y), (7.2, ny_fpn[0] + 0.38),
                         (neck_x - 1.2, ny_fpn[0] + 0.38)])

    # P4 concat
    draw_path_arrow(ax, [(bh_x + 3.5, p4_out_y), (7.55, p4_out_y), (7.55, ny_fpn[1]),
                         (neck_x - 1.2, ny_fpn[1])])

    # P3 concat area
    draw_path_arrow(ax, [(bh_x + 3.5, p3_out_y), (7.9, p3_out_y), (7.9, ny_fpn[2]),
                         (neck_x - 1.2, ny_fpn[2])])

    # ---- Enhanced P3 ----
    ep3_y = 4.2
    draw_rounded_box(ax, neck_x, ep3_y, 2.4, 0.8, 'ECA',
                     COLORS['attention'], COLORS['attention_edge'], fontsize=9,
                     sublabel='通道注意力')
    draw_arrow(ax, neck_x, ny_fpn[2] - 0.38, neck_x, ep3_y + 0.4)

    draw_rounded_box(ax, neck_x + 2.8, ep3_y, 2.4, 0.8, 'DWConv\n+dilated+ECA',
                     COLORS['innovate'], COLORS['highlight'], fontsize=8,
                     sublabel='Enhanced P3', lw=2.5)
    draw_arrow(ax, neck_x + 1.2, ep3_y, neck_x + 2.8 - 1.2, ep3_y)

    # ---- Neck PAN (bottom-up) ----
    ny_pan = [2.8, 1.3]
    nnames_pan = ['Conv s=2\n+Concat+CSPPC', 'Conv s=2\n+Concat+CSPPC']
    nsubs_pan = ['P4 PAN', 'P5 PAN']

    for i, (y, name, sub) in enumerate(zip(ny_pan, nnames_pan, nsubs_pan)):
        if 'CSPPC' in name:
            draw_rounded_box(ax, neck_x, y, 2.6, 0.85, name,
                             COLORS['innovate'], COLORS['csp_edge'], fontsize=8,
                             sublabel=sub, lw=2.5)
        else:
            draw_rounded_box(ax, neck_x, y, 2.6, 0.8, name,
                             COLORS['conv'], COLORS['conv_edge'], fontsize=8,
                             sublabel=sub)

    draw_arrow(ax, neck_x, ep3_y - 0.4, neck_x, ny_pan[0] + 0.43)
    draw_arrow(ax, neck_x, ny_pan[0] - 0.43, neck_x, ny_pan[1] + 0.4)

    # PAN connections
    draw_path_arrow(ax, [(neck_x + 1.3, ny_fpn[1]), (11.6, ny_fpn[1]), (11.6, ny_pan[0]),
                         (neck_x - 1.3, ny_pan[0])])
    draw_path_arrow(ax, [(bh_x + 4.7, p5_out_y), (7.4, p5_out_y), (7.4, ny_pan[1]),
                         (neck_x - 1.3, ny_pan[1])])

    # Final P3 context injection
    fp3_y = 0.0
    draw_rounded_box(ax, neck_x, fp3_y, 2.4, 0.75, 'Upsample\n+Concat+C3k2',
                     COLORS['conv'], COLORS['conv_edge'], fontsize=8,
                     sublabel='Final P3')
    draw_path_arrow(ax, [(neck_x, ny_pan[0] + 0.43), (8.0, ny_pan[0] + 0.43), (8.0, fp3_y),
                         (neck_x - 1.2, fp3_y)])
    draw_path_arrow(ax, [(neck_x + 2.8, ep3_y), (13.0, ep3_y), (13.0, fp3_y),
                         (neck_x + 1.2, fp3_y)])

    # ---- Detect head ----
    detect_y = 5.8
    draw_rounded_box(ax, head_x, detect_y, 2.6, 1.2, 'Detect\nHead',
                     COLORS['detect'], COLORS['detect_edge'], fontsize=10, lw=2)

    draw_path_arrow(ax, [(neck_x + 1.2, fp3_y), (14.5, fp3_y), (14.5, detect_y - 0.3),
                         (head_x - 1.3, detect_y - 0.3)])
    draw_path_arrow(ax, [(neck_x + 1.3, ny_pan[0]), (14.0, ny_pan[0]), (14.0, detect_y),
                         (head_x - 1.3, detect_y)])
    draw_path_arrow(ax, [(neck_x + 1.3, ny_pan[1]), (13.5, ny_pan[1]), (13.5, detect_y + 0.3),
                         (head_x - 1.3, detect_y + 0.3)])

    ax.text(head_x, detect_y - 0.85, '(P3, P4, P5)', ha='center', fontsize=8,
            color='#7F8C8D')

    # ---- Legend ----
    legend_items = [
        ('Conv/C3k2', COLORS['conv'], COLORS['conv_edge']),
        ('CSPPC (创新)', COLORS['innovate'], COLORS['csp_edge']),
        ('ECA 注意力', COLORS['attention'], COLORS['attention_edge']),
        ('SPPELAN', COLORS['pooling'], COLORS['pooling_edge']),
        ('Concat/Upsample', COLORS['concat'], COLORS['concat_edge']),
        ('Enhanced P3', COLORS['innovate'], COLORS['highlight']),
    ]
    for i, (name, fc, ec) in enumerate(legend_items):
        y_leg = 0.3 - i * 0.45
        box = FancyBboxPatch((14.5, y_leg - 0.15), 0.5, 0.3,
                              boxstyle="round,pad=0.02",
                              facecolor=fc, edgecolor=ec, linewidth=1)
        ax.add_patch(box)
        ax.text(15.15, y_leg, name, ha='left', va='center', fontsize=8)

    add_caption(fig, '图1 改进YOLO模型整体架构图。Backbone采用SPPELAN增强全局上下文建模能力，'
                'Neck层引入CSPPC替代C3实现更高效的特征融合，并在P3层设计Enhanced P3小目标增强模块，'
                '配合ECA通道注意力机制提升特征表达能力。')

    save_figure(fig, "fig_01_overall_architecture")
    plt.close()
    print("  [1/6] fig_01_overall_architecture exported")


# ==================== 图2: CSPPC 模块 ====================

def plot_csppc_module():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1, 14)
    ax.set_ylim(-1, 9)
    ax.axis('off')
    ax.set_title('CSPPC (Cross Stage Partial C) 模块结构',
                 fontsize=15, fontweight='bold', pad=15)

    cx = 6.5
    cy = 4.5

    main_w, main_h = 10, 7
    ax.add_patch(FancyBboxPatch((cx - main_w / 2, cy - main_h / 2), main_w, main_h,
                                 boxstyle="round,pad=0.03",
                                 facecolor='#FEF9E7', edgecolor='#F39C12',
                                 linewidth=2.5, alpha=0.25))
    ax.text(cx, cy + main_h / 2 + 0.35, 'CSPPC Module', ha='center',
            fontsize=12, fontweight='bold', color='#9A7D0A')

    # Input split
    draw_rounded_box(ax, cx - 4, cy + 2.5, 2, 0.7, 'Split\n(部分连接)',
                     COLORS['concat'], COLORS['concat_edge'], fontsize=9)

    # Main branch (top): CBS -> CBS -> CBS -> Concat
    mb_y = cy + 1.0
    main_blocks = [('CBS\nk=1', -2.5), ('CBS\nk=3', 0), ('CBS\nk=1', 2.5)]
    for name, dx in main_blocks:
        draw_rounded_box(ax, cx + dx, mb_y, 1.6, 0.7, name,
                         COLORS['conv'], COLORS['conv_edge'], fontsize=8)

    draw_arrow(ax, cx - 4.25, cy + 2.15, cx - 3.3, mb_y + 0.35)
    draw_arrow(ax, cx - 1.7, mb_y, cx - 0.9, mb_y)
    draw_arrow(ax, cx + 0.9, mb_y, cx + 1.7, mb_y)

    # Concat in main branch
    draw_rounded_box(ax, cx + 3.5, mb_y, 1.5, 0.7, 'Concat',
                     COLORS['concat'], COLORS['concat_edge'], fontsize=9)
    draw_arrow(ax, cx + 2.5 + 0.8, mb_y, cx + 3.5 - 0.75, mb_y)

    # Skip branch
    skip_y = cy - 1.0
    draw_rounded_box(ax, cx, skip_y, 1.8, 0.7, 'Skip\nConnection',
                     COLORS['upsample'], COLORS['upsample_edge'], fontsize=9)
    draw_path_arrow(ax, [(cx - 3.7, cy + 2.15), (cx - 3.7, skip_y + 0.75),
                         (cx - 0.9, skip_y + 0.75), (cx, skip_y + 0.35)])

    # Final Concat
    draw_rounded_box(ax, cx + 3.5, cy - 0.5, 1.5, 0.7, 'Concat',
                     COLORS['concat'], COLORS['concat_edge'], fontsize=9, lw=2)
    draw_arrow(ax, cx + 3.5, mb_y - 0.35, cx + 3.5, cy - 0.5 + 0.35)
    draw_arrow(ax, cx + 0.9, skip_y, cx + 3.5 - 0.75, cy - 0.5,
               connectionstyle="arc3,rad=-0.2")

    # Output CBS
    draw_rounded_box(ax, cx + 3.5, cy - 2.0, 1.8, 0.7, 'CBS\nk=1 (输出)',
                     COLORS['conv'], COLORS['conv_edge'], fontsize=9, lw=2)
    draw_arrow(ax, cx + 3.5, cy - 0.5 - 0.35, cx + 3.5, cy - 2.0 + 0.35)

    # Output label
    ax.text(cx + 4.8, cy - 2.0, 'Output', ha='left', va='center', fontsize=10,
            fontweight='bold', color=COLORS['text'])

    # Input label
    ax.text(cx - 5.2, cy + 2.5, 'Input\n$X_{in}$', ha='right', va='center',
            fontsize=10, fontweight='bold', color=COLORS['text'])

    # Annotations
    ax.annotate('梯度分流路径', xy=(cx - 2.5, mb_y), xytext=(cx - 5.1, mb_y + 1.55),
                fontsize=8, color='#E74C3C',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=0.8))
    ax.annotate('跨阶段残差连接', xy=(cx, skip_y), xytext=(cx + 1.9, skip_y - 1.15),
                fontsize=8, color='#2980B9',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='#2980B9', lw=0.8))

    # Parameter table
    table_data = [
        ['组件', '配置', '作用'],
        ['CBS (k=1)', 'Conv1×1-BN-SiLU', '通道调整'],
        ['CBS (k=3)', 'Conv3×3-BN-SiLU', '特征提取'],
        ['Split', '通道切分', '部分连接'],
        ['Concat', '通道拼接', '多尺度融合'],
    ]
    for ri, row in enumerate(table_data):
        for ci, val in enumerate(row):
            fx = 9.5 + ci * 1.8
            fy = 6.5 - ri * 0.7
            bg = '#ECF0F1' if ri == 0 else 'white'
            ax.add_patch(Rectangle((fx - 0.85, fy - 0.28), 1.7, 0.56,
                                    facecolor=bg, edgecolor='#BDC3C7', linewidth=0.5))
            fs = 8 if ri == 0 else 7
            fw = 'bold' if ri == 0 else 'normal'
            ax.text(fx, fy, val, ha='center', va='center', fontsize=fs,
                    fontweight=fw, color=COLORS['text'])

    ax.text(10.4, 7.1, '模块参数表', ha='center', fontsize=9, fontweight='bold')

    add_caption(fig, '图2 CSPPC模块结构图。该模块采用Cross Stage Partial设计，通过部分连接将输入特征分为两路处理：'
                '主路经三级CBS卷积提取深层特征后拼接，旁路保留原始信息作为残差。最终融合输出兼顾局部细节与全局语义，'
                '相比标准C3模块具有更强的特征复用能力和更低的计算冗余。')

    save_figure(fig, "fig_02_csppc_module")
    plt.close()
    print("  [2/6] fig_02_csppc_module exported")


# ==================== 图3: ECA 注意力机制 ====================

def plot_eca_module():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ---- Left: ECA Structure ----
    ax = axes[0]
    ax.set_xlim(-1, 9)
    ax.set_ylim(-1, 8)
    ax.axis('off')
    ax.set_title('ECA (Efficient Channel Attention) 结构',
                 fontsize=14, fontweight='bold', pad=10)

    cx, cy = 4, 4

    # Global Avg Pool
    draw_rounded_box(ax, cx - 3, cy + 1.5, 2.2, 0.9,
                     'Global Avg\nPooling', COLORS['pooling'], COLORS['pooling_edge'],
                     fontsize=9, sublabel='GAP: H×W → 1×1')

    # 1D Conv kernel size k
    draw_rounded_box(ax, cx, cy + 1.5, 2.2, 0.9,
                     '1D Conv\n(k自适应)', COLORS['conv'], COLORS['conv_edge'],
                     fontsize=9, sublabel='kernel: γ = f(C)')

    # Sigmoid
    draw_rounded_box(ax, cx + 3, cy + 1.5, 2.0, 0.9,
                     'Sigmoid', COLORS['attention'], COLORS['attention_edge'],
                     fontsize=10, sublabel='σ(·) ∈ [0,1]')

    # Input feature map
    draw_rounded_box(ax, cx - 3, cy - 1.5, 2.2, 0.9,
                     'Input Feature\nH × W × C', COLORS['input'], COLORS['input_edge'],
                     fontsize=9)

    # Multiply (channel attention)
    draw_rounded_box(ax, cx + 3, cy - 1.5, 2.0, 0.9,
                     'Channel\nMultiply', COLORS['highlight'], COLORS['detect_edge'],
                     fontsize=9, sublabel='逐通道加权', lw=2)

    # Output
    ax.text(cx + 5.5, cy - 1.5, 'Output\nH×W×C', ha='left', va='center',
            fontsize=10, fontweight='bold', color=COLORS['text'])

    # Arrows
    draw_arrow(ax, cx - 1.9, cy + 1.5, cx - 1.1, cy + 1.5)
    draw_arrow(ax, cx + 1.1, cy + 1.5, cx + 2.0, cy + 1.5)
    draw_arrow(ax, cx - 3, cy + 1.05, cx - 3, cy - 1.05)
    draw_arrow(ax, cx + 3, cy + 1.05, cx + 3, cy - 1.05)
    draw_arrow(ax, cx - 1.9, cy - 1.5, cx + 2.0, cy - 1.5)
    ax.plot([cx + 4.0, cx + 4.0], [cy + 1.05, cy - 1.05], '--',
            color=COLORS['attention_edge'], lw=1.5, alpha=0.6)
    ax.text(cx + 4.3, cy, '通道权重\n$\\vec{a}$', fontsize=8, color=COLORS['attention_edge'],
            va='center')

    # Formula annotation
    formula_box = FancyBboxPatch((0.3, -0.7), 7.4, 1.2,
                                  boxstyle="round,pad=0.02",
                                  facecolor='#EBF5FB', edgecolor='#2980B9',
                                  linewidth=1, alpha=0.8)
    ax.add_patch(formula_box)
    ax.text(4, -0.1, r'$k = \gamma\left(\frac{C}{b}\right) = \left|\frac{\log_2(C/b)+\phi}{2}\right|_{odd}$',
            ha='center', va='center', fontsize=10, family='serif')
    ax.text(4, -0.65, r'$a = \sigma(Conv1D(GAP(x); k)) \quad,\quad y = a \odot x$',
            ha='center', va='center', fontsize=9, family='serif', color='#2C3E50')

    # ---- Right: Comparison / Visualization ----
    ax2 = axes[1]
    ax2.set_xlim(-1, 9)
    ax2.set_ylim(-1, 8)
    ax2.axis('off')
    ax2.set_title('ECA vs SE 注意力对比',
                  fontsize=14, fontweight='bold', pad=10)

    # SE block
    se_y = 5.5
    ax2.add_patch(FancyBboxPatch((0.3, se_y - 0.8), 3.4, 1.8,
                                  boxstyle="round,pad=0.02",
                                  facecolor='#FDEDEC', edgecolor='#E74C3C',
                                  linewidth=1.5, alpha=0.4))
    ax2.text(2.0, se_y + 0.65, 'SE Attention', ha='center', fontsize=10,
             fontweight='bold', color='#922B21')

    se_blocks = [('GAP', 0.8), ('FC1\n(降维)', 2.0), ('ReLU', 3.2),
                 ('FC2\n(升维)', 4.4), ('Sigmoid', 5.6)]
    for name, sx in se_blocks:
        draw_rounded_box(ax2, sx, se_y, 1.0, 0.55, name,
                         '#FADBD8' if 'FC' in name or 'ReLU' in name else COLORS['pooling'],
                         COLORS['attention_edge'], fontsize=7)
    for i in range(len(se_blocks) - 1):
        draw_arrow(ax2, se_blocks[i][1] + 0.5, se_y,
                   se_blocks[i + 1][1] - 0.5, se_y, lw=0.8)

    ax2.text(2.0, se_y - 0.7, '参数量: 2C^2/r  |  降维瓶颈', ha='center',
             fontsize=8, color='#C0392B', style='italic')

    # ECA block
    eca_y = 2.5
    ax2.add_patch(FancyBboxPatch((0.3, eca_y - 0.8), 3.4, 1.8,
                                  boxstyle="round,pad=0.02",
                                  facecolor='#E8F8F5', edgecolor='#1ABC9C',
                                  linewidth=1.5, alpha=0.4))
    ax2.text(2.0, eca_y + 0.65, 'ECA (Ours)', ha='center', fontsize=10,
             fontweight='bold', color='#117864')

    eca_blocks = [('GAP', 0.8), ('1D Conv\n(k自适应)', 2.0), ('Sigmoid', 3.2)]
    for name, ex in eca_blocks:
        draw_rounded_box(ax2, ex, eca_y, 1.2, 0.55, name,
                         COLORS['conv'] if 'Conv' in name else COLORS['pooling'],
                         COLORS['conv_edge'] if 'Conv' in name else COLORS['pooling_edge'],
                         fontsize=7)
    for i in range(len(eca_blocks) - 1):
        draw_arrow(ax2, eca_blocks[i][1] + 0.6, eca_y,
                   eca_blocks[i + 1][1] - 0.6, eca_y, lw=0.8)

    ax2.text(2.0, eca_y - 0.7, r'参数量: C×k ≈ C  |  无降维损失', ha='center',
             fontsize=8, color='#117A64', style='italic')

    # Advantages list
    adv_text = '''优势：
  - 用1D卷积替代全连接层
  - 自适应卷积核大小 k
  - 参数量从 O(C^2) 降至 O(C)
  - 保持通道间局部交互
  - 计算效率提升 ~10×'''
    ax2.text(5.5, 4.5, adv_text, fontsize=9, va='top',
             linespacing=1.6,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF9E7',
                       edgecolor='#F39C12', alpha=0.8))

    add_caption(fig, '图3 ECA高效通道注意力机制。(左) ECA模块结构：通过全局平均池化压缩空间维度后，'
                '使用一维卷积捕获通道间局部依赖关系，经Sigmoid激活得到通道权重并逐通道加权原特征。'
                '卷积核大小k根据通道数C自适应计算。(右) 与SE注意力的对比：ECA以轻量级1D卷积替代SE中'
                '的两次全连接降维-升维操作，在保持性能的同时大幅降低参数量和计算开销。')

    save_figure(fig, "fig_03_eca_module")
    plt.close()
    print("  [3/6] fig_03_eca_module exported")


# ==================== 图4: SPPELAN 模块 ====================

def plot_sppelan_module():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(-1.5, 15)
    ax.set_ylim(-1.5, 10)
    ax.axis('off')
    ax.set_title('SPPELAN (Spatial Pyramid Pooling E-LAN) 模块结构',
                 fontsize=15, fontweight='bold', pad=15)

    cx, cy = 6.5, 5

    ax.add_patch(FancyBboxPatch((cx - 6, cy - 4), 12, 8,
                                 boxstyle="round,pad=0.03",
                                 facecolor='#F5EEF8', edgecolor='#8E44AD',
                                 linewidth=2.5, alpha=0.2))
    ax.text(cx, cy + 4.1, 'SPPELAN Module', ha='center',
            fontsize=12, fontweight='bold', color='#6C3483')

    # Input
    draw_rounded_box(ax, cx - 5, cy + 2.5, 2.0, 0.8,
                     'Input\nC×H×W', COLORS['input'], COLORS['input_edge'], fontsize=9)

    # MaxPool branches
    branches = [
        ('Identity', 0, '1×1'),
        ('MaxPool\n5×5', -2.5, '5×5'),
        ('MaxPool\n9×9', 0, '9×9'),
        ('MaxPool\n13×13', 2.5, '13×13'),
    ]
    pool_y = cy + 0.8
    for name, dx, ks in branches:
        bx = cx + dx
        if 'Identity' in name:
            draw_rounded_box(ax, bx, pool_y, 1.8, 0.7, name,
                             COLORS['upsample'], COLORS['upsample_edge'], fontsize=8)
        else:
            draw_rounded_box(ax, bx, pool_y, 1.8, 0.7, name,
                             COLORS['pooling'], COLORS['pooling_edge'], fontsize=8,
                             sublabel=ks)

    # Arrows from input to pools
    for _, dx, _ in branches:
        bx = cx + dx
        if dx != 0:
            rad = 0.15 if dx > 0 else -0.15
            draw_arrow(ax, cx - 4, cy + 2.5, bx, pool_y + 0.35,
                       connectionstyle=f"arc3,rad={rad}")
        else:
            draw_arrow(ax, cx - 4, cy + 2.5, bx, pool_y + 0.35)

    # Concat all branches
    concat_y = cy - 0.5
    draw_rounded_box(ax, cx, concat_y, 2.0, 0.8,
                     'Concat\n(4路拼接)', COLORS['concat'], COLORS['concat_edge'],
                     fontsize=9, lw=1.8)
    for _, dx, _ in branches:
        bx = cx + dx
        draw_arrow(ax, bx, pool_y - 0.35, cx, concat_y + 0.4,
                   connectionstyle="arc3,rad=0" if dx == 0 else f"arc3,rad={-0.1 if dx>0 else 0.1}")

    # CBS block
    cbs_y = cy - 2.0
    draw_rounded_box(ax, cx, cbs_y, 2.0, 0.8,
                     'CBS\nConv1×1', COLORS['conv'], COLORS['conv_edge'],
                     fontsize=9)
    draw_arrow(ax, cx, concat_y - 0.4, cx, cbs_y + 0.4)

    # E-LAN part (right side)
    elan_x = cx + 4
    elan_top_y = cy + 1.5

    ax.add_patch(FancyBboxPatch((elan_x - 2.2, elan_top_y - 3.5), 4.4, 4.2,
                                 boxstyle="round,pad=0.02",
                                 facecolor='#FEF9E7', edgecolor='#F39C12',
                                 linewidth=1.5, alpha=0.3))
    ax.text(elan_x, elan_top_y + 0.85, 'E-LAN 局部聚合', ha='center',
            fontsize=10, fontweight='bold', color='#9A7D0A')

    # E-LAN internal structure: 2-level aggregation
    elan_nodes = [
        ('Conv\n1×1', elan_x, elan_top_y),
        ('Conv\n1×1', elan_x - 1.5, elan_top_y - 1.3),
        ('Conv\n3×3', elan_x + 1.5, elan_top_y - 1.3),
        ('Concat', elan_x, elan_top_y - 2.6),
    ]

    for name, nx, ny in elan_nodes[:3]:
        fc = COLORS['conv'] if '3×3' in name else COLORS['csp']
        ec = COLORS['conv_edge'] if '3×3' in name else COLORS['csp_edge']
        draw_rounded_box(ax, nx, ny, 1.4, 0.65, name, fc, ec, fontsize=8)

    draw_rounded_box(ax, elan_x, elan_top_y - 2.6, 1.4, 0.65,
                     'Concat', COLORS['concat'], COLORS['concat_edge'], fontsize=8)

    # E-LAN arrows
    draw_arrow(ax, cx + 1.0, cbs_y, elan_x - 2.2, elan_top_y)
    draw_arrow(ax, elan_x, elan_top_y - 0.33, elan_x - 1.5, elan_top_y - 1.3 + 0.33)
    draw_arrow(ax, elan_x, elan_top_y - 0.33, elan_x + 1.5, elan_top_y - 1.3 + 0.33)
    draw_arrow(ax, elan_x - 1.5, elan_top_y - 1.3 - 0.33, elan_x, elan_top_y - 2.6 + 0.33)
    draw_arrow(ax, elan_x + 1.5, elan_top_y - 1.3 - 0.33, elan_x, elan_top_y - 2.6 + 0.33)

    # Output
    out_y = cy - 3.5
    draw_rounded_box(ax, elan_x, out_y, 2.0, 0.8,
                     'Output\nC\'×H×W', COLORS['input'], COLORS['input_edge'],
                     fontsize=9, lw=2)
    draw_arrow(ax, elan_x, elan_top_y - 2.6 - 0.33, elan_x, out_y + 0.4)

    # Compact legend for pooling branches
    scale_text = 'SPP分支:\n1×1: 原始特征\n5×5: 细粒度上下文\n9×9: 中尺度上下文\n13×13: 大感受野信息'
    ax.text(0.45, 6.35, scale_text, fontsize=7.8, va='top', linespacing=1.35,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor='#D7DBDD', alpha=0.95))

    # Info box
    info_text = '''核心思想:
  - SPP: 多尺度空间金字塔池化
    - 并行多尺度MaxPool捕获
      不同感受野的特征
    
  - E-LAN: 高效局部聚合网络
    - 分支式卷积提取局部特征
    - Concat融合多分支信息
    - 参数效率优于传统ELAN'''
    ax.text(12, 6, info_text, fontsize=8.5, va='top', linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#EBF5FB',
                      edgecolor='#2980B9', alpha=0.9))

    add_caption(fig, '图4 SPPELAN模块结构图。该模块由两部分组成：(1) 空间金字塔池化(SPP)部分，'
                '使用1×1、5×5、9×9、13×13四种尺度的最大池化并行处理输入特征，捕获不同感受野下的'
                '多尺度空间信息；(2) E-LAN高效局部聚合部分，通过分叉-卷积-拼接结构进一步提炼特征。'
                '四路池化结果经Concat和CBS整合后送入E-LAN，最终输出富含多尺度上下文信息的特征表示。')

    save_figure(fig, "fig_04_sppelan_module")
    plt.close()
    print("  [4/6] fig_04_sppelan_module exported")


# ==================== 图5: Enhanced P3 模块 ====================

def plot_enhanced_p3():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(-1.5, 16)
    ax.set_ylim(-1.5, 10)
    ax.axis('off')
    ax.set_title('Enhanced P3 小目标特征增强模块',
                 fontsize=15, fontweight='bold', pad=15)

    cx, cy = 7, 4.5

    # Background region
    ax.add_patch(FancyBboxPatch((cx - 6, cy - 3.5), 12, 7,
                                 boxstyle="round,pad=0.03",
                                 facecolor='#FDEDEC', edgecolor='#E74C3C',
                                 linewidth=2.5, alpha=0.15))
    ax.text(cx, cy + 3.6, 'Enhanced P3 Small Object Enhancement Module',
            ha='center', fontsize=12, fontweight='bold', color='#922B21')

    # Input from P3 FPN
    draw_rounded_box(ax, cx - 5, cy + 2, 2.2, 0.8,
                     'FPN-P3\nFeature', COLORS['input'], COLORS['input_edge'],
                     fontsize=9, sublabel='来自FPN上采样')

    # Stage 1: ECA channel attention
    draw_rounded_box(ax, cx - 2, cy + 2, 2.2, 0.8,
                     'ECA\n通道注意力', COLORS['attention'], COLORS['attention_edge'],
                     fontsize=9, sublabel='筛选关键通道', lw=2)
    draw_arrow(ax, cx - 3.9, cy + 2, cx - 3.1, cy + 2)

    # Stage 2: DWConv local detail
    draw_rounded_box(ax, cx + 1, cy + 2, 2.4, 0.8,
                     'Depthwise Conv\n3×3, s=1', COLORS['conv'], COLORS['conv_edge'],
                     fontsize=9, sublabel='局部细节提取', lw=1.5)
    draw_arrow(ax, cx - 0.9, cy + 2, cx - 0.2, cy + 2)

    # Stage 3: Dilated Conv
    draw_rounded_box(ax, cx + 4.2, cy + 2, 2.4, 0.8,
                     'Dilated Conv\n3×3, d=2', COLORS['csp'], COLORS['csp_edge'],
                     fontsize=9, sublabel='扩大感受野', lw=2)
    draw_arrow(ax, cx + 2.2, cy + 2, cx + 3.0, cy + 2)

    # Add + Residual
    ax.text(cx + 6.7, cy + 2, '+', fontsize=18, fontweight='bold',
            color=COLORS['text'], ha='center', va='center')

    # Residual line
    ax.annotate('', xy=(cx + 6.5, cy + 0.3),
                xytext=(cx - 5, cy + 2),
                arrowprops=dict(arrowstyle='-', color='#7F8C8D', lw=1.5,
                                linestyle='--', connectionstyle="arc3,rad=0.3"))
    ax.text(cx + 6.8, cy + 1.2, 'Skip\nConnection', fontsize=7, color='#7F8C8D',
            ha='center', va='center', style='italic')

    # Stage 4: Final ECA refine
    draw_rounded_box(ax, cx + 4.2, cy - 0.3, 2.4, 0.8,
                     'ECA\n通道精炼', COLORS['attention'], COLORS['attention_edge'],
                     fontsize=9, sublabel='二次通道校准', lw=2)
    draw_arrow(ax, cx + 4.2, cy + 2 - 0.4, cx + 4.2, cy - 0.3 + 0.4)

    # Merge residual
    merge_x = cx + 1
    draw_rounded_box(ax, merge_x, cy - 0.3, 2.0, 0.8,
                     'Element-wise\nAdd', COLORS['concat'], COLORS['concat_edge'],
                     fontsize=9, sublabel='残差融合')
    draw_arrow(ax, cx + 3.0, cy - 0.3, merge_x + 1.0, cy - 0.3)

    # Output
    draw_rounded_box(ax, merge_x, cy - 2.0, 2.2, 0.8,
                     'Enhanced\nP3 Output', COLORS['highlight'], COLORS['detect_edge'],
                     fontsize=9, sublabel='小目标增强特征', lw=2.5)
    draw_arrow(ax, merge_x, cy - 0.3 - 0.4, merge_x, cy - 2.0 + 0.4)

    # Why it works explanation box
    why_text = '''设计动机:

  问题: 小目标在浅层(P3)特征图中
       占有像素少、信噪比低

  解决策略:
  ① ECA预筛选: 抑制噪声通道，
     增强目标相关通道响应
     
  ② DWConv: 轻量级逐通道卷积，
     提取局部纹理细节
     
  ③ Dilated(d=2): 零膨胀扩大
     感受野至5×5，不增加参数，
     捕获小目标周围上下文
     
  ④ ECA精炼: 二次通道校准，
     确保增强特征的判别性
     
  ⑤ 残差连接: 保护原始FPN特征，
     防止过度变换导致退化'''

    ax.text(12.5, 5.5, why_text, fontsize=8.5, va='top', linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF9E7',
                      edgecolor='#F39C12', alpha=0.9))

    # Feature map visualization hint
    viz_y = cy - 3.3
    for i, label in enumerate(['弱响应区', '目标区域\n(增强)', '背景抑制']):
        vx = cx - 3 + i * 3
        colors_viz = ['#BDC3C7', '#E74C3C', '#27AE60']
        rect = Rectangle((vx - 1, viz_y - 0.4), 2, 0.8,
                          facecolor=colors_viz[i], edgecolor='#2C3E50',
                          linewidth=1, alpha=0.6)
        ax.add_patch(rect)
        ax.text(vx, viz_y, label, ha='center', va='center', fontsize=7,
                color='white' if i == 1 else COLORS['text'], fontweight='bold')

    ax.text(cx, viz_y - 0.7, '特征响应示意 (处理后)', ha='center', fontsize=8,
            color='#7F8C8D', style='italic')

    add_caption(fig, '图5 Enhanced P3小目标特征增强模块。针对空中障碍物中小目标占比极低的问题，'
                '在FPN输出的P3层特征之后设计了四级增强流水线：(1) ECA通道注意力预筛选关键特征通道；'
                '(2) 深度可分离卷积(DWConv)轻量级提取局部纹理细节；(3) 空洞卷积(dilation=2)在不增加'
                '参数量的前提下扩大感受野，捕获小目标周围上下文信息；(4) ECA二次通道精炼确保特征判别性。'
                '残差连接保护原始FPN特征，防止过度变换导致信息丢失。')

    save_figure(fig, "fig_05_enhanced_p3")
    plt.close()
    print("  [5/6] fig_05_enhanced_p3 exported")


# ==================== 图6: 层次结构图 ====================

def plot_hierarchy():
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(-1, 18)
    ax.set_ylim(-1, 12)
    ax.axis('off')
    ax.set_title('改进YOLO模型层次结构与数据流',
                 fontsize=16, fontweight='bold', pad=15)

    # Column positions
    col_input = 1.5
    col_backbone = 5.5
    col_neck = 10
    col_head = 15.4

    # ---- Layer labels ----
    layer_configs = [
        (col_input, 10.5, '输入层\nInput', 2.2, 1.0, COLORS['input'], COLORS['input_edge']),
        (col_backbone, 10.5, '骨干网络\nBackbone', 2.8, 1.0, '#E8F8F5', '#1ABC9C'),
        (col_neck, 10.5, '颈部网络\nNeck', 2.8, 1.0, '#FEF9E7', '#F39C12'),
        (col_head, 10.5, '检测头\nHead', 2.2, 1.0, '#FDEDEC', '#E74C3C'),
    ]
    for x, y, label, w, h, fc, ec in layer_configs:
        draw_rounded_box(ax, x, y, w, h, label, fc, ec, fontsize=11, lw=2.5)

    # Vertical separator lines
    for x in [3.0, 7.8, 12.2]:
        ax.axvline(x=x, ymin=0.05, ymax=0.88, color='#BDC3C7',
                   linewidth=1, linestyle='--', alpha=0.5)

    # ===== INPUT LAYER =====
    draw_rounded_box(ax, col_input, 9.0, 2.2, 0.9,
                     '图像输入\n640×640×3', COLORS['input'], COLORS['input_edge'], fontsize=9)
    draw_arrow(ax, col_input, 8.55, col_input, 7.95)

    # ===== BACKBONE =====
    bb_layers = [
        ('Stem\nConv 3×3', 8.5, COLORS['conv'], COLORS['conv_edge'], False),
        ('Conv 3×3', 7.5, COLORS['conv'], COLORS['conv_edge'], False),
        ('C3k2 ×2', 6.5, COLORS['csp'], COLORS['csp_edge'], False),
        ('Conv 3×3', 5.5, COLORS['conv'], COLORS['conv_edge'], False),
        ('C3k2 ×2', 4.5, COLORS['csp'], COLORS['csp_edge'], False),
        ('Conv 3×3', 3.5, COLORS['conv'], COLORS['conv_edge'], False),
        ('C3k2 ×2', 2.5, COLORS['csp'], COLORS['csp_edge'], False),
    ]
    for name, y, fc, ec, innov in bb_layers:
        lw = 2.5 if innov else 1.2
        draw_rounded_box(ax, col_backbone, y, 2.6, 0.7, name, fc, ec, fontsize=8, lw=lw)
    for i in range(len(bb_layers) - 1):
        draw_arrow(ax, col_backbone, bb_layers[i][1] - 0.35,
                   col_backbone, bb_layers[i + 1][1] + 0.35)

    # SPPELAN & C2PSA (innovation highlight)
    draw_rounded_box(ax, col_backbone, 1.3, 2.6, 0.7,
                     'SPPELAN', COLORS['pooling'], COLORS['pooling_edge'],
                     fontsize=9, lw=2.5, sublabel='创新①')
    draw_arrow(ax, col_backbone, 2.5 - 0.35, col_backbone, 1.3 + 0.35)

    draw_rounded_box(ax, col_backbone, 0.3, 2.6, 0.7,
                     'C2PSA', COLORS['attention'], COLORS['attention_edge'],
                     fontsize=9, lw=2.5, sublabel='创新②')
    draw_arrow(ax, col_backbone, 1.3 - 0.35, col_backbone, 0.3 + 0.35)

    # Backbone output markers
    p3_y, p4_y, p5_y = 6.5, 4.5, 0.3
    for py, label, color in [(p3_y, 'P3/8', '#3498DB'), (p4_y, 'P4/16', '#E67E22'),
                              (p5_y, 'P5/32', '#E74C3C')]:
        ax.plot([col_backbone + 1.4, col_backbone + 2.0], [py, py],
                '-', color=color, lw=2.5, alpha=0.8)
        ax.text(col_backbone + 2.1, py, label, fontsize=8, color=color,
                fontweight='bold', va='center')

    # ===== NECK (FPN + PAN) =====
    neck_layers = [
        ('Upsample ×2', 8.5, COLORS['upsample'], COLORS['upsample_edge'], False),
        ('Concat +\nCSPPC', 7.3, COLORS['innovate'], COLORS['csp_edge'], True),
        ('Upsample ×2', 6.1, COLORS['upsample'], COLORS['upsample_edge'], False),
        ('Concat +\nCSPPC', 4.9, COLORS['innovate'], COLORS['csp_edge'], True),
        ('ECA', 3.7, COLORS['attention'], COLORS['attention_edge'], True),
        ('DWConv +\nDilated + ECA', 2.5, COLORS['innovate'], COLORS['highlight'], True),
        ('Conv ↓ +\nConcat + CSPPC', 1.3, COLORS['innovate'], COLORS['csp_edge'], True),
        ('Conv ↓ +\nConcat + CSPPC', 0.1, COLORS['innovate'], COLORS['csp_edge'], True),
    ]
    for name, y, fc, ec, innov in neck_layers:
        lw = 2.5 if innov else 1.2
        draw_rounded_box(ax, col_neck, y, 2.8, 0.75, name, fc, ec, fontsize=7.5, lw=lw)
    for i in range(len(neck_layers) - 1):
        draw_arrow(ax, col_neck, neck_layers[i][1] - 0.38,
                   col_neck, neck_layers[i + 1][1] + 0.38)

    # Neck output markers
    neck_outputs = [(0.1, 'P5', '#E74C3C'), (1.3, 'P4', '#E67E22')]
    for ny, label, color in neck_outputs:
        ax.plot([col_neck + 1.5, col_neck + 2.0], [ny, ny],
                '-', color=color, lw=2, alpha=0.8)
        ax.text(col_neck + 2.1, ny, label, fontsize=8, color=color,
                fontweight='bold', va='center')

    # Final P3 marker
    ax.plot([col_neck + 1.5, col_neck + 2.0], [2.5, 2.5],
            '-', color='#3498DB', lw=2, alpha=0.8)
    ax.text(col_neck + 2.1, 2.5, 'P3*', fontsize=8, color='#3498DB',
            fontweight='bold', va='center')

    # ===== HEAD =====
    draw_rounded_box(ax, col_head, 4.0, 2.4, 2.0,
                     'Detect\nHead\n\n(P3*, P4, P5)\n三尺度预测',
                     COLORS['detect'], COLORS['detect_edge'], fontsize=9, lw=2.5)

    # Cross-layer connections (Backbone -> Neck)
    conn_style_solid = dict(arrowstyle='->', color='#2980B9', lw=1.5)
    conn_style_dashed = dict(arrowstyle='->', color='#E74C3C', lw=1.5, linestyle='--')

    # P5 -> FPN top
    draw_path_arrow(ax, [(col_backbone + 2.0, p5_y), (8.6, p5_y), (8.6, 8.88),
                         (col_neck - 1.4, 8.88)], color='#2980B9', lw=1.5)

    # P4 -> Concat
    draw_path_arrow(ax, [(col_backbone + 2.0, p4_y), (8.95, p4_y), (8.95, 7.3),
                         (col_neck - 1.4, 7.3)], color='#2980B9', lw=1.5)

    # P3 -> Concat
    draw_path_arrow(ax, [(col_backbone + 2.0, p3_y), (9.3, p3_y), (9.3, 4.9),
                         (col_neck - 1.4, 4.9)], color='#2980B9', lw=1.5)

    # P4_PAN -> Detect
    draw_path_arrow(ax, [(col_neck + 2.0, 1.3), (13.15, 1.3), (13.15, 4.8),
                         (col_head - 1.2, 4.8)], color='#2980B9', lw=1.5)

    # P5_PAN -> Detect
    draw_path_arrow(ax, [(col_neck + 2.0, 0.1), (12.8, 0.1), (12.8, 5.2),
                         (col_head - 1.2, 5.2)], color='#2980B9', lw=1.5)

    # P3* -> Detect
    draw_path_arrow(ax, [(col_neck + 2.0, 2.5), (13.5, 2.5), (13.5, 4.4),
                         (col_head - 1.2, 4.4)], color='#2980B9', lw=1.5)

    # Input -> Backbone
    draw_arrow(ax, col_input, 9.0 - 0.45, col_backbone - 1.4, 8.5 + 0.35)

    # Innovation legend
    innov_items = [
        ('创新① SPPELAN: 多尺度空间金字塔池化', COLORS['pooling'], COLORS['pooling_edge']),
        ('创新② ECA: 高效通道注意力', COLORS['attention'], COLORS['attention_edge']),
        ('创新③ CSPPC: 跨阶段部分连接融合', COLORS['innovate'], COLORS['csp_edge']),
        ('创新④ Enhanced P3: 小目标增强', COLORS['innovate'], COLORS['highlight']),
    ]
    for i, (text, fc, ec) in enumerate(innov_items):
        iy = 10.0 - i * 0.55
        box = FancyBboxPatch((-0.8, iy - 0.18), 0.5, 0.36,
                              boxstyle="round,pad=0.01",
                              facecolor=fc, edgecolor=ec, linewidth=1)
        ax.add_patch(box)
        ax.text(-0.2, iy, text, ha='left', va='center', fontsize=8)

    add_caption(fig, '图6 改进YOLO模型层次结构图。模型自下而上分为四层：(1) 输入层接收640×640 RGB图像；'
                '(2) Backbone骨干网络采用改进的C3k2作为基础单元，末端引入SPPELAN(创新①)进行多尺度'
                '全局上下文建模和C2PSA(创新②)空间注意力增强，输出P3/P4/P5三级特征；'
                '(3) Neck颈部网络构建FPN+PAN双向特征金字塔，用CSPPC(创新③)替代标准C3实现高效特征融合，'
                '并在P3层部署Enhanced P3(创新④)小目标增强模块配合ECA通道注意力(创新②)；'
                '(4) Detect检测头基于三个尺度的增强特征输出最终的类别和位置预测。')

    save_figure(fig, "fig_06_hierarchy")
    plt.close()
    print("  [6/6] fig_06_hierarchy exported")


# ==================== 主流程 ====================

def main():
    print("=" * 60)
    print(" 论文科研图片生成脚本")
    print(" YOLO11-CSPPC-ECA-SPPELAN 改进模型")
    print("=" * 60)
    print(f"输出目录: {OUTPUT_DIR.absolute()}")
    print(f"分辨率: {DPI} DPI")
    print()

    plot_overall_architecture()
    plot_csppc_module()
    plot_eca_module()
    plot_sppelan_module()
    plot_enhanced_p3()
    plot_hierarchy()

    print("\n" + "=" * 60)
    print(f" 全部图片已保存至: {OUTPUT_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
