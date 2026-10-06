# -*- coding: utf-8 -*-
"""
可视化.py — 论文图件统一出图（图 2–图 6 共五张；图 1 为 Visio 机制图，不在本脚本范围）
========================================================================================
数据来源: results/logs/ 与 results/gis_data/ 的实验产出 JSON/NPY（无硬编码数据字面量）
输出:     results/charts/paper/ 下 F1–F5 的 SVG(矢量) + PNG(200 dpi)
  F5_gap_bands            图 2  三城九算例认证间隙带状图 (ablation4_post.json)
  F4_certificate_quality  图 3  证书质量双联板 (det_curve_route2.log / a6_cert_tightening_route2.json; 路线 2 电量约束重跑)
  F1_2d_fatal_failure     图 4  二维模型致命失败 (p3_2d3d.json)
  F2_physics_panels       图 5  物理特性六联板 (arc_phys_table_*.npy / model_experiments_v2.json)
  F3_energy_structure     图 6  能量四通道结构: 悬停+三几何飞行 (p4_energy_decomposition.json 新 schema)
运行: python 可视化.py
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



# ===== 论文图集 (A1-A7 / P1 / P2, 2026-10-02 定稿并入) =====
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "results" / "logs"
GIS = ROOT / "results" / "gis_data"
OUT = ROOT / "results" / "charts" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

C_SG, C_SZ, C_SH = "#1E88E5", "#FB8C00", "#43A047"
CITY_C = {"sg": C_SG, "sz": C_SZ, "sh": C_SH}
CITY_N = {"sg": "Singapore", "sz": "Shenzhen", "sh": "Shanghai"}
CELLS = ["sg_60", "sz_60", "sh_60", "sg_90", "sz_90", "sh_90", "sg_120", "sz_120", "sh_120"]
plt.rcParams.update({"svg.fonttype": "none", "font.size": 12.5,
                     "axes.titlesize": 10, "axes.labelsize": 9})

# ===== F 系列 (2026-10-03 视觉重制定稿): Okabe-Ito 判决色设计系统 =====
# 设计语言 (以 fig4 锚点, 用户拍板): 判决色语义 (中性=灰/危险=朱红/受益=蓝绿)
# + 数值直标 + 结论徽章 + 零杂物
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermilion": "#D55E00",
      "grey": "#999999"}
C_DANGER = OI["vermilion"]      # 危险/天真/上界
C_SAFE = OI["green"]            # 受益/完整算法
C_NEUTRAL = "#8C8C8C"           # 中性/基准
C_UB, C_LB, C_BAND = C_DANGER, OI["blue"], OI["yellow"]
F_CITIES = {"sg": "Singapore", "sz": "Shenzhen", "sh": "Shanghai"}
F_CELLS = ["sg_60", "sz_60", "sh_60", "sg_90", "sz_90", "sh_90",
           "sg_120", "sz_120", "sh_120"]
F_RED, F_GREEN = "#C62828", "#2E7D32"   # 饱和判决色 (贴锚点图, 现代红绿对)
FS_LABEL, FS_TICK, FS_ANNO, FS_TITLE = 15, 12.5, 13, 14   # 温和档: 原画幅下缩印后 >=7pt 且不碰撞
ARROW = "\u2192"    # →
MINUS = "\u2212"    # −


def _fstyle():
    plt.rcParams.update({
        "svg.fonttype": "none", "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.edgecolor": "#4D4D4D", "axes.linewidth": 0.8,
        "grid.linestyle": ":", "grid.alpha": 0.45, "grid.linewidth": 0.7,
        "legend.frameon": False, "lines.linewidth": 1.8,
        "lines.markersize": 5, "figure.facecolor": "white"})


def _badge(ax, x, y, text, color, fs=FS_ANNO):
    ax.annotate(text, xy=(x, y), ha="center", va="center", fontsize=fs,
                fontweight="bold", color="white", zorder=6,
                bbox=dict(boxstyle="round,pad=0.32", facecolor=color,
                          edgecolor="none"))


def _despine(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def f5_gap_bands():
    """F5 主结果图: UB/LB 认证带状九宫格 (用户参考图语言)。"""
    _fstyle()
    j = json.loads((LOGS / "ablation4_post.json").read_text(encoding="utf-8"))
    d = {}
    for cell, row in j.items():
        for arm, a in row["arms"].items():
            d[(cell, arm)] = a
    fig, axes = plt.subplots(3, 3, figsize=(11.0, 7.2))
    for k, cell in enumerate(F_CELLS):
        ax = axes[k // 3][k % 3]
        ab, n = cell[:2], int(cell.split("_")[1])
        m1, m2, m4 = d[(cell, "M1")], d[(cell, "M2")], d[(cell, "M4")]
        x = [0, 1, 2]
        ub = [m1["ub_kJ"], m1["ub_kJ"], m4["ub_kJ"]]
        lb = [m1["lb_kJ"], m2["lb_kJ"], m2["lb_kJ"]]
        ax.fill_between(x, lb, ub, color=C_BAND, alpha=0.35, zorder=1,
                        label="Certified gap" if k == 0 else None)
        ax.plot(x, ub, "-o", color=C_UB, markersize=4.5, zorder=3,
                label="Upper bound (feasible)" if k == 0 else None)
        ax.plot(x, lb, "-s", color=C_LB, markersize=4.2, zorder=3,
                label="Certified lower bound" if k == 0 else None)
        for xi, v in zip(x, ub):
            ax.annotate(f"{v:.0f}", (xi, v), xytext=(0, 6),
                        textcoords="offset points", ha="center",
                        fontsize=FS_ANNO - 0.6, color=C_UB)
        for xi, v in zip(x, lb):
            ax.annotate(f"{v:.0f}", (xi, v), xytext=(0, -12),
                        textcoords="offset points", ha="center",
                        fontsize=FS_ANNO - 0.6, color=C_LB)
        gy = ub[0] - (ub[0] - min(lb)) * 0.22
        _badge(ax, 1.0, gy, f"{m1['gap_pct']:.1f}% {ARROW} {m4['gap_pct']:.1f}%",
               "#7A6A00", fs=FS_ANNO - 0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(["base", "+cert", "full"], fontsize=FS_TICK)
        ax.set_xlim(-0.35, 2.35)
        span = max(ub) - min(lb)
        ax.set_ylim(min(lb) - span * 0.28, max(ub) + span * 0.28)
        ax.set_title(f"{ab.upper()}{n}", fontsize=FS_TITLE,
                     fontweight="bold")
        ax.grid(axis="y")
        _despine(ax)
        if k % 3 == 0:
            ax.set_ylabel("Energy (kJ)", fontsize=FS_LABEL - 1)
    plt.tight_layout()
    plt.savefig(OUT / "F5_gap_bands.svg", bbox_inches="tight")
    plt.savefig(OUT / "F5_gap_bands.png", dpi=200, bbox_inches="tight")
    plt.close()


def f4_certificate_quality():
    """F4 证书质量双联板 (路线 2): (a) 预算曲线 (b) Farley 棘轮。"""
    _fstyle()
    fig = plt.figure(figsize=(8.2, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.02], wspace=0.42)
    # (a) det 预算曲线: 解析 det_curve_route2.log (电量约束模型, 细网格认证)
    import re as _re
    _log = (LOGS / "det_curve_route2.log").read_text(encoding="utf-8")
    pts = {int(m.group(1)): float(m.group(2))
           for m in _re.finditer(r"CURVE det_iters=(\d+)\s*:\s*LB=([\d.]+)", _log)}
    x = sorted(pts)
    y = [pts[k] for k in x]
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, y, "-o", color=C_LB, linewidth=1.8, markersize=5,
            label="fine cert (1020)")
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.0f}", (xi, yi), xytext=(0, 6),
                    textcoords="offset points",
                    ha="left" if xi == x[0] else "center",
                    fontsize=FS_ANNO - 0.6, color=C_LB)
    ax.axhline(max(y), color=C_LB, linestyle=":", alpha=0.5, linewidth=0.9)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("det iterations", fontsize=FS_LABEL - 1.5)
    ax.set_ylabel("certified LB (kJ)", fontsize=FS_LABEL - 1)
    ax.set_title(f"(a) budget {ARROW} LB (SG120)", fontsize=FS_TITLE - 1,
                 fontweight="bold")
    ax.grid()
    _despine(ax)
    # (b) Farley 棘轮
    ax = fig.add_subplot(gs[0, 1])
    a6 = json.loads(
        (LOGS / "a6_cert_tightening_route2.json").read_text(encoding="utf-8"))
    h = a6["farley_history_kJ"]
    ax.step(range(len(h)), h, where="post", color=OI["orange"], linewidth=1.6)
    ax.axhline(a6["lb_tree_kJ"], color="#212121", linestyle=":", linewidth=0.9)
    ax.annotate(f"final LB = {a6['lb_tree_kJ']:.0f} kJ",
                (len(h) * 0.42, a6["lb_tree_kJ"]), xytext=(0, 4),
                textcoords="offset points", fontsize=FS_ANNO - 0.6,
                fontweight="bold")
    ax.set_xlabel("Farley ratchet updates", fontsize=FS_LABEL - 1.5)
    ax.set_title("(b) monotone ratchet (SZ90)", fontsize=FS_TITLE - 1,
                 fontweight="bold")
    ax.grid()
    _despine(ax)
    plt.savefig(OUT / "F4_certificate_quality.svg", bbox_inches="tight")
    plt.savefig(OUT / "F4_certificate_quality.png", dpi=200, bbox_inches="tight")
    plt.close()


def f1_2d_failure():
    """F1 二维致命失败: 饱和判决色 + 共享 y 轴 + 紧凑单行徽章。"""
    _fstyle()
    p3 = json.loads((LOGS / "p3_2d3d.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.1), sharey=True)
    cell3 = [("sg", "sg_90"), ("sz", "sz_90"), ("sh", "sh_90")]
    ymax = max(p3["legs"][c]["S2"] for _, c in cell3) * 1.14
    for ax, (ab, cell) in zip(axes, cell3):
        s1 = p3["legs"][cell]["S1"]
        s2 = p3["legs"][cell]["S2"]
        s3 = p3["S3_univ_UB"][cell]
        bars = ax.bar([r"$S_1$", r"$S_2$", r"$S_3$"], [s1, s2, s3],
                      width=0.62, color=[C_NEUTRAL, F_RED, F_GREEN],
                      edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, [s1, s2, s3]):
            ax.annotate(f"{v:.0f} kJ", (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=FS_ANNO, color="#212121")
        _badge(ax, 1, s2 * 0.72, f"+{100*(s2-s1)/s1:.0f}% deficit", F_RED,
               fs=FS_ANNO - 0.8)
        _badge(ax, 2, s3 * 0.30, f"{MINUS}{100*(s2-s3)/s2:.0f}% saved", F_GREEN,
               fs=FS_ANNO - 0.8)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Total Mission Energy (kJ)" if ab == "sg" else None,
                      fontsize=FS_LABEL)
        ax.set_title(f"({chr(97+['sg','sz','sh'].index(ab))}) {F_CITIES[ab]} ($N$=90)",
                     fontsize=FS_TITLE, fontweight="bold")
        ax.grid(axis="y")
        _despine(ax)
    plt.tight_layout()
    plt.savefig(OUT / "F1_2d_fatal_failure.svg", bbox_inches="tight")
    plt.savefig(OUT / "F1_2d_fatal_failure.png", dpi=200, bbox_inches="tight")
    plt.close()


def f2_physics_panels():
    """F2 物理六联板: (a-c) 载荷-能耗 (d-f) 高度敏感性。"""
    _fstyle()
    _p2src = LOGS / "model_experiments_v2.json"
    me = json.loads(_p2src.read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(12.4, 2.9))
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 0.06, 1, 1, 1],
                          wspace=0.55)
    axes = [fig.add_subplot(gs[0, i]) for i in range(7)]
    axes[3].set_visible(False)  # 间隔列, 不渲染空轴框
    loads = np.linspace(0, 1.019, 21)
    for i, ab in enumerate(("sg", "sz", "sh")):
        ax = axes[i]
        city = {"sg": "singapore_marinabay", "sz": "shenzhen_futian",
                "sh": "shanghai_lujiazui"}[ab]
        T = np.load(GIS / f"arc_phys_table_{city}_opt.npy")
        n = T.shape[0]
        for j, ls in zip((1, n // 3, 2 * n // 3), ("-", "--", ":")):
            ax.plot(loads, T[0, j, :] / 1e3, ls, color=CITY_C[ab],
                    linewidth=1.6)
        ax.set_title(f"({chr(97+i)}) {F_CITIES[ab]}",
                     fontsize=FS_TITLE - 1, fontweight="bold")
        ax.set_xlabel("payload (kg)", fontsize=FS_LABEL - 1.5)
        ax.grid()
        _despine(ax)
        if i == 0:
            ax.set_ylabel("arc energy (kJ)", fontsize=FS_LABEL - 1)
    p2 = me["P2"]
    order = ["sg", "sz", "sh"]
    full = {"sg": "singapore_marinabay", "sz": "shenzhen_futian",
            "sh": "shanghai_lujiazui"}
    for i, ab in enumerate(order):
        ax = axes[4 + i]
        cur = p2[full[ab]]
        for w, c in (("W=20N", "#B0BEC5"), ("W=25N", "#78909C"),
                     ("W=30N", CITY_C[ab])):
            r = cur[w]["ratio_by_height"]
            ax.plot([int(hh) for hh in r], list(r.values()), "-o", color=c,
                    markersize=3.4, linewidth=1.5, label=w)
        ax.axhline(1.0, color="#616161", linestyle=":", linewidth=0.9)
        ax.set_title(f"({chr(100+i)}) {F_CITIES[ab]}", fontsize=FS_TITLE - 1,
                     fontweight="bold")
        ax.set_xlabel("forced altitude (m)", fontsize=FS_LABEL - 1.5)
        ax.grid()
        _despine(ax)
        if i == 0:
            ax.set_ylabel("energy ratio", fontsize=FS_LABEL - 1)
    plt.savefig(OUT / "F2_physics_panels.svg", bbox_inches="tight")
    plt.savefig(OUT / "F2_physics_panels.png", dpi=200, bbox_inches="tight")
    plt.close()


def f3_energy_structure():
    """F3 能量结构 (路线 2 四通道): 悬停 + 直飞/越顶/绕行三几何飞行通道, 100% 堆叠。"""
    _fstyle()
    p4 = json.loads((LOGS / "p4_energy_decomposition.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1), sharey=True)
    chan = [("hover_kJ", "Hover service", "#9E9E9E"),
            ("flight_direct_kJ", "Direct cruise", OI["sky"]),
            ("flight_overtop_kJ", "Overtop climb", OI["orange"]),
            ("flight_bypass_kJ", "Bypass detour", C_DANGER)]
    for ax, ab in zip(axes, ("sg", "sz", "sh")):
        left = np.zeros(3)
        scales = ["60", "90", "120"]
        tot = [p4[f"{ab}_{n}"]["sum_kJ"] for n in scales]
        for key, lab, c in chan:
            vals = np.array([100.0 * p4[f"{ab}_{n}"][key] /
                             p4[f"{ab}_{n}"]["sum_kJ"] for n in scales])
            ax.barh(range(3), vals, left=left, color=c, edgecolor="white",
                    linewidth=0.8, label=lab, height=0.62)
            for yi, (v, l0) in enumerate(zip(vals, left)):
                if v >= 10:
                    ax.text(l0 + v / 2, yi, f"{v:.0f}%", ha="center",
                            va="center", fontsize=FS_ANNO,
                            color="white" if key == "flight_bypass_kJ"
                            else "#212121",
                            fontweight="bold")
                elif v >= 3:
                    ax.text(l0 + v / 2, yi - 0.50, f"{v:.0f}%", ha="center",
                            va="center", fontsize=FS_TICK,
                            color=c, fontweight="bold")
            left += vals
        for yi, t in enumerate(tot):
            ax.text(101, yi, f"{t:.0f} kJ", ha="left", va="center",
                    fontsize=FS_ANNO - 0.4, color="#212121")
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "100%"])
        ax.invert_yaxis()
        ax.set_yticks(range(3))
        ax.set_yticklabels(["N=60", "N=90", "N=120"], fontsize=FS_TICK)
        ax.set_title(F_CITIES[ab], fontsize=FS_TITLE, fontweight="bold")
        ax.grid(axis="x")
        _despine(ax)
    axes[0].set_ylabel("Fleet scale", fontsize=FS_LABEL)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.02), fontsize=FS_TICK)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(OUT / "F3_energy_structure.svg", bbox_inches="tight")
    plt.savefig(OUT / "F3_energy_structure.png", dpi=200, bbox_inches="tight")
    plt.close()




def main_paper():
    f5_gap_bands(); print("F5 ok")
    f4_certificate_quality(); print("F4 ok")
    f1_2d_failure(); print("F1 ok")
    f2_physics_panels(); print("F2 ok")
    f3_energy_structure(); print("F3 ok")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main_paper()
