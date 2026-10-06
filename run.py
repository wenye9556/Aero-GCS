#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aero-GCS 通用入口（开源版唯一入口）。

一条配置串跑所有城市/规模 —— 零城市分支表，唯一自适应规则按实例规模 N：

    基础串:  ng_endgame = 8
             dp_mode    = "frontier" if N >= 90 else "legacy"
                         # 90 档起切能量感知前向标号 (带 2M 止损→legacy@126 回退)
             dp_grid    = 1020 (frontier) / 126 (legacy)
             record_sites = True               # 被动位点录制, 零轨迹扰动
    机队规模: K = ceil(Σd/1kg) + max(0, ceil((n-60)/30))   # 90/120 档含电量余量项
    电量硬约束: 逐航线总能耗 ≤ 参数.BATTERY_SPECS.capacity_kJ (360 kJ)，
                由定价标签电量维/闭合门/列准入门/终局断言按构造保证
    组件A（后置证书器）: certify_post(cert_grid=1020, window_s=120, max_calls=8)
    组件B（终局单调内核）: mono_kernel(share=0.25)     # 窗口 = 0.25 × 预算

组件为同一次基础运行的后处理器（共享轨迹解耦）: A 只升 LB、B 只降 UB,
故 certified gap 相对基础解按构造单调不增 —— 与城市/规模/墙钟无关。

用法:
    python run.py --city shanghai_lujiazui --nodes 60 --budget 500
    python run.py --city singapore_marinabay --nodes 120 --budget 2000 --seed 42

输出: LB / UB / 认证 gap / 墙钟(基础+证书+内核分列), 以及可选 --out routes.json。
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from 算法 import BPEngine
from 实验 import CERT_POST, KERNEL_SHARE, base_kw, default_budget, fleet_k

ROOT = Path(__file__).resolve().parent
CITIES = ("singapore_marinabay", "shenzhen_futian", "shanghai_lujiazui")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aero-GCS certified global lower-bound solver")
    ap.add_argument("--city", required=True, choices=CITIES)
    ap.add_argument("--nodes", type=int, required=True, choices=(60, 90, 120))
    ap.add_argument("--budget", type=float, default=None,
                    help="时间预算秒; 缺省 60n=500, 90n=1000, 120n=2000")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="航线结果 json 输出路径")
    args = ap.parse_args()

    budget = args.budget if args.budget is not None else default_budget(args.nodes)
    gis = ROOT / "results" / "gis_data"
    cust = json.loads(
        (gis / f"{args.city}_{args.nodes}nodes.json").read_text(encoding="utf-8")
    )["customers"]
    tname = (f"arc_phys_table_{args.city}_opt.npy" if args.nodes == 60
             else f"arc_phys_table_{args.city}_{args.nodes}nodes_opt.npy")
    T = np.load(gis / tname)
    K = fleet_k(sum(c.get("demand_kg", 0.0) * 1000.0 for c in cust), args.nodes)

    eng = BPEngine(T, cust, K=K, dynamic_payload=True, time_budget_s=budget,
                   seed=args.seed, verbose=False, **base_kw(args.nodes))
    t0 = time.time()
    r = eng.solve()
    t_base = time.time() - t0
    if not eng.ub_routes:          # 极低预算下可能尚无可行 incumbent
        print("no feasible incumbent within budget; increase --budget")
        return
    ub1, lb1 = eng.ub, r["lb_tree_kJ"] * 1e3

    t0 = time.time()
    lb2 = eng.certify_post(lb1, ub1, **CERT_POST)      # 组件A: LB 只升
    t_cert = time.time() - t0
    t0 = time.time()
    ub2 = eng.mono_kernel(share=KERNEL_SHARE)          # 组件B: UB 只降
    t_kern = time.time() - t0
    assert lb2 >= lb1 - 1e-6 and ub2 <= ub1 + 1e-6, "后置单调性被破坏!"

    gap = 100.0 * (ub2 - lb2) / ub2
    print(f"city={args.city} nodes={args.nodes} seed={args.seed} budget={budget:.0f}s K={K}")
    print(f"LB={lb2/1e3:.2f} kJ  UB={ub2/1e3:.2f} kJ  certified gap={gap:.2f}%")
    print(f"wall: base={t_base:.0f}s + cert_post={t_cert:.0f}s + "
          f"kernel={t_kern:.0f}s = {t_base+t_cert+t_kern:.0f}s")
    if args.out:
        out = {"ub_kJ": ub2 / 1e3, "lb_tree_kJ": lb2 / 1e3, "gap_pct": gap,
               "base_lb_kJ": lb1 / 1e3, "base_ub_kJ": ub1 / 1e3,
               "cert_post_gain_kJ": (lb2 - lb1) / 1e3,
               "kernel_gain_kJ": (ub1 - ub2) / 1e3,
               "times_s": {"base": round(t_base, 1), "cert_post": round(t_cert, 1),
                           "kernel": round(t_kern, 1)},
               "routes": eng.routes_payload()}
        Path(args.out).write_text(
            json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
