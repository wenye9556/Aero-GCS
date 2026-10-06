# -*- coding: utf-8 -*-
"""统一实验驱动（2026-10-02 定稿; 2026-09-30 后置解耦 v2 修订）：取代全部一次性战役补丁脚本。

五种模式（CLI）：
  python 实验.py selfcheck                        # 守卫自检（默认值锁定 + 后置面 + 能量门）
  python 实验.py run --city sh --nodes 60 [--budget 500] [--K 5] [--out x.json]
                                                  # 通用配置串单格求解（含后置组件）
  python 实验.py ablation --city sz --nodes 90 [--all] [--K 8]
                                                  # M1-M4 四臂消融（共享轨迹, 单次求解）
  python 实验.py model                            # P1 载荷 / P2 高度 / P5 实例统计
  python 实验.py p3 --city sg --nodes 120         # P3 平面重解（S1/S2, 需先有平面张量）
  python 实验.py p4                               # P4 四通道能量分解 (悬停+三几何飞行, oracle 重放)

通用配置串（唯一部署形态, 零城市分支, 详见 算法.md）：
  基础串（四臂共用）: ng_endgame=8; dp_mode=frontier if N>=90 else legacy;
    dp_grid 1020/126; 其余默认 + record_sites=True（被动录制, 零轨迹扰动）
  组件A（后置证书器）: certify_post(cert_grid=1020, window_s=120, max_calls=8)
  组件B（终局单调内核）: mono_kernel(share=0.25×T_MAX)
单调性（v2 核心修正, 按构造成立而非统计）:
  四臂 = 同一次基础运行的四个读数 ⇒ LB(M1)≤LB(M2)≤LB(M4), UB(M1)=UB(M2)≥UB(M3)≥UB(M4),
  gap 逐臂单调不增; 旧形态（在环证书 + 搜索期 larr shell）存在墙钟/列池轨迹耦合,
  6/9 格违反排序, 已废弃（保留引擎参数仅为复现历史腿）。
预算口径：60n=500s、90n=1000s、120n=2000s（anytime 引擎, 基础求解时间是输入;
组件A/B 为后置处理, 其墙钟单列报告; 120n 预算 2026-10-05 用户裁决延长）。
电量硬约束（路线 2, 2026-10-05）: E_route ≤ 360 kJ (参数.BATTERY_SPECS) 为逐航线
硬约束进模型; A* 标签四维 (rc,mask,load,E), frontier 内核三维标号 (g,rc,E),
legacy/u_table 为能量不感知超集松弛; --K 为机队规模覆盖 (试点用)。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
GIS = ROOT / "results" / "gis_data"
LOGS = ROOT / "results" / "logs"

CITIES = {"sg": "singapore_marinabay", "sz": "shenzhen_futian", "sh": "shanghai_lujiazui"}

# 组件后置参数（通用: 与城市/规模无关的常数 + 随预算自适应的窗口份额）
CERT_POST = dict(cert_grid=1020, window_s=120.0, max_calls=8)
KERNEL_SHARE = 0.25

# 机队规模 K 覆盖（CLI --K; 0=不覆盖走默认公式）。试点定版后公式冻结。
K_OVERRIDE = 0

# 求解预算（秒）: 60:500 / 90:1000 / 120:2000 (2026-10-05 用户裁决延长 120 档)
BUDGETS_S = {60: 500.0, 90: 1000.0, 120: 2000.0}


def default_budget(n: int) -> float:
    return BUDGETS_S[n]


def fleet_k(dem_sum_g: float, n: int) -> int:
    """机队规模 K 规则（默认公式）。

    路线 2 候选定版（待 K 可行性试点冻结）: 容量轴 ceil(Σd/1000) 为基线;
    60 档维持历史值不变（约束不 binding, 回归可比）; n≥90 档并入能量轴余量
    +ceil((n-60)/30)（90:+1, 120:+2）—— 按规模不按城市, 与 base_kw 哲学一致。
    """
    k_cap = int(np.ceil(dem_sum_g / 1000.0 - 1e-9))
    if n < 90:
        return k_cap
    return k_cap + int(np.ceil((n - 60) / 30.0))


def load_instance(ab: str, n: int):
    city = CITIES[ab]
    cust = json.loads((GIS / f"{city}_{n}nodes.json").read_text(encoding="utf-8"))["customers"]
    t = f"arc_phys_table_{city}_opt.npy" if n == 60 else f"arc_phys_table_{city}_{n}nodes_opt.npy"
    T = np.load(GIS / t)
    dem_sum = sum(c.get("demand_kg", 0.0) * 1000.0 for c in cust)
    K = fleet_k(dem_sum, n)
    if K_OVERRIDE > 0:
        K = int(K_OVERRIDE)
    return city, cust, T, K


def base_kw(n: int) -> dict:
    """通用基础串: M1-M4 四臂共用（组件不在求解 kwargs 中分层）。"""
    kw = dict(ng_endgame=8, dp_mode="legacy", dp_grid=126, dp_time_share=0.0,
              astar_bound=False, t2b_max_cols=40, record_sites=True)
    if n >= 90:                       # 唯一自适应规则: 按规模, 非城市
        kw["dp_mode"] = "frontier"    # [路线 2] 90 档起切能量感知 frontier 标号
        kw["dp_grid"] = 1020
    return kw


def solve_leg(ab: str, n: int, budget: float, seed: int = 42,
              tensor=None, tag: str = ""):
    """共享轨迹单腿: 一次基础求解 + 组件A/B后处理 ⇒ M1-M4 四臂读数。

    M1=基础, M2=+A(证书后置), M3=+B(单调内核), M4=完整。
    单调性按构造: A 只升 LB (UB 逐位不变), B 只降 UB (LB 逐位不变)。
    """
    import 算法 as G
    city, cust, T, K = load_instance(ab, n)
    eng = G.BPEngine(tensor if tensor is not None else T, cust, K=K,
                     dynamic_payload=True, time_budget_s=budget, seed=seed,
                     verbose=False, **base_kw(n))
    t0 = time.time()
    r1 = eng.solve()
    t_base = time.time() - t0
    ub1, lb1 = eng.ub, r1["lb_tree_kJ"] * 1e3

    t0 = time.time()
    lb2 = eng.certify_post(lb1, ub1, **CERT_POST)
    t_cert = time.time() - t0
    routes_m1 = eng.routes_payload()      # kernel 前 incumbent (离线内核迭代用)

    t0 = time.time()
    ub3 = eng.mono_kernel(share=KERNEL_SHARE)
    t_kern = time.time() - t0

    assert lb2 >= lb1 - 1e-6, "A 组件后置必须 LB 只升不降!"
    assert ub3 <= ub1 + 1e-6, "B 组件后置必须 UB 只降不升!"
    arms = {}
    for name, lb, ub in (("M1", lb1, ub1), ("M2", lb2, ub1),
                         ("M3", lb1, ub3), ("M4", lb2, ub3)):
        arms[name] = {"lb_kJ": round(lb / 1e3, 4), "ub_kJ": round(ub / 1e3, 4),
                      "gap_pct": round(100.0 * (ub - lb) / ub, 4)}
    assert arms["M1"]["gap_pct"] >= arms["M2"]["gap_pct"] >= arms["M4"]["gap_pct"]
    assert arms["M1"]["gap_pct"] >= arms["M3"]["gap_pct"] >= arms["M4"]["gap_pct"]

    head = f"LEG {ab}_{n} {tag} seed={seed}"
    for name in ("M1", "M2", "M3", "M4"):
        a = arms[name]
        print(f"{head} {name}: LB={a['lb_kJ']:.2f} UB={a['ub_kJ']:.2f} "
              f"gap={a['gap_pct']:.2f}%", flush=True)
    print(f"{head} times: base={t_base:.0f}s cert_post={t_cert:.0f}s "
          f"kernel={t_kern:.0f}s sites={len(eng.site_log)} "
          f"cert_gain={eng.cert_post_gain/1e3:.2f} kJ "
          f"kernel_gain={eng.kernel_gain/1e3:.2f} kJ", flush=True)
    times = {"t_base_s": round(t_base, 1), "t_cert_post_s": round(t_cert, 1),
             "t_kernel_s": round(t_kern, 1), "sites": len(eng.site_log),
             "iters": r1["cg_iterations"]}
    return arms, eng, times, routes_m1


# ---------- 模式 ----------
def mode_selfcheck(a=None) -> None:
    import 算法 as G
    import inspect
    d = inspect.signature(G.BPEngine.__init__).parameters
    assert G.BPEngine.__init__.__defaults__[-4:] == (False, True, True, False), \
        "引擎默认值被改动, 证书链风险!"
    assert d["record_sites"].default is False, "record_sites 必须默认关!"
    src = (ROOT / "算法.py").read_text(encoding="utf-8")
    node = src.split("def _node_cg", 1)[1].split("\n    def solve", 1)[0]
    assert "separate_rci(" not in node
    for token in ("def certify_post", "def mono_kernel", "def routes_payload",
                  "def _record_site"):
        assert token in src, f"后置面缺失: {token}"
    # 位点录制必须被动: 不得读时钟做分支/不得触碰 RNG 或列池
    rec = src.split("def _record_site", 1)[1].split("\n    def ", 1)[0]
    assert "rng" not in rec and "cols" not in rec and "heap" not in rec
    # [路线 2] 电量硬约束守卫: 常数 + 各定价/构造/邻域能量门在位
    import 参数 as P
    assert G.E_MAX_J == P.BATTERY_SPECS["capacity_kJ"] * 1000.0 == 360000.0, \
        "E_MAX_J 必须锚定 参数.BATTERY_SPECS.capacity_kJ"
    for fn, needle in (
            ("def a_star(self", "en_vec"), ("def a_star(self", "E_MAX_J"),
            ("def greedy_insertion", "E_MAX_J"),
            ("def fleet_local_search", "E_MAX_J"),
            ("def destroy_repair", "E_MAX_J"),
            ("def best_cross_moves", "E_MAX_J"),
            ("def swap_pass", "E_MAX_J"),
            ("def greedy_fleet", "E_MAX_J"),
            ("def assert_partition", "E_MAX_J"),
            ("def _kernel_v2", "E_max"),
            ("def _consume_dp_route", "energy_feasible")):
        seg = src.split(fn, 1)[1].split("\ndef ", 1)[0]
        assert needle in seg, f"能量门缺失: {fn} 无 {needle}"
    cg_seg = src.split("def _node_cg", 1)[1].split("\n    def ", 1)[0]
    assert "E_MAX_J" in cg_seg, "CG 列准入全局能量门缺失"
    solve_seg = src.split("def solve", 1)[1]
    assert "energy_feasible" in solve_seg, "solve() 种子能量过滤缺失"
    print("selfcheck PASS: 默认值锁定 + 节点CG守卫 + 后置解耦面 + 电量硬约束门完整")


def mode_run(a) -> None:
    budget = a.budget if a.budget else default_budget(a.nodes)
    arms, eng, times, _ = solve_leg(a.city, a.nodes, budget, a.seed, tag="univ")
    m4 = arms["M4"]
    print(f"UNIV {a.city}_{a.nodes}: LB={m4['lb_kJ']:.2f} UB={m4['ub_kJ']:.2f} "
          f"gap={m4['gap_pct']:.2f}% total={sum(v for k, v in times.items() if k.endswith('_s')):.0f}s")
    if a.out:
        r = {"ub_kJ": m4["ub_kJ"], "lb_tree_kJ": m4["lb_kJ"],
             "gap_pct": m4["gap_pct"], "arms": arms, "times": times,
             "routes": eng.routes_payload()}
        Path(a.out).write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"routes -> {a.out}")


def mode_ablation(a) -> None:
    if a.all:
        cells = [(ab, n) for ab in CITIES for n in (60, 90, 120)]
    else:
        assert a.city and a.nodes, "单格模式需要 --city 与 --nodes"
        cells = [(a.city, a.nodes)]
    out_path = LOGS / ("ablation4_post.json" if a.all
                       else f"ablation4_post_{a.city}_{a.nodes}.json")
    results = {}
    for ab, n in cells:
        budget = a.budget if a.budget else default_budget(n)
        arms, eng, times, routes_m1 = solve_leg(ab, n, budget, a.seed)
        results[f"{ab}_{n}"] = {"arms": arms, "times": times,
                                "routes_M1": routes_m1,
                                "routes_M4": eng.routes_payload()}
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"saved -> {out_path}")


def mode_model(a=None) -> None:
    """P1/P2/P5（与 results/logs/model_experiments.json 同口径）。"""
    rng = np.random.default_rng(42)
    out = {"P1": {"per_instance": {}}, "P2": {}, "P5": {}}
    for ab, city in CITIES.items():
        for n in (60, 90, 120):
            t = np.load(GIS / (f"arc_phys_table_{city}_opt.npy" if n == 60
                               else f"arc_phys_table_{city}_{n}nodes_opt.npy"))
            m = t.shape[0]
            pairs = sorted({(0, j) for j in range(1, m)} |
                           {tuple(sorted(p)) for p in rng.integers(1, m, size=(400, 2))
                            if p[0] != p[1]})[:300]
            ratios = [t[i, j, -1] / t[i, j, 0] for i, j in pairs]
            viol = int(sum(np.any(np.diff(t[i, j, :]) < -1e-9) for i, j in pairs))
            out["P1"]["per_instance"][f"{ab}_{n}"] = {
                "full_over_empty_ratio_mean": float(np.mean(ratios)),
                "mono_violations": viol, "arcs": len(pairs)}
            cust = json.loads((GIS / f"{city}_{n}nodes.json").read_text(encoding="utf-8"))["customers"]
            dem = np.array([c.get("demand_kg", 0.0) * 1000 for c in cust[1:]])
            pos = np.array([c["pos"] for c in cust])
            d = np.linalg.norm(pos[1:, None, :2] - pos[1:, :2][None], axis=-1)
            out["P5"][f"{ab}_{n}"] = {
                "customers": len(cust) - 1,
                "K_min": int(np.ceil(dem.sum() / 1019.0 - 1e-9)),
                "capacity_tightness": float(dem.sum() / (np.ceil(dem.sum() / 1019.0) * 1019.0)),
                "pair_dist_m_mean": float(d.mean())}
            if n == 60:
                # P2 高度敏感性（规范协议, 2026-10-05 入库）: P1 同款 300 弧样本（60 档）。
                # 语义: 平飞最优剖面基准 vs 强制高度偏移剖面——在弧段端点平面之上
                # 抬升 h 米、巡航、再下降 h 米 (h=10..120), 两者均由
                # optimal_profile_energy 在同一速度包线内变分寻优。悬停服务能耗与
                # 高度无关, 口径双侧剔除; 比值对弧取均值, 随 h 近线性、随全重单调放大。
                # 注: 历史 model_experiments.json 的 P2 由未入库的一次性脚本生成,
                # 采样与规范协议存在差异; 本规范重算为论文数字的最终依据。
                import math
                import 模型 as M
                gd = json.loads((GIS / f"{city}_{n}nodes.json").read_text(encoding="utf-8"))
                pos2 = [c["pos"] for c in gd["customers"]]
                d_xy = [math.hypot(pos2[j][0] - pos2[i][0], pos2[j][1] - pos2[i][1])
                        for i, j in pairs]
                for W in (20.0, 25.0, 30.0):
                    base = np.array([M.optimal_profile_energy(L, 0.0, 0.0, W) for L in d_xy])
                    r_h = {}
                    for h in range(10, 121, 10):
                        forced = [M.optimal_profile_energy(L, float(h), float(h), W)
                                  for L in d_xy]
                        r_h[str(h)] = float(np.mean(np.array(forced) / base))
                    out["P2"].setdefault(city, {})[f"W={int(W)}N"] = {
                        "flat_energy_kJ_mean": float(base.mean() / 1e3),
                        "ratio_by_height": r_h}
                print(f"P2 {city} done", flush=True)
        print(f"P1/P5 {city} done", flush=True)
    (LOGS / "model_experiments_v2.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> results/logs/model_experiments_v2.json")


def mode_p3(a) -> None:
    """P3: 平面张量上的通用串重解 (S1) + 3D 真值评估 (S2), 增量写回 p3_2d3d.json。"""
    import 算法 as G
    city, cust, T3d, K = load_instance(a.city, a.nodes)
    Tpl = np.load(GIS / f"arc_planar_table_{city}_{a.nodes}nodes.npy")
    arms, eng, times, _ = solve_leg(a.city, a.nodes, default_budget(a.nodes),
                                 tensor=Tpl, tag="planar")
    costs3d = G.ArcCosts(T3d, eng.demands_g, dynamic_payload=True)
    s2 = sum(costs3d.route_cost(rt["sequence"]) for rt in eng.routes_payload()) / 1e3
    print(f"P3 {a.city}_{a.nodes}: S1={arms['M4']['ub_kJ']:.2f} S2={s2:.2f}", flush=True)

    cell = f"{a.city}_{a.nodes}"
    p = LOGS / "p3_2d3d.json"
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    d.setdefault("legs", {})[cell] = {"S1": arms["M4"]["ub_kJ"],
                                      "S2": round(s2, 4)}
    d.setdefault("S3_univ_UB", {})
    try:    # S3 = 3D 战役完整算法 UB (ablation4_post.json M4)
        ab = json.loads((LOGS / "ablation4_post.json").read_text(encoding="utf-8"))
        d["S3_univ_UB"][cell] = ab[cell]["arms"]["M4"]["ub_kJ"]
    except (OSError, KeyError):
        pass
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"updated -> {p}", flush=True)


def mode_p4(a=None) -> None:
    """P4: 四通道能量分解 (悬停 + 直飞/越顶/绕行三几何飞行通道, oracle 重放)。

    数据源: results/logs/ablation4_post.json 的 routes_M4 (各格最终解, 已过
    引擎终局能量断言)。方法: 逐弧在精确载荷处重放 TangentOracle3D ——
    branch ∈ {direct, overtop, bypass} 给飞行能量 ub; 悬停 = P_hover(W)×svc[v]
    (v≠0)。校验: 逐弧 (ub+hover) 求和 vs 张量插值 route_cost 的残差
    (recon_err_kJ, 21 档插值所致, 历史量级 ≤3 kJ)。
    写出: p4_energy_decomposition.json (新 schema: hover_kJ + 三飞行通道;
    旧字段 flat/vert/obs 保留为含悬停几何桶口径) 与 p4_routes/*.json。
    """
    import 算法 as G
    import 模型 as Mo
    import 参数 as P
    ab4 = json.loads((LOGS / "ablation4_post.json").read_text(encoding="utf-8"))
    out = {}
    routes_dir = LOGS / "p4_routes"
    routes_dir.mkdir(exist_ok=True)
    W0 = Mo.ENV_CONSTANTS["W_min"]
    g0 = Mo.ENV_CONSTANTS["g"]
    for cell, cell_d in sorted(ab4.items()):
        ab, n = cell.split("_")
        city, cust, T, _ = load_instance(ab, int(n))
        data = json.loads((GIS / f"{city}_{n}nodes.json").read_text(encoding="utf-8"))
        pos = {c["id"]: c["pos"] for c in cust}
        svc = {c["id"]: float(c.get("service_time_s", 0.0)) for c in cust}
        dem = {c["id"]: int(round(c.get("demand_kg", 0.0) * 1000.0)) for c in cust}
        blds = list(data.get("buildings", []))
        for b in data.get("skybridges", []):     # 保守全棱柱 (与张量构建一致)
            blds.append({"id": b.get("id"), "polygon": b.get("polygon"),
                         "z_max": b.get("z_max")})
        oracle = Mo.TangentOracle3D(blds, h_safe=5.0, safety_margin_2d=1.0)
        ch = {"direct": 0.0, "overtop": 0.0, "bypass": 0.0}   # 飞行通道 (J)
        ch_hov = {"direct": 0.0, "overtop": 0.0, "bypass": 0.0}
        counts = {"direct": 0, "overtop": 0, "bypass": 0}
        hover_sum, flight_sum, replay_sum = 0.0, 0.0, 0.0
        routes_out, route_E = [], []
        for rt in cell_d["routes_M4"]:
            seq = rt["sequence"]
            legs_out = []
            load = sum(dem[u] for u in seq[1:-1])
            cur = load
            for t in range(len(seq) - 1):
                u, v = seq[t], seq[t + 1]
                qkg = cur / 1000.0
                branch, _, ub = oracle.evaluate_3d_arc_branches(
                    pos[u], pos[v], qkg, use_optimal_speed=True)
                W = W0 + qkg * g0
                hov = (Mo.compute_bemt_power(0.0, 0.0, W) * svc[v]) if v != 0 else 0.0
                ch[branch] += ub
                ch_hov[branch] += hov
                counts[branch] += 1
                hover_sum += hov
                flight_sum += ub
                replay_sum += ub + hov
                legs_out.append({"u": int(u), "v": int(v), "load_g": int(cur),
                                 "branch": branch, "flight_J": round(ub, 3),
                                 "hover_J": round(hov, 3)})
                if v != 0:
                    cur -= dem[v]
            routes_out.append({"sequence": [int(x) for x in seq],
                               "load_g": int(load),
                               "cost_kJ": rt["cost_kJ"], "legs": legs_out})
            route_E.append(rt["cost_kJ"])
        ub_kJ = cell_d["arms"]["M4"]["ub_kJ"]
        sum_kJ = replay_sum / 1e3
        out[cell] = {
            "ub_kJ": ub_kJ, "sum_kJ": round(sum_kJ, 3),
            "recon_err_kJ": round(sum_kJ - ub_kJ, 3),
            "hover_kJ": round(hover_sum / 1e3, 3),
            "flight_kJ": round(flight_sum / 1e3, 3),
            "flight_direct_kJ": round(ch["direct"] / 1e3, 3),
            "flight_overtop_kJ": round(ch["overtop"] / 1e3, 3),
            "flight_bypass_kJ": round(ch["bypass"] / 1e3, 3),
            "leg_counts": counts, "routes_n": len(routes_out),
            "max_route_E_kJ": round(max(route_E), 3),
            "E_MAX_kJ": P.BATTERY_SPECS["capacity_kJ"],
            # 向后兼容旧口径 (含悬停的几何桶)
            "flat": round((ch["direct"] + ch_hov["direct"]) / 1e3, 3),
            "vert": round((ch["overtop"] + ch_hov["overtop"]) / 1e3, 3),
            "obs": round((ch["bypass"] + ch_hov["bypass"]) / 1e3, 3),
        }
        (routes_dir / f"{cell}_routes.json").write_text(
            json.dumps({"ub_kJ": ub_kJ, "routes": routes_out},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        assert abs(sum_kJ - ub_kJ) <= 5.0, f"{cell} 重放残差 {sum_kJ-ub_kJ:.2f} kJ 超限"
        assert max(route_E) <= P.BATTERY_SPECS["capacity_kJ"] + 1e-3, \
            f"{cell} 航线超电量!"
        print(f"P4 {cell}: hover={out[cell]['hover_kJ']:.1f} "
              f"direct={out[cell]['flight_direct_kJ']:.1f} "
              f"overtop={out[cell]['flight_overtop_kJ']:.1f} "
              f"bypass={out[cell]['flight_bypass_kJ']:.1f} "
              f"maxE={out[cell]['max_route_E_kJ']:.1f} kJ", flush=True)
    (LOGS / "p4_energy_decomposition.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved -> results/logs/p4_energy_decomposition.json "
          "+ results/logs/p4_routes/*.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("selfcheck")
    p = sub.add_parser("run")
    p.add_argument("--city", required=True, choices=CITIES)
    p.add_argument("--nodes", type=int, required=True, choices=(60, 90, 120))
    p.add_argument("--budget", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--K", type=int, default=None, help="机队规模覆盖 (默认公式)")
    p.add_argument("--out", default=None)
    p = sub.add_parser("ablation")
    p.add_argument("--city", default=None, choices=list(CITIES) + [None])
    p.add_argument("--nodes", type=int, default=None, choices=(60, 90, 120, None))
    p.add_argument("--budget", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--K", type=int, default=None, help="机队规模覆盖 (默认公式)")
    p.add_argument("--all", action="store_true", help="9 格顺序战役")
    sub.add_parser("model")
    p = sub.add_parser("p3")
    p.add_argument("--city", required=True, choices=CITIES)
    p.add_argument("--nodes", type=int, required=True, choices=(60, 90, 120))
    sub.add_parser("p4")
    a = ap.parse_args()
    if getattr(a, "K", None):
        globals()["K_OVERRIDE"] = int(a.K)
    {"selfcheck": mode_selfcheck, "run": mode_run, "ablation": mode_ablation,
     "model": mode_model, "p3": mode_p3, "p4": mode_p4}[a.mode](a)


if __name__ == "__main__":
    main()
