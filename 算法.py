from scipy.optimize import linear_sum_assignment
"""
全局下界.py — True Branch-and-Price 认证引擎
================================================================

推翻 Price-and-Branch（根节点截断 + 死列池 IP），搭建真正的分支定价树：

  1. 统一物理成本 SSOT：连续高度解析速度剖面 + 3D 避障（direct/bypass/overtop 择优）
     的真实 BEMT 弧能耗张量 c(u,v,q)（21 载荷档线性插值，含卸载前服务悬停）。
     LB 定价与 UB 评价共用同一成本函数 ⇒ 认证 Gap 纯化为组合层整数隙。
  2. 主问题：集合分划 SP (覆盖等式 + 机队 ≤ K) 的列生成 RMP（scipy/HiGHS）。
  3. 定价子问题：动态后缀载荷逆向 ng-route A*（元素性松弛 + Pareto 支配 +
     U(j,g) 完成界 DP 剪枝）；Wentges 对偶平滑 π̃ = α·π̂ + (1-α)·π 化解 K=5
     强退化震荡；认证终止必须用真实对偶（α=0）证明无负检验数列。
  4. Ryan-Foster 弧流分支：γ_uv = Σ_r a^r_uv λ_r ∈ (0,1) 时二叉分支：
       DIFFER(u,v)：定价图删弧 + 列池清洗；
       SAME(u,v)：R1 强制后继 / R2 封锁前驱拦截（T1–T8 一致性检测）。
  5. 树下界：LB_tree = min(UB, min_{open} LB_valid)，LB_valid 仅由认证节点值与
     Farley 截断界 z + K·min(0, μ̂) 沿祖先链单调继承；未认证节点禁止界剪枝。
  6. 时限由调用方给出。Gap = (UB − LB_tree)/UB，5% 是目标，不是每个算例的保证。
     当前默认不含 AP-ELB，也不分离客户计数圆整容量割。

物理裕度教义（SSOT 参数.py）：侧向绕行 1.0 m + 翻越垂直净空 5.0 m，
空中连廊按保守全棱柱 [0, z_max] 处理；原始建筑棱柱穿楼率绝对为 0
（由 verify_bp_strict.py 独立断言）。
"""

import heapq
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.optimize as opt

import 模型 as M
import 参数 as P
from numba import njit

GIS_DIR = os.path.join("results", "gis_data")
Q_STEPS = 21
Q_MAX_KG = float(P.ENV_CONSTANTS["Q_max"])   # SSOT 锚定 参数.py (1.019 kg)
EPS_RC = 1e-3          # J: 检验数负性判定阈值
BIG_M = 1e8            # RMP 覆盖松弛大 M (J)

# ==================== 逐航线电量硬约束 (路线 2, 2026-10-05) ====================
# SSOT: 参数.BATTERY_SPECS.capacity_kJ = 360.0 kJ (100 Wh 有效可用)。
# 弧代价张量本身即能耗 (J): route_cost = 逐弧张量插值求和 = 航线总能耗
# (含逐客户服务悬停, 见 build_physical_tensor)。故电量硬约束
#     E_route = route_cost(route) <= E_MAX_J
# 是逐航线成本上限, 无需第二套能耗核算 —— 定价标签/列准入/邻域移动一律
# 以 E_MAX_J 为门, 报告口径与约束口径逐位一致。
E_MAX_J = float(P.BATTERY_SPECS["capacity_kJ"] * 1000.0)


def energy_feasible(costs: "ArcCosts", route: List[int]) -> bool:
    """电量硬约束判定: 航线总能耗 (张量插值精确值) 不超过 E_MAX_J。"""
    return costs.route_cost(route) <= E_MAX_J + 1e-6
# ---------------- 节点层触发频率观测 (默认关闭) ----------------
NODE_TIER_STATS = {
    "cg_iterations": 0, "tier1_triggered": 0, "tier2b_triggered": 0,
    "greedy_time_s": 0.0, "tier1_time_s": 0.0, "tier2b_time_s": 0.0,
    "dp_time_s": 0.0, "rmp_time_s": 0.0, "u_table_time_s": 0.0,
    "greedy_cols": 0, "tier1_cols": 0, "tier2b_cols": 0, "dp_cols": 0,
}


def node_tier_reset_stats():
    for k in NODE_TIER_STATS:
        NODE_TIER_STATS[k] = 0


GAP_TARGET = 0.05      # 5.0% 认证死线

# ---------------- Step 0 杠杆测量（观测性插桩, 默认关闭） ----------------
# 填充方: Pricer.a_star; 读取方: step0_leverage_probe.py。不改变任何行为。
A_STAR_STATS = {
    "calls": 0, "pops": 0, "exact_f_ties": 0, "ties_broken_by_mask": 0,
    "window_counts": [],      # 每次 pop: ε 窗口内候选扩展数 (按 ε 档分桶)
    "pushes": 0, "pruned_f": 0, "pruned_rc": 0, "pruned_dominated": 0,
    "elapsed_s": 0.0, "cols_emitted": 0,
    "call_tier": None,        # 调用方标记: 'tier1' / 'tier2b'
    "stop_reasons": [],       # (tier, reason, pops)
    "final_cols": [],         # 每次 call 返回时: [(rc, route), ...] — 状态级标签的真值源
    "pi_snapshot": [],         # 每次 call: (pi, sigma) — 对偶相对化用 (仅 tier1)
}
A_STAR_STATS_ON = False


def a_star_reset_stats():
    for k in ("calls", "pops", "exact_f_ties", "ties_broken_by_mask",
              "pushes", "pruned_f", "pruned_rc", "pruned_dominated",
              "elapsed_s", "cols_emitted"):
        A_STAR_STATS[k] = 0
    A_STAR_STATS["window_counts"] = []
    A_STAR_STATS["stop_reasons"] = []
    A_STAR_STATS["final_cols"] = []
    A_STAR_STATS["call_tier"] = None
    A_STAR_STATS["pi_snapshot"] = []


def pricing_round_plan(enable_two_pass, enable_throttle, early, cg_iter, late, remain_s):
    """Return (run_exact_dp, update_farley).

    The early pass may search for columns, but it must not price the bound.
    A two-pass certificate uses only an exact DP result from the current dual.
    """
    if enable_two_pass and early:
        return False, False
    if enable_two_pass or not enable_throttle:
        run_dp = True
    else:
        run_dp = cg_iter <= 3 or cg_iter % 5 == 0 or late or remain_s < 60.0
    return run_dp, run_dp or not enable_two_pass


def mark_ancestor_labels(paths, features, accepted_paths):
    """Label pricing states after the search, using only completed columns."""
    counts = {}
    for path in accepted_paths:
        for i in range(len(path)):
            suffix = path[i:]
            counts[suffix] = counts.get(suffix, 0) + 1
    y1 = np.fromiter((1 if path in counts else 0 for path in paths),
                     dtype=np.int8, count=len(paths))
    y2 = np.fromiter((counts.get(path, 0) for path in paths),
                     dtype=np.int32, count=len(paths))
    return np.asarray(features, dtype=np.float32), y1, y2


def dp_iteration_features(pi, pi_prev, lam, cols, z, z_prev, cg_iter):
    """Scale-free features for the exact-DP trigger. No city or energy level."""
    pi = np.asarray(pi, dtype=float)
    prev = pi if pi_prev is None else np.asarray(pi_prev, dtype=float)
    rel = float(np.linalg.norm(pi - prev) / (np.linalg.norm(pi) + 1.0))
    cv = float(np.std(pi) / (np.mean(np.abs(pi)) + 1.0))
    lam = np.asarray(lam, dtype=float)
    frac = float(np.max(np.minimum(lam, 1.0 - lam))) if len(lam) else 0.0
    neg = float(np.mean([c.get("rc", 0.0) < -EPS_RC for c in cols])) if cols else 0.0
    z0 = z if z_prev is None else z_prev
    return np.array([rel, cv, frac, neg, (z - z0) / (abs(z) + 1.0), min(1.0, cg_iter / 200.0)],
                    dtype=np.float64)


def destroy_repair(costs, routes, kick, Q, K, demands):
    """Exact cheapest reinsertion after removing ``kick``. None if infeasible."""
    routes = [list(r) for r in routes]
    removed = []
    for u in kick:
        for r in routes:
            if u in r:
                r.remove(u)
                removed.append(u)
                break
    routes = [r for r in routes if len(r) > 2]
    for u in removed:
        qu = int(demands[u])
        best_m, best_d = None, math.inf
        for r in routes:
            load = sum(int(demands[x]) for x in r[1:-1])
            if load + qu > Q:
                continue
            for p in range(1, len(r)):
                cand = r[:p] + [u] + r[p:]
                cc = costs.route_cost(cand)
                # 电量硬约束: 插入后航线能耗不得超限 (超限插入位不可选)
                if cc <= E_MAX_J + 1e-6 and cc - costs.route_cost(r) < best_d:
                    best_m, best_d = (r, p), cc - costs.route_cost(r)
        if best_m is None:
            routes.append([0, u, 0])
        else:
            best_m[0].insert(best_m[1], u)
    if len(routes) > K:
        return None
    refined = fleet_local_search(costs, routes, Q)
    try:
        assert_partition(refined, len(demands) - 1, K, Q, demands, costs=costs)
    except AssertionError:
        return None
    return refined


def best_cross_moves(costs, routes, Q, demands):
    """One best exact cross-route relocation of 1–3 customers."""
    routes = [list(r) for r in routes if len(r) > 2]
    if len(routes) < 2:
        return routes, False
    loads = [sum(int(demands[u]) for u in r[1:-1]) for r in routes]
    base = [costs.route_cost(r) for r in routes]
    best = None
    for a, ra0 in enumerate(routes):
        for i in range(1, len(ra0) - 1):
            for length in (1, 2, 3):
                if i + length >= len(ra0):
                    continue
                seg = ra0[i:i + length]
                qseg = sum(int(demands[u]) for u in seg)
                ra = ra0[:i] + ra0[i + length:]
                ca = costs.route_cost(ra) if len(ra) > 2 else 0.0
                for b, rb0 in enumerate(routes):
                    if a == b or loads[b] + qseg > Q:
                        continue
                    for p in range(1, len(rb0)):
                        options = (seg,) if length == 1 else (seg, list(reversed(seg)))
                        for piece in options:
                            rb = rb0[:p] + list(piece) + rb0[p:]
                            crb = costs.route_cost(rb)
                            # 电量硬约束: 目标航线并入段后不得超限 (删段后的
                            # 源航线不作前置校验——张量不保证三角不等式, 删段
                            # 未必降耗; 其整线可行性由 _try_update_ub 出口断言兜底)
                            if crb > E_MAX_J + 1e-6:
                                continue
                            delta = ca + crb - base[a] - base[b]
                            if best is None or delta < best[0]:
                                best = (delta, a, b, ra, rb)
    if best is None or best[0] >= -1e-3:
        return routes, False
    _, a, b, ra, rb = best
    routes[a], routes[b] = ra, rb
    return [r for r in routes if len(r) > 2], True


def swap_pass(costs, routes, Q, demands):
    """穷举 1-1 客户交换 (最佳改进, 容量可行) 循环至无改进 (基础 VND 未覆盖邻域)。"""
    routes = [list(r) for r in routes if len(r) > 2]
    while True:
        loads = [sum(int(demands[u]) for u in r[1:-1]) for r in routes]
        base = [costs.route_cost(r) for r in routes]
        best = None
        for a in range(len(routes)):
            for i in range(1, len(routes[a]) - 1):
                for b in range(a + 1, len(routes)):
                    for j in range(1, len(routes[b]) - 1):
                        ua, ub = routes[a][i], routes[b][j]
                        qa, qb = int(demands[ua]), int(demands[ub])
                        if loads[a] - qa + qb > Q or loads[b] - qb + qa > Q:
                            continue
                        ra = list(routes[a]); ra[i] = ub
                        rb = list(routes[b]); rb[j] = ua
                        cra, crb = costs.route_cost(ra), costs.route_cost(rb)
                        # 电量硬约束: 交换后双方航线能耗均不得超限
                        if cra > E_MAX_J + 1e-6 or crb > E_MAX_J + 1e-6:
                            continue
                        d = (cra + crb - base[a] - base[b])
                        if d < -1e-3 and (best is None or d < best[0]):
                            best = (d, a, b, ra, rb)
        if best is None:
            return routes
        _, a, b, ra, rb = best
        routes[a], routes[b] = ra, rb


def _demands_from(customers: List[dict]) -> np.ndarray:
    """客户需求 (整克) 数组，索引 = 客户 id（0 为基地）。"""
    d = np.zeros(len(customers), dtype=np.int64)
    for c in customers:
        d[c["id"]] = int(round(c.get("demand_kg", 0.0) * 1000.0))
    return d


# ==============================================================================
# 1. 物理弧成本张量（SSOT 成本函数：LB 定价与 UB 评价共用）
# ==============================================================================
def build_physical_tensor(city_key: str, use_optimal_speed: bool,
                          gis_dir: str = GIS_DIR) -> np.ndarray:
    """构建/加载连续高度 3D 避障真实 BEMT 弧能耗张量 (N,N,21)。

    - SSOT 双裕度：h_safe=5.0 m 垂直净空 + safety_margin_2d=1.0 m 侧向。
    - 空中连廊并入为保守全棱柱 [0, z_max]（绕越天桥必安全，成本为保守上界）。
    - 含卸载前（pre-release）服务悬停能耗。
    """
    tag = "opt" if use_optimal_speed else "base"
    cache = os.path.join(gis_dir, f"arc_phys_table_{city_key}_{tag}.npy")
    if os.path.exists(cache):
        return np.load(cache)

    data = json.load(open(os.path.join(gis_dir, f"{city_key}_60nodes.json"),
                          encoding="utf-8"))
    custs = data["customers"]
    blds = list(data.get("buildings", []))
    for b in data.get("skybridges", []):     # 保守全棱柱
        blds.append({"id": b.get("id"), "polygon": b.get("polygon"),
                     "z_max": b.get("z_max")})
    oracle = M.TangentOracle3D(blds, h_safe=5.0, safety_margin_2d=1.0)

    n = len(custs)
    pos = [c["pos"] for c in custs]
    svc = [c.get("service_time_s", 0.0) for c in custs]
    q_grid = np.linspace(0.0, Q_MAX_KG, Q_STEPS)
    T = np.zeros((n, n, Q_STEPS), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for qi, qkg in enumerate(q_grid):
                W = M.ENV_CONSTANTS["W_min"] + qkg * M.ENV_CONSTANTS["g"]
                _, _, ub = oracle.evaluate_3d_arc_branches(
                    pos[i], pos[j], qkg, use_optimal_speed=use_optimal_speed)
                T[i, j, qi] = ub + (M.compute_bemt_power(0.0, 0.0, W) * svc[j]
                                    if j != 0 else 0.0)
        print(f"  [tensor {city_key}/{tag}] row {i+1}/{n}", flush=True)
    np.save(cache, T)
    return T


class ArcCosts:
    """弧成本张量封装：21 载荷档线性插值（连续高度 BEMT 能耗的 SSOT 查询口）。"""

    def __init__(self, tensor: np.ndarray, demands_g: np.ndarray,
                 dynamic_payload: bool = True):
        self.M = tensor
        self.N = tensor.shape[0] - 1
        self.demands_g = demands_g
        self.dynamic = dynamic_payload
        # 载荷单调性是插值下界语义的物理前提（更重 ⇒ 更耗能）
        d = np.diff(tensor, axis=2)
        assert np.all(d >= -1e-6 * np.maximum(1.0, np.abs(tensor[:, :, :-1]))), \
            "弧能耗张量必须关于载荷单调不减"
        self._inv_step = (Q_STEPS - 1) / (Q_MAX_KG * 1000.0)  # 每克的级距

    def cost(self, u: int, v: int, load_g: int) -> float:
        """弧 (u→v) 在载重 load_g (克, 含 v 的包裹, pre-release) 上的插值能耗。"""
        if not self.dynamic or load_g <= 0:
            return self.M[u, v, 0]
        x = load_g * self._inv_step
        i0 = int(x)
        if i0 >= Q_STEPS - 1:
            return self.M[u, v, Q_STEPS - 1]
        fr = x - i0
        c0 = self.M[u, v, i0]
        return c0 + fr * (self.M[u, v, i0 + 1] - c0)

    def route_cost(self, route: List[int]) -> float:
        """航线 [0, i1, ..., im, 0] 的精确插值成本（后缀载荷动态回溯）。"""
        tot = 0
        load = 0
        for u in route[1:-1]:
            load += self.demands_g[u]
        for t in range(len(route) - 1):
            u, v = route[t], route[t + 1]
            tot += self.cost(u, v, load)
            if v != 0:
                load -= self.demands_g[v]
        return tot


# ==============================================================================
# 2. 分支上下文：Ryan-Foster SAME/DIFFER 的图编译（T1–T8 全堵）
# ==============================================================================
class BranchCtx:
    """不可变分支上下文：DIFFER 禁弧集 + SAME 链 (succ/pred)。"""

    def __init__(self, N: int):
        self.N = N
        self.differ: frozenset = frozenset()
        self.succ: Dict[int, int] = {}
        self.pred: Dict[int, int] = {}

    def child_differ(self, u: int, v: int) -> "BranchCtx":
        c = BranchCtx(self.N)
        c.differ = self.differ | {(u, v)}
        c.succ = dict(self.succ)
        c.pred = dict(self.pred)
        return c

    def child_same(self, u: int, v: int) -> Optional["BranchCtx"]:
        # 机库弧 SAME: (u,0)=u 必为末客户; (0,v)=v 必为首客户; (0,0) 非法
        if u == 0 and v == 0:
            return None
        # T1: 与已有 DIFFER 矛盾
        if (u, v) in self.differ:
            return None
        # T2: u 已有不同强制后继 / T3: v 已有不同强制前驱
        if self.succ.get(u, v) != v or self.pred.get(v, u) != u:
            return None
        c = BranchCtx(self.N)
        c.differ = self.differ
        c.succ = dict(self.succ)
        c.pred = dict(self.pred)
        c.succ[u] = v
        c.pred[v] = u
        # T8: SAME 链成环检测 (仅客户链; depot 端点自然断链)
        if u != 0:
            seen, x = {u}, u
            while x in c.succ and c.succ[x] != 0:
                x = c.succ[x]
                if x in seen:
                    return None
                seen.add(x)
        return c

    # ---------------- 图编译谓词 ----------------
    @property
    def trivial(self) -> bool:
        """无任何分支规则 (根上下文): 全部谓词 O(1) 直通。"""
        return not self.differ and not self.succ and not self.pred

    def arc_ok(self, h: int, j: int) -> bool:
        """弧 h→j (0=depot) 是否被分支规则允许（含 R1/R2 拦截, depot 感知）。"""
        if not self.differ and not self.succ and not self.pred:
            return True
        if (h, j) in self.differ:
            return False
        if h != 0 and self.succ.get(h, j) != j:
            return False        # R1: h 的唯一允许后继是 succ[h]
        if h == 0 and 0 in self.succ and self.succ[0] != j:
            return False        # R1(机库): succ[0]=v 时唯一允许始发弧 0→v
        if j != 0 and self.pred.get(j, h) != h:
            return False        # R2: j 的唯一允许前驱是 pred[j]
        if j == 0 and 0 in self.pred and self.pred[0] != h:
            return False        # R2(机库): pred[0]=u 时唯一允许收束弧 u→0
        return True

    def close_ok(self, j: int) -> bool:
        """j 可否作为航线首客户（弧 0→j, 允许 pred[j]=0）。"""
        return self.pred.get(j, 0) == 0 and self.arc_ok(0, j)

    def end_ok(self, j: int) -> bool:
        """j 可否作为航线末客户（弧 j→0, 允许 succ[j]=0）。"""
        return self.succ.get(j, 0) == 0 and self.arc_ok(j, 0)

    def col_ok(self, route: List[int]) -> bool:
        """列（完整航线）是否满足本分支上下文的全部规则。"""
        if not self.differ and not self.succ and not self.pred:
            return True
        seq = route[1:-1]
        for t in range(len(route) - 1):
            if not self.arc_ok(route[t], route[t + 1]):
                return False
        for u, v in self.succ.items():
            if u == 0 or v == 0:
                continue        # 机库弧语义已由弧检查覆盖
            has_u, has_v = u in seq, v in seq
            if has_u != has_v:
                return False    # SAME 对必须同航线共现
        return True

    def arc_mask_matrix(self) -> np.ndarray:
        """(N+1, N+1) 布尔邻接掩码（供 DP/定价向量化, 按上下文缓存）。"""
        cached = getattr(self, "_mask_cache", None)
        if cached is not None:
            return cached
        A = np.ones((self.N + 1, self.N + 1), dtype=bool)             if self.trivial else np.zeros((self.N + 1, self.N + 1), dtype=bool)
        if not self.trivial:
            np.fill_diagonal(A, False)
            for h in range(self.N + 1):
                for j in range(self.N + 1):
                    if h != j:
                        A[h, j] = self.arc_ok(h, j)
        self._mask_cache = A
        return A


# ==============================================================================
# 3. RMP：集合分划主问题 LP（scipy/HiGHS）+ 圆整容量割 (RCI)
# ==============================================================================
def solve_rmp(costs: ArcCosts, cols: List[dict], K: int,
              cuts: Optional[List[Tuple[int, int]]] = None,
              sris: Optional[List[int]] = None
              ) -> Tuple[float, np.ndarray, float, np.ndarray, bool, np.ndarray,
                         np.ndarray]:
    """min Σ c_r λ_r + M Σ s_i
    s.t. Σ a_ir λ_r + s_i = 1;  Σ λ_r ≤ K;
         Σ_r |r∩S| λ_r ≥ ceil(d(S)/Q)  ∀S ∈ RCI 割池;  λ, s ≥ 0.

    返回 (z, π, σ ≤ 0, λ, slack_active, θ (n_cuts,) ≥ 0)。
    割行对偶按客户可加分解 Θ_h = Σ_{S∋h} θ_S, 定价以 π' = π + Θ 进行。
    """
    N = costs.N
    n = len(cols)
    n_cuts = len(cuts) if cuts else 0
    c = np.concatenate([[col["cost"] for col in cols], np.full(N, BIG_M)])
    A_eq = np.zeros((N, n + N))
    for j, col in enumerate(cols):
        A_eq[:, j] = col["a"]
    A_eq[:, n:] = np.eye(N)
    m_min = int(np.ceil(costs.demands_g[1:].sum() / 1019.0))   # 机队基数割 (全局有效)
    sris = sris or []
    # [注] RCI 在等式覆盖 (Σ_r∋i λ = 1) 下恒满足: x(S)=Σ_r|r∩S|λ_r=|S|≥k(S),
    # 故 RCI 分离永空手 (保留接口仅作文档); 集合分划的正确割族为 SRI。
    n_soft = 2 + len(cuts or [])               # 机队上/下界行的软松弛列
    c = np.concatenate([c, np.full(n_soft, BIG_M)])
    A_eq = np.hstack([A_eq, np.zeros((N, n_soft))])
    # Σλ − s0 ≤ K;  −Σλ − s1 ≤ −m;  −Σ cnt λ − s_c ≤ −kS (行内松弛防子池不可行)
    row0 = np.concatenate([np.ones(n), np.zeros(N),
                           np.array([-1.0] + [0.0] * (n_soft - 1))])
    row1 = np.concatenate([-np.ones(n), np.zeros(N),
                           np.array([0.0, -1.0] + [0.0] * (n_soft - 2))])
    rows_ub = [row0, row1]
    b_ub = [float(K), -float(m_min)]
    for (S_bits, kS) in (cuts or []):
        row = np.zeros(n + N + n_soft)
        for j, col in enumerate(cols):
            cnt = bin(col["bits"] & S_bits).count("1")
            if cnt:
                row[j] = -cnt
        row[n + N + 2 + (len(b_ub) - 2)] = -1.0     # 该割行的软松弛
        rows_ub.append(row)           # −Σ cnt λ − s ≤ −kS
        b_ub.append(-float(kS))
    for t3, S_bits in enumerate(sris):
        row = np.zeros(n + N + n_soft)
        for j, col in enumerate(cols):
            cnt = bin(col["bits"] & S_bits).count("1")
            if cnt >= 2:
                row[j] = cnt // 2     # floor(|r∩S|/2)
        rows_ub.append(row)           # Σ floor(|r∩S|/2) λ_r ≤ 1 (硬约束:
        b_ub.append(1.0)              #  单客户列 floor=0 ⇒ 池恒可满足)
    A_ub = np.vstack(rows_ub)
    res = opt.linprog(c, A_ub=A_ub, b_ub=np.asarray(b_ub), A_eq=A_eq,
                      b_eq=np.ones(N), bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"RMP LP 失败: {res.message}")
    pi = np.asarray(res.eqlin.marginals, dtype=float)
    # σ_eff = σ(≤K, ≤0) − ν(≥m, ≥0): rc = c − Σπ − σ_eff, 与既有约定一致
    # (m≤K 恒成立: m<K 时基数行不结合 ν=0; m=K 时 Farley 修正项 (m−K)ν=0 ——
    #  界方向严格保持, 见论文定理 A 实现注记)
    sigma = float(res.ineqlin.marginals[0]) - float(res.ineqlin.marginals[1])
    theta = -np.asarray(res.ineqlin.marginals[2:2 + n_cuts], dtype=float) \
        if n_cuts else np.zeros(0)
    lam = np.asarray(res.x[:n], dtype=float)
    slack_active = bool(res.x[n:].sum() > 1e-6)
    theta = -np.asarray(res.ineqlin.marginals[2:2 + n_cuts], dtype=float) \
        if n_cuts else np.zeros(0)
    n_sri = len(sris)
    sri_pen = np.asarray(res.ineqlin.marginals[2 + n_cuts:2 + n_cuts + n_sri],
                         dtype=float) if n_sri else np.zeros(0)   # ≤ 0
    lam = np.asarray(res.x[:n], dtype=float)
    slack_active = bool(res.x[n:].sum() > 1e-6)
    return float(res.fun), pi, sigma, lam, slack_active, theta, sri_pen


def cut_thetas_to_pi(costs: ArcCosts, cuts: List[Tuple[int, int]],
                     theta: np.ndarray, sris: Optional[List[int]] = None,
                     sri_pen: Optional[np.ndarray] = None) -> np.ndarray:
    """Θ_h = Σ_{S∋h} θ_S (RCI) + Σ_{S∋h} pen_S/2 (SRI 半权) → π' = π + Θ。

    SRI 半权分解: floor(|r∩S|/2)·pen ≤ Σ_{h∈r∩S} pen/2 对 pen ≥ 0 成立
    ⇒ DP 定价罚不超过真罚 ⇒ rc_dp ≤ rc_true, 认证链保持可采纳 (命题 A-3)。
    """
    Theta = np.zeros(costs.N)
    for (S_bits, _kS), th in zip(cuts, theta):
        if th <= 0.0:
            continue
        for h in range(1, costs.N + 1):
            if S_bits >> (h - 1) & 1:
                Theta[h - 1] += th
    pens = sri_pen if sri_pen is not None else np.zeros(0)
    for S_bits, pen in zip(sris or [], pens):
        if pen >= -1e-9:
            continue
        half = -pen / 2.0
        for h in range(1, costs.N + 1):
            if S_bits >> (h - 1) & 1:
                Theta[h - 1] += half
    return Theta


def separate_sri(costs: ArcCosts, cols: List[dict], lam: np.ndarray,
                 existing: set, max_cuts: int = 8) -> List[int]:
    """SRI-3 分离: 支撑列上 Σ_r floor(|r∩S|/2)λ_r > 1+ε 的三元组 S。"""
    import itertools
    support = [(col, lam[j]) for j, col in enumerate(cols) if lam[j] > 1e-4]
    if not support:
        return []
    cand = set()
    for col, _ in support:
        cand.update(col["route"][1:-1])
    cand_list = sorted(cand)
    if len(cand_list) < 3:
        return []
    out = []
    for s1, s2, s3 in itertools.combinations(cand_list, 3):
        bits = (1 << (s1 - 1)) | (1 << (s2 - 1)) | (1 << (s3 - 1))
        if bits in existing:
            continue
        lhs = 0.0
        for col, w in support:
            if bin(col["bits"] & bits).count("1") >= 2:
                lhs += w
        if lhs > 1.02:
            existing.add(bits)
            out.append(bits)
            if len(out) >= max_cuts:
                break
    return out


def reduced_cost(costs: ArcCosts, route: List[int], pi: np.ndarray,
                 sigma: float) -> float:
    return costs.route_cost(route) - sum(pi[u - 1] for u in route[1:-1]) - sigma


def reduced_cost_cut(costs: ArcCosts, route: List[int], pi: np.ndarray,
                     sigma: float, sris: Optional[List[int]] = None,
                     sri_pen: Optional[np.ndarray] = None) -> float:
    """割增广检验数 (Theorem 5.1 baseline 精确结算)。

    RC_cut = c − π'a − σ − Σ_S θ_S·γ_R^S,  γ_R^S = floor(|R∩S|/2)。
    SRI 行 ≤ 1 ⇒ θ_S = sri_pen ≤ 0, 故 −θ_S·γ ≥ 0 (割只增检验数)。
    """
    rc = reduced_cost(costs, route, pi, sigma)
    if not sris:
        return rc
    bits = 0
    for u in route[1:-1]:
        bits |= 1 << (u - 1)
    pens = sri_pen if sri_pen is not None else np.zeros(0)
    for S_bits, pen in zip(sris, pens):
        if pen < 0.0:
            gamma = bin(bits & S_bits).count("1") // 2
            if gamma:
                rc -= pen * gamma
    return rc


def make_col(costs: ArcCosts, route: List[int], pi: Optional[np.ndarray] = None,
             sigma: float = 0.0) -> dict:
    N = costs.N
    a = np.zeros(N)
    bits = 0
    for u in route[1:-1]:
        a[u - 1] += 1.0
        bits |= 1 << (u - 1)
    assert not (a > 1).any(), f"非初等列: {route}"
    cost = costs.route_cost(route)
    rc = reduced_cost(costs, route, pi, sigma) if pi is not None else 0.0
    return {"route": route, "cost": cost, "a": a, "rc": rc, "bits": bits}


# ==============================================================================
# 4. 定价引擎：U 表完成界 DP + 逆向 ng-route A* + Wentges 平滑钩子
# ==============================================================================
class Pricer:
    def __init__(self, costs: ArcCosts, K: int, ng_size: int = 10,
                 dynamic_payload: bool = True, ng_size_cert: int = 6):
        self.costs = costs
        self.K = K
        self.N = costs.N
        self.dem = costs.demands_g
        self.Q = int(round(Q_MAX_KG * 1000))
        # 双邻域: 大邻域供 CG 搜索多样性, 小邻域供认证级穷尽
        # (ng 超集论证与邻域大小无关, 缩窗仅收缩状态空间, 认证语义不变)
        step_g = self.Q / (Q_STEPS - 1)
        self.dem_grid = np.minimum((self.dem // step_g).astype(int), Q_STEPS - 1)
        self._ng_masks = self._static_neighborhoods(ng_size)
        self._ng_masks_cert = self._static_neighborhoods(ng_size_cert)
        self._n_mem_bits = max(3, ng_size_cert)   # 每 node 记忆槽 2^ng_size_cert
        self._ng4_tables = self._build_ng4_tables()

    def _static_neighborhoods(self, k: int) -> np.ndarray:
        N = self.N
        mid = self.costs.M[:, :, Q_STEPS // 2]
        masks = np.zeros(N + 1, dtype=object)
        for i in range(1, N + 1):
            d = mid[i, :] + mid[:, i]
            order = np.argsort(d)
            nb = [j for j in order if 1 <= j <= N and j != i][:k - 1]
            m = int(1) << (int(i) - 1)
            for j in nb:
                m |= int(1) << (int(j) - 1)
            masks[i] = m
        return masks

    # ---------------- U(j, g) 完成界 DP（walk 松弛 + 2-cycle 消除） ----------------
    def u_table(self, pi: np.ndarray, sigma: float, bctx: BranchCtx
                ) -> Tuple[np.ndarray, float]:
        """U[j, g] = 最优前缀 0→…→h→j（网格载荷 g, 含 j）的检验数下界。

        松弛：载荷 floor 量化 + 元素性松弛（walk 允许）+ 容量网格化 + 2-cycle
        消除（best/second-best 前驱回溯）⇒ 可采纳且抑制 i↔j 二循环刷对偶。
        返回 (U, mu_hat)：mu_hat 为全列最小检验数的下界（Farley 用）。

        [路线 2 | 能量不感知的超集松弛] 本 DP 刻意不追踪电量维度: 其路径空间
        (全体 walk) ⊇ 能量可行列集 Ω_E, 故 U 仍是任意能量可行完成的可采纳
        完成界 (A* 启发式有效性不变), mu_hat ≤ min_{Ω_E} rc 仍成立
        (Farley 有效性不变); 代价是界可能偏松 —— 状态空间免于能量维爆炸。
        """
        N, G = self.N, Q_STEPS - 1
        Mx = self.costs.M if self.costs.dynamic else \
            np.repeat(self.costs.M[:, :, :1], Q_STEPS, axis=2)
        A = bctx.arc_mask_matrix()
        U = np.full((N + 1, G + 1), np.inf, dtype=np.float64)
        U_prev = np.zeros((N + 1, G + 1), dtype=np.int32)   # 最优前驱
        U_sec = np.full((N + 1, G + 1), np.inf, dtype=np.float64)
        pad_pi = np.concatenate([[0.0], pi])
        dem_g = self.dem_grid
        idx = np.arange(N + 1)
        hr = np.arange(1, N + 1)
        for g in range(G, -1, -1):
            E = Mx[:, :, g]
            close = np.where(A[0, :], E[0, :] - sigma, np.inf)
            close[0] = np.inf
            shift = g + dem_g
            ok = shift <= G
            # 每个 h 在 g+dem 处的最优/次优完成（前驱区分由 U_prev 给出）
            u1 = np.full(N + 1, np.inf)
            u1[ok] = U[idx[ok], shift[ok]]
            u2 = np.full(N + 1, np.inf)
            u2[ok] = U_sec[idx[ok], shift[ok]]
            p1 = np.zeros(N + 1, dtype=np.int32)
            p1[ok] = U_prev[idx[ok], shift[ok]]
            base = np.where(A[1:, 1:], E[1:, 1:] - pad_pi[1:, None], np.inf)
            for j in range(1, N + 1):
                # h 的完成若其最优前驱恰为 j (2-cycle j→h→j) 则用次优
                use2 = (p1[1:] == j)
                comp = np.where(use2, u2[1:], u1[1:])
                cand = base[:, j - 1] + comp
                h_min = int(np.argmin(cand))
                v_min = cand[h_min]
                v_close = close[j]
                if v_min <= v_close:
                    U[j, g] = v_min
                    U_prev[j, g] = h_min + 1
                    # 次优: min(close, 第二好的 h)
                    cand2 = np.delete(cand, h_min)
                    v2 = cand2.min() if cand2.size else np.inf
                    U_sec[j, g] = min(v_close, v2)
                else:
                    U[j, g] = v_close
                    U_prev[j, g] = 0
                    U_sec[j, g] = v_min
        # mu_hat: 种子 j→0 (载荷 0) + U[j, grid(d_j)]
        e_j0 = self.costs.M[np.arange(1, N + 1), 0, 0]
        gd = self.dem_grid[1:]
        end_allow = np.array([bctx.end_ok(j) for j in range(1, N + 1)])
        mu = e_j0 - pi + U[np.arange(1, N + 1), gd]
        mu = np.where(end_allow, mu, np.inf)
        return U, float(np.min(mu))

    # ---------------- Tier-0 贪心插入定价（构造性初等, 增量 delta 评价） ----------------
    def greedy_insertion(self, pi: np.ndarray, sigma: float, bctx: BranchCtx,
                         rng: random.Random, trials: int = 14
                         ) -> List[dict]:
        """随机化贪心插入: 构造性初等列, 免疫 U 弱界。

        插入 x 于位置 p 的检验数增量用 O(len) 前缀代价差分精确计算:
          Δrc = [Σ_{t<p-1}(c_t(S_t+d_x) − c_t(S_t))]
                + c(r_{p-1}, x, S_{p-1}+d_x) + c(x, r_p, S_{p-1})
                − c(r_{p-1}, r_p, S_{p-1}) − π_x .
        """
        N, Q = self.N, self.Q
        cost = self.costs.cost
        dem = self.dem
        order_by_pi = list(np.argsort(-pi))          # 高对偶优先候选
        out, seen = [], set()
        for _ in range(trials):
            j = int(order_by_pi[rng.randrange(min(N, 20))]) + 1
            if dem[j] > Q or not (bctx.end_ok(j) and bctx.close_ok(j)):
                continue
            route = [0, j, 0]
            in_route = {j}
            while True:
                m = len(route) - 1                    # 弧数
                S = [0] * m                           # 每弧后缀载荷
                tot = sum(int(dem[u]) for u in route[1:-1])
                cur = tot
                for t in range(m):
                    S[t] = cur
                    if route[t + 1] != 0:
                        cur -= int(dem[route[t + 1]])
                base_arc = [cost(route[t], route[t + 1], S[t]) for t in range(m)]
                Bpre = [0.0] * (m + 1)
                for t in range(m):
                    Bpre[t + 1] = Bpre[t] + base_arc[t]
                best = None
                for x in order_by_pi:
                    x = int(x) + 1
                    dx = int(dem[x])
                    if x in in_route or tot + dx > Q:
                        continue
                    prefixes = [0.0] * (m + 1)
                    for t in range(m):
                        prefixes[t + 1] = prefixes[t] + \
                            cost(route[t], route[t + 1], S[t] + dx)
                    for p in range(1, m + 1):
                        a, b2 = route[p - 1], route[p]
                        if not (bctx.arc_ok(a, x) and bctx.arc_ok(x, b2)):
                            continue
                        delta = (prefixes[p - 1] - Bpre[p - 1]
                                 + cost(a, x, S[p - 1] + dx)
                                 + cost(x, b2, S[p - 1])
                                 - base_arc[p - 1] - pi[x - 1])
                        if delta < -EPS_RC and (best is None or delta < best[0]):
                            # [路线 2] 电量硬约束: 插入后整条航线能耗不得超限
                            # (delta 含 −π_x, 还原真实成本增量后与旧总成本相加)
                            if Bpre[m] + delta + pi[x - 1] > E_MAX_J + 1e-6:
                                continue
                            best = (delta, p, x)
                if best is None:
                    break
                _, p, x = best
                route = route[:p] + [x] + route[p:]
                in_route.add(x)
            if len(route) > 3 and bctx.col_ok(route):
                kt = tuple(route)
                if kt not in seen:
                    seen.add(kt)
                    col = make_col(self.costs, route, pi, sigma)
                    if col["rc"] < -EPS_RC:
                        out.append(col)
        out.sort(key=lambda c: c["rc"])
        return out

    # ---------------- ng 精确 DP 认证预言机 (多项式穷尽, 细载荷网格) ----------------
    G_FINE = 126             # 网格档数 (论文口径: legacy 粗网格档, 步长 ≈ 8.1 g;
                             #  细网格 1020 由 dp_grid/cert_grid 指定。命名历史
                             #  遗留: 相对更早的粗档得名 fine)

    def _dp_tables(self, n_bits: int, nb_override: Optional[np.ndarray] = None):
        """ng-2^n_bits DP 转移表 (按 (位数, 版本) 缓存; 支持对偶感知邻域)。"""
        key = int(n_bits)
        if nb_override is None and key in getattr(self, "_dp_cache", {}):
            return self._dp_cache[key]
        if not hasattr(self, "_dp_cache"):
            self._dp_cache = {}
        if nb_override is not None:
            self._dp_ver = getattr(self, "_dp_ver", 0) + 1
            key = (int(n_bits), self._dp_ver)
        N = self.N
        n_slot = 1 << key[0] if isinstance(key, tuple) else 1 << key
        n_bits_eff = key[0] if isinstance(key, tuple) else key
        if nb_override is not None:
            nb = nb_override
        elif key == self._n_mem_bits if not isinstance(key, tuple) else False:
            nb = self._ng_masks_cert
        else:
            nb = self._static_neighborhoods(n_bits_eff)
        mask_dicts: List[dict] = []
        for h in range(1, N + 1):
            base = int(nb[h]) | (1 << (h - 1))
            others = [i for i in range(N) if base >> i & 1 and i != h - 1][:n_bits_eff]
            mems = {1 << (h - 1): 0}
            for i in others:
                for m in list(mems):
                    mems.setdefault(m | (1 << i), len(mems))
            mask_dicts.append(mems)
        TRANS = np.full((N + 1, n_slot, N + 1), n_slot, dtype=np.int16)
        for h in range(1, N + 1):
            for mem, mi in mask_dicts[h - 1].items():
                if mi >= n_slot:
                    continue
                for k in range(1, N + 1):
                    if mem >> (k - 1) & 1:
                        continue
                    mem2 = (mem & int(nb[k])) | (1 << (k - 1))
                    TRANS[h, mi, k] = mask_dicts[k - 1].get(mem2, n_slot)
        mi_self = np.zeros(N + 1, dtype=np.int16)
        for h in range(1, N + 1):
            mi_self[h] = mask_dicts[h - 1][1 << (h - 1)]
        if isinstance(key, tuple):
            # 对偶感知版本仅保留最近一个, 防缓存膨胀
            self._dp_cache = {k: v for k, v in self._dp_cache.items()
                              if not isinstance(k, tuple)} if                 sum(isinstance(k, tuple) for k in self._dp_cache) > 1 else self._dp_cache
        self._dp_cache[key] = (TRANS, mi_self, n_slot)
        return self._dp_cache[key]

    def _build_ng4_tables(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """初始化默认认证位数 (ng-6) 的 DP 表并返回。"""
        self._n_slot = 1 << self._n_mem_bits
        return self._dp_tables(self._n_mem_bits)

    def cert_nb_from_duals(self, pi: np.ndarray, n_bits: int) -> np.ndarray:
        """对偶感知认证邻域: N(h) = {h} ∪ top(κ-1) 按 π_j − c_hj 降序。

        套利走线偏好重访"高对偶收益、低抵达成本"的客户; 将其显式放入
        N(h) 的记忆即阻断对应循环, 收紧 μ̂ (ng 超集论证不受邻域构造影响)。
        """
        N = self.N
        mid = self.costs.M[:, :, Q_STEPS // 2]
        nb = np.zeros(N + 1, dtype=object)
        kappa = max(3, n_bits)
        for h in range(1, N + 1):
            score = pi - mid[h, 1:]          # (N,) 套利磁铁得分
            order = np.argsort(-score)
            m = int(1) << (int(h) - 1)
            cnt = 1
            for j in order:
                if int(j) + 1 == h:
                    continue
                m |= int(1) << int(j)
                cnt += 1
                if cnt >= kappa:
                    break
            nb[h] = m
        return nb

    def exact_ng4_dp(self, pi: np.ndarray, sigma: float, bctx: BranchCtx,
                     n_bits: Optional[int] = None,
                     nb_override: Optional[np.ndarray] = None) -> Tuple[float, bool]:
        """ng 细网格松弛的精确最小检验数 μ̂ (穷尽 DP, 无预算赌注)。

        松弛链: 细网格 floor 载荷成本 ≤ 精确成本 (凸插值再取低点),
        ng-walk ⊇ 初等 ⇒ μ̂_dp ≤ min_{elem} rc。μ̂_dp ≥ −ε 即认证。
        [路线 2 | 能量不感知的超集松弛] 向量化后向 DP 不追踪电量维 (状态空间
        不可行), 其走线空间 ⊇ Ω_E ⇒ μ̂ 仍是 min_{Ω_E} rc 的合法下界 (Farley
        有效性不变), 代价是可能被能量不可行走线拖松。回收走线由调用方
        (_consume_dp_route) 以精确 route_cost 过能量门。
        n_bits: 邻域位数 (默认 ng-6; 认证冲刺可传 8)。
        返回 (mu_hat, certified_no_negative)。
        """
        N, G = self.N, self.G_FINE
        Mx = self._fine_cost_table()
        A = bctx.arc_mask_matrix()
        nb_bits = self._n_mem_bits if n_bits is None else int(n_bits)
        TRANS, mi_self, NS = self._dp_tables(nb_bits, nb_override)
        choice = np.zeros((G + 1, N + 1, NS), dtype=np.int16) if getattr(self, "recover_routes", False) else None
        step_g = self.Q / G
        dem_g = np.minimum((self.dem // step_g).astype(int), G)
        U = np.full((G + 1, N + 1, NS + 1), np.inf, dtype=np.float64)
        U[:, :, NS] = np.inf
        pad_pi = np.concatenate([[0.0], pi])
        ks = np.arange(1, N + 1)
        close_allow = np.array([bctx.close_ok(h) for h in range(N + 1)])
        for g in range(G, -1, -1):
            E = Mx[:, :, g]                                   # (N+1, N+1)
            close_v = np.where(close_allow, E[0, :] - sigma, np.inf)
            # cand[h, mi, k] = E[k,h,g] − π_k + U[g+d_k, k, TRANS[h,mi,k]]
            g2 = np.minimum(g + dem_g, G + 1)                 # 越界 → 无效
            base = E[1:, 1:] - pad_pi[1:, None]               # (k, h) = [k-1, h-1]
            valid_g = (g2[1:] <= G)[None, None, :]            # (1, 1, k)
            tgt_g = np.where(g2[1:] <= G, g2[1:], 0)
            trans = TRANS[1:, :, 1:]                          # (h, mi, k)
            valid_m = trans < NS                              # (h, mi, k)
            uu = np.where(valid_m & valid_g,
                          U[tgt_g[None, None, :],
                            ks[None, None, :],
                            np.where(trans < NS, trans, NS)],
                          np.inf)
            cand = base.T[:, None, :] + uu                    # (h, 1, k) + (h, mi, k)
            cand = np.where(A[1:, 1:].T[:, None, :], cand, np.inf)
            best = np.min(cand, axis=2)                       # (h, mi)
            if choice is None:
                U[g, 1:, :NS] = np.minimum(best, close_v[1:, None])
            else:
                k_star = np.argmin(cand, axis=2) + 1
                closed = close_v[1:, None] <= best
                U[g, 1:, :NS] = np.where(closed, close_v[1:, None], best)
                choice[g, 1:, :NS] = np.where(closed, np.int16(-1), k_star.astype(np.int16))
        # μ̂: 种子 [j, 0] + U[grid(d_j), j, {j}]
        e_j0 = Mx[np.arange(1, N + 1), 0, 0]
        gd = dem_g[1:]
        end_allow = np.array([bctx.end_ok(j) for j in range(1, N + 1)])
        seed_v = e_j0 - pi + U[gd, ks, mi_self[1:]]
        mu = float(np.min(np.where(end_allow, seed_v, np.inf)))
        self._last_recovery = None
        if choice is not None and np.isfinite(mu):
            self._last_recovery = self._walk_dp_route(
                choice, Mx, pi, sigma, TRANS, dem_g, mi_self, seed_v, end_allow, G, NS, mu)
        return mu, bool(mu >= -1e-3)

    def _walk_dp_route(self, choice, Mx, pi, sigma, TRANS, dem_g, mi_self,
                       seed_v, end_allow, G, NS, mu):
        """Replay the DP argmin. dp_rc must match mu before the route is usable."""
        masked = np.where(end_allow, seed_v, np.inf)
        if not np.isfinite(masked).any():
            return None
        j = int(np.argmin(masked)) + 1
        g, h, mi = int(dem_g[j]), j, int(mi_self[j])
        seq = [j]
        acc = float(Mx[j, 0, 0] - pi[j - 1])
        seen = {j}
        elementary = True
        for _ in range(self.N * 2 + 1):
            if not (0 <= g <= G) or mi < 0 or mi >= NS or h <= 0:
                return None
            pick = int(choice[g, h, mi])
            if pick < 0:
                acc += float(Mx[0, h, g] - sigma)
                route = [0] + list(reversed(seq)) + [0]
                return {"route": route, "elementary": elementary,
                        "dp_rc": acc, "mu": float(mu)}
            if pick == 0:
                return None
            if pick in seen:
                elementary = False
            acc += float(Mx[pick, h, g] - pi[pick - 1])
            mi = int(TRANS[h, mi, pick])
            g += int(dem_g[pick])
            h = pick
            seq.append(pick)
            seen.add(pick)
        return None

    def _fine_cost_table(self) -> np.ndarray:
        """把 21 档张量插值到细网格 (每城市缓存一次)。

        细档值 = 21 档线性插值。保界依据是载荷单调性 (ArcCosts 构造断言) 与
        floor 量化: 插值落在区间两端点之间, 而 DP 的 g 网格向下取整 ⇒ 取到的
        插值点 ≤ 该载荷处的精确成本; 凸性非必需 (实测满/空比 0.97–1.00 量级)。
        """
        if getattr(self, "_fine_M", None) is not None:
            return self._fine_M
        G = self.G_FINE
        base = self.costs.M if self.costs.dynamic else \
            np.repeat(self.costs.M[:, :, :1], Q_STEPS, axis=2)
        n0, n1, _ = base.shape
        fine = np.empty((n0, n1, G + 1))
        x_old = np.linspace(0.0, 1.0, Q_STEPS)
        x_new = np.linspace(0.0, 1.0, G + 1)
        flat = base.reshape(-1, Q_STEPS)
        fine = np.stack([np.interp(x_new, x_old, flat[i])
                         for i in range(flat.shape[0])], axis=0
                        ).reshape(n0, n1, G + 1)
        self._fine_M = np.ascontiguousarray(fine)
        return self._fine_M

    # ---------------- 逆向 ng-route A* 精确/启发式定价 ----------------
    def a_star(self, pi: np.ndarray, sigma: float, bctx: BranchCtx,
               U: np.ndarray, max_pops: int = 30000, time_limit: float = 5.0,
               max_cols: int = 40, ng_masks: Optional[np.ndarray] = None,
               stop_on_max_cols: bool = True,
               trace: Optional[dict] = None) -> Tuple[List[dict], bool, float]:
        """返回 (负检验数初等列, 是否认证无负列, mu_observed)。

        ng_masks: 覆盖邻域 (默认大邻域; 认证级调用可传小邻域)。
        stop_on_max_cols=False 时越过列收集阈值继续搜索直至穷尽/键非负
        (认证模式)。
        """
        N, Q = self.N, self.Q
        Mx, dem = self.costs.M, self.dem
        inv = self.costs._inv_step if self.costs.dynamic else None
        _stats_on = A_STAR_STATS_ON
        _t_stats = time.time() if _stats_on else 0.0
        if _stats_on:
            A_STAR_STATS["calls"] += 1
            _eps_grid = (1.0, 5.0, 20.0, 100.0)   # J: ~0.1%/0.5%/2%/10% 量级
            _win = {e: [] for e in _eps_grid}
        ng = self._ng_masks if ng_masks is None else ng_masks
        t0 = time.time()
        pq: list = []
        # [路线 2] 标签携带 E 维: (rc, mask, load, E) — E 为沿途回程向累计的
        # 真实弧成本 (= 能耗, 与 costs.cost 同一插值口径), E > E_MAX_J 即剪枝,
        # 闭合门校验整线 E ≤ E_MAX_J (能量单调增长 ⇒ 剪枝不丢可行完成)。
        # 支配保持三维 (能量维不进支配, 见扩展处注释: 换取定价吞吐)。
        for j in range(1, N + 1):
            if dem[j] <= Q and bctx.end_ok(j):
                rc0 = Mx[j, 0, 0] - pi[j - 1]
                gd = int(dem[j] * inv) if inv else 0
                gd = min(gd, Q_STEPS - 1)
                f = rc0 + U[j, gd]
                if f < -EPS_RC:
                    heapq.heappush(pq, (f, rc0, 1 << (j - 1), int(dem[j]), j,
                                        (j, 0), float(Mx[j, 0, 0])))
        pareto: Dict[Tuple[int, int], List[Tuple[float, int, int]]] = {}
        cols: List[dict] = []
        seen = set()
        min_found = 0.0
        certified = False
        pops = 0
        _stop_reason = None            # 'pops_cap' / 'time_cap' / 'exhausted' / 'key_nonneg'
        while pq:
            if pops >= max_pops or time.time() - t0 > time_limit:
                _stop_reason = 'pops_cap' if pops >= max_pops else 'time_cap'
                break
            f, rc, mask, load, j, path, e = heapq.heappop(pq)
            if trace is not None:
                visited = int(mask.bit_count()) if hasattr(mask, "bit_count") else bin(mask).count("1")
                trace["paths"].append(path)
                trace["features"].append((
                    len(path) / N, load / Q, rc / 1e3, f / 1e3, visited / N,
                    float(pi[j - 1]) / 1e3, dem[j] / Q, float(len(path) == 2),
                ))
            if f >= -EPS_RC:
                certified = True     # A* 可采纳 ⇒ 队列最小键非负 ⇒ 无更负完成
                _stop_reason = 'key_nonneg'
                break
            pops += 1
            if _stats_on:
                st = A_STAR_STATS
                st["pops"] += 1
                if _win is not None and len(pq) <= 4000 and pops <= 600:
                    for e in _eps_grid:
                        cnt = 0
                        for item in pq:
                            if item[0] <= f + e:
                                cnt += 1
                                if cnt >= 50:
                                    break
                        _win[e].append(min(cnt, 50))
                if len(pq) > 1:
                    t0_, t1_ = pq[0], pq[1]
                    if t0_[0] == t1_[0]:
                        st["exact_f_ties"] += 1
                        if t0_[1] == t1_[1]:
                            st["ties_broken_by_mask"] += 1
            # 闭合：弧 0→j 承载全程载荷
            if bctx.close_ok(j):
                full = rc + self.costs.cost(0, j, load) - sigma
                if full < min_found:
                    min_found = full     # 含能量不可行闭合: μ̂ 作超集下界仍有效
                # [路线 2] 电量硬约束: 闭合航线总能耗超限者不入列池
                if full < -EPS_RC and e + self.costs.cost(0, j, load) <= E_MAX_J + 1e-6:
                    seq = path[:-1]
                    if len(seq) == len(set(seq)):    # 初等过滤（Set-Partition 语义）
                        route = [0] + list(path)
                        key = tuple(route)
                        if key not in seen:
                            seen.add(key)
                            cols.append(make_col(self.costs, route, pi, sigma))
                            if stop_on_max_cols and len(cols) >= max_cols:
                                _stop_reason = 'max_cols'
                                break
            # 前驱扩展 h → j（弧载荷 = 当前后缀载荷 load）
            x = load * inv if inv else 0.0
            i0 = int(x)
            if i0 > Q_STEPS - 2:
                i0, fr = Q_STEPS - 2, 1.0
            else:
                fr = x - i0
            if self.costs.dynamic:
                e_vec = Mx[1:, j, i0] + fr * (Mx[1:, j, i0 + 1] - Mx[1:, j, i0])
            else:
                e_vec = Mx[1:, j, 0]
            rc_vec = rc + e_vec - pi
            en_vec = e + e_vec            # [路线 2] 途中能耗向量 (E + 弧成本)
            second = path[1] if len(path) > 1 else -1
            gd_vec = np.minimum(((load + dem) * inv).astype(int) if inv
                                else np.zeros(N + 1, dtype=int), Q_STEPS - 1)
            for h in range(1, N + 1):
                hb = 1 << (h - 1)
                if mask & hb or h == second:
                    continue
                load2 = load + int(dem[h])
                if load2 > Q:
                    continue
                if not bctx.arc_ok(h, j):
                    continue
                rc2 = rc_vec[h - 1]
                if rc2 >= -EPS_RC:
                    continue
                e2 = en_vec[h - 1]        # [路线 2] 电量剪枝: 能量单调增长
                if e2 > E_MAX_J:
                    continue
                if _stats_on:
                    A_STAR_STATS["pruned_rc"] += 1
                g2 = int(gd_vec[h])
                f2 = rc2 + U[h, g2]
                if f2 >= -EPS_RC:
                    if _stats_on:
                        A_STAR_STATS["pruned_f"] += 1
                    continue
                mask2 = (mask & ng[h]) | hb
                bucket = pareto.setdefault((h, g2), [])
                # [路线 2] 支配保持三维 (rc, mask⊆, load): A* 在生产串中仅为
                # 列生成器 (证书链全在 DP μ̂, astar_bound=False 不取 A* 穷尽界),
                # 能量维不进支配换取吞吐; 超限标签由 E 剪枝与闭合门拦截,
                # 桶内支配虽可能漏掉个别能量可行列 (仅完整性, 非有效性),
                # 漏列由 DP 认证路径兜底。四维支配实测 -35% 迭代吞吐, 弃。
                dominated = False
                for (prc, pmask, pload) in bucket:
                    if prc <= rc2 and pmask & ~mask2 == 0 and pload <= load2:
                        dominated = True
                        break
                if dominated:
                    if _stats_on:
                        A_STAR_STATS["pruned_dominated"] += 1
                    continue
                bucket[:] = [p for p in bucket
                             if not (rc2 <= p[0] and mask2 & ~p[1] == 0
                                     and load2 <= p[2])]
                bucket.append((rc2, mask2, load2))
                heapq.heappush(pq, (f2, rc2, mask2, load2, h, (h,) + path, e2))
                if _stats_on:
                    A_STAR_STATS["pushes"] += 1
        if not pq:
            certified = True
        if _stop_reason is None:
            _stop_reason = 'exhausted' if not pq else 'unknown'
        mu_hat = min(0.0, min_found) if certified else \
            min(min_found, pq[0][0] if pq else 0.0)
        if trace is not None and trace["paths"]:
            accepted = [tuple(col["route"][1:]) for col in cols]
            x, y1, y2 = mark_ancestor_labels(trace["paths"], trace["features"], accepted)
            trace["X"], trace["y1"], trace["y2"] = x, y1, y2
        cols.sort(key=lambda c: c["rc"])
        if _stats_on:
            st = A_STAR_STATS
            st["elapsed_s"] += time.time() - _t_stats
            st["cols_emitted"] += len(cols)
            st["stop_reasons"].append((st.get("call_tier") or "unknown",
                                       _stop_reason, pops))
            if st.get("call_tier") == "tier1":
                st["final_cols"].append(
                    ([(round(c["rc"], 3), c["route"]) for c in cols]))
                st["pi_snapshot"].append((pi.copy(), float(sigma)))
            for e, arr in _win.items():
                st["window_counts"].append((e, arr))
        return cols[:max_cols], certified, mu_hat


# ==============================================================================
# 5. 原问题侧：轻量局部搜索（Intra-2opt + Inter-Relocate）与 FFD 装载修复
# ==============================================================================
def _two_opt(costs: ArcCosts, route: List[int]) -> List[int]:
    best_r, best_c = list(route), costs.route_cost(route)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_r) - 2):
            for j in range(i + 1, len(best_r) - 1):
                cand = best_r[:i] + best_r[i:j + 1][::-1] + best_r[j + 1:]
                c = costs.route_cost(cand)
                if c < best_c - 1e-3:
                    best_r, best_c, improved = cand, c, True
                    break
            if improved:
                break
    return best_r


def fleet_local_search(costs: ArcCosts, routes: List[List[int]], Q: int,
                       rounds: int = 8) -> List[List[int]]:
    """Intra-2opt + Inter-Relocate（单点与 Or-2/Or-3 段, 容量可行）。

    严格保持互斥划分；段可反转（Or-opt 标准邻域）。
    """
    routes = [r for r in routes if len(r) > 2]
    for _ in range(rounds):
        improved = False
        for idx in range(len(routes)):
            opt_r = _two_opt(costs, routes[idx])
            if costs.route_cost(opt_r) < costs.route_cost(routes[idx]) - 1e-3:
                routes[idx] = opt_r
                improved = True
        if improved:
            continue
        loads = [sum(costs.demands_g[u] for u in r[1:-1]) for r in routes]
        done = False
        for a in range(len(routes)):
            if len(routes[a]) <= 3 or done:
                continue
            segs = []
            for i in range(1, len(routes[a]) - 1):
                segs.append((i, 1))
                if i + 2 < len(routes[a]):
                    segs.append((i, 2))
                if i + 3 < len(routes[a]):
                    segs.append((i, 3))
            for (i, L) in segs:
                if done:
                    break
                seg = routes[a][i:i + L]
                qseg = sum(int(costs.demands_g[u]) for u in seg)
                for b in range(len(routes)):
                    if a == b or done:
                        continue
                    if loads[b] + qseg > Q:
                        continue
                    base = costs.route_cost(routes[a]) + costs.route_cost(routes[b])
                    ra = routes[a][:i] + routes[a][i + L:]
                    best = None
                    for p in range(1, len(routes[b])):
                        for s in ([seg] if L == 1 else [seg, seg[::-1]]):
                            rb = routes[b][:p] + s + routes[b][p:]
                            crb = costs.route_cost(rb)
                            # 电量硬约束: 目标航线不得超限 (删段后的源航线 ra
                            # 不作前置校验——删段未必降耗(张量无三角不等式),
                            # 其整线可行性由 _try_update_ub 出口断言兜底)
                            if crb > E_MAX_J + 1e-6:
                                continue
                            d = costs.route_cost(ra) + crb
                            if d < base - 1e-3 and (best is None or d < best[0]):
                                best = (d, rb)
                    if best is not None:
                        routes[a], routes[b] = ra, best[1]
                        loads[a] -= qseg
                        loads[b] += qseg
                        improved = done = True
                        break
            if done:
                break
        if not improved:
            break
    return [r for r in routes if len(r) > 2]


def greedy_fleet(costs: ArcCosts, K: int, Q: int, rng: random.Random
                 ) -> List[List[int]]:
    """最近邻贪心 + FFD 兜底装载，保证 ≤K 条互斥覆盖航线。

    电量可行性不在本函数内把关——种子航线的能量门由调用方
    (BPEngine.solve 的种子准入门) 统一校验。"""
    N = costs.N
    dem = costs.demands_g
    mid = costs.M[:, :, Q_STEPS // 2]   # 中载弧成本作为邻近度
    unserved = set(range(1, N + 1))
    routes: List[List[int]] = []
    while unserved:
        route, load, last = [0], 0, 0
        while True:
            best, best_c = None, None
            for c in unserved:
                if load + dem[c] <= Q:
                    # 电量硬约束: 候选并入后整条航线能耗不得超限
                    if costs.route_cost(route + [c, 0]) > E_MAX_J + 1e-6:
                        continue
                    cc = mid[last, c]
                    if best_c is None or cc < best_c:
                        best, best_c = c, cc
            if best is None:
                break
            route.append(best)
            load += dem[best]
            unserved.remove(best)
            last = best
        route.append(0)
        routes.append(route)
        if len(routes) > K:    # 超机队 ⇒ FFD 重装
            items = sorted((u for r in routes for u in r[1:-1]),
                           key=lambda u: -dem[u])
            bins: List[List[int]] = [[] for _ in range(K)]
            loads = [0] * K
            ok = True
            for u in items:
                placed = False
                for b in range(K):
                    if loads[b] + dem[u] <= Q:
                        bins[b].append(u)
                        loads[b] += dem[u]
                        placed = True
                        break
                if not placed:
                    ok = False
                    break
            if not ok:
                raise RuntimeError("FFD 装载失败: 实例需求超过机队容量")
            routes = [_two_opt(costs, [0] + b + [0]) for b in bins if b]
            break
    if rng is not None:
        rng.shuffle(routes)
    return routes


def assert_partition(routes: List[List[int]], N: int, K: int, Q: int,
                     demands_g: np.ndarray,
                     costs: Optional["ArcCosts"] = None) -> None:
    served = [u for r in routes for u in r[1:-1]]
    assert len(routes) <= K, f"航线数 {len(routes)} 超机队 {K}"
    assert len(served) == N and len(set(served)) == N, \
        f"划分破坏: {len(served)}/{N} 服务, 去重 {len(set(served))}"
    for r in routes:
        assert r[0] == 0 and r[-1] == 0
        load = sum(int(demands_g[u]) for u in r[1:-1])
        assert load <= Q, f"航线 {r} 载荷 {load}g 超容量 {Q}g"
    if costs is not None:
        # 电量硬约束: 逐航线总能耗 (=route_cost) 不得超过 E_MAX_J
        for r in routes:
            e = costs.route_cost(r)
            assert e <= E_MAX_J + 1e-6, \
                f"航线 {r} 能耗 {e/1e3:.1f} kJ 超电量 {E_MAX_J/1e3:.0f} kJ"


# ==============================================================================
# 6. Branch-and-Price 主引擎
# ==============================================================================
class BPEngine:
    def _compute_ap_elb(self) -> float:
        """定理 B: 基于 depot x K 复制的指派松弛下界 AP-ELB (耗时 <20ms, 永不为负的安全地板)"""
        try:
            T = self.costs.M
            n = T.shape[0]
            N = n - 1
            K = self.K
            dem = self.demands_g.astype(float)
            Q_G = 1019.0
            Q_STEPS = int(T.shape[2])
            inv = (Q_STEPS - 1) / Q_G
            C = np.empty((n, n))
            for v in range(n):
                i0 = int(min(Q_STEPS - 2, int(dem[v] * inv)))
                fr = dem[v] * inv - i0
                C[:, v] = T[:, v, i0] + fr * (T[:, v, i0 + 1] - T[:, v, i0])
            np.fill_diagonal(C, 1e12)
            C[0, 0] = 1e12
            Msz = N + K
            BIG = 1e12
            A = np.full((Msz, Msz), BIG)
            for a_i in range(K):
                for v in range(N):
                    A[a_i, K + v] = C[0, v + 1]
                for b_i in range(K):
                    A[a_i, b_i] = BIG
            for u in range(N):
                for v in range(N):
                    if u != v:
                        A[K + u, K + v] = C[u + 1, v + 1]
                for b_i in range(K):
                    A[K + u, b_i] = C[u + 1, 0]
            ri, ci = linear_sum_assignment(A)
            psi = float(A[ri, ci].sum())
            dq = Q_G / (Q_STEPS - 1)
            slope = np.diff(T, axis=2) / dq
            off = ~np.eye(n, dtype=bool)
            lam_min = float(slope[off].min())
            d = np.sort(dem[1:])[::-1]
            Lam = 0.0
            for t in range(N):
                Lam += (t // K + 1) * d[t]
            Sd = float(dem[1:].sum())
            return psi + lam_min * max(0.0, Lam - Sd)
        except Exception as exc:
            return -math.inf

    def __init__(self, tensor: np.ndarray, customers: List[dict], K: int = 5,
                 dynamic_payload: bool = True, stabilize: bool = True,
                 time_budget_s: float = 300.0, ng_size: int = 10,
                 ng_endgame: int = 8,
                 root_share: float = 0.77,
                 dual_box: bool = False,
                 dp_time_share: float = 0.0,
                 lns_time_share: float = 0.0,
                 dp_mode: str = "legacy",
                 det_iters: int = 0,
                 tier_pop_bounded: bool = False,
                 cod_refine: bool = False,
                 larr: bool = False,
                 astar_bound: bool = False,
                 t2b_max_cols: int = 40,
                 dp_grid: int = 126,
                 dual_polish: bool = False,
                 larr_cert: bool = False,
                 cert_grid: int = 0,
                 cert_max_calls: int = 0,
                 record_sites: bool = False,
                 seed: int = 42, verbose: bool = True,
                 enable_ap_elb: bool = False, enable_throttle: bool = True,
                 enable_exp3: bool = True, enable_two_pass: bool = False):
        self.K = K
        self.Q = int(round(Q_MAX_KG * 1000))
        self.N = len(customers) - 1
        self.demands_g = np.zeros(self.N + 1, dtype=np.int64)
        for c in customers:
            self.demands_g[c["id"]] = int(round(c.get("demand_kg", 0.0) * 1000.0))
        self.costs = ArcCosts(tensor, self.demands_g, dynamic_payload)
        self.pricer = Pricer(self.costs, K, ng_size, dynamic_payload)
        self.ng_endgame = int(ng_endgame)   # 末期 ng 精算邻域位数 (P0 探针: 8/10/12)
        self.root_share = float(root_share)   # 根节点时间份额 (默认 0.77, 零行为变化)
        self.dual_box = bool(dual_box)        # [组件一] 间隙自收缩对偶盒 (默认关; 推导见项目纪要)
        # [重工程地基] 精算 DP 时间份额上限 (0=关, 逐位不变): 把"DP 调频"从
        # 墙钟隐式调度变为节点内显式预算, 消除 SG120 调度彩票的混淆项。
        self.dp_time_share = float(dp_time_share)
        # [UB 组件] LNS 时间份额上限 (0=关, 逐位不变): 0.0 走旧硬帽
        # (min(60, remain-36)); >0 时为根节点剩余墙钟的份额, 把已测为零的死时间
        # (树阶段 + 根尾段) 释放给唯一活着的 UB 通道。实测饱和点 ~400s。
        self.lns_time_share = float(lns_time_share)
        # [LB 组件 DP-F] 精算预言机选择: "legacy" = 向量化后向满网格 DP (默认, 逐位不变);
        # "frontier" = 前向 Pareto 支配标号 (numba)。二者 mu_hat 逐位等价 (实测 dev~1e-11 J),
        # 加速比随 ng 增长 (ng-9 2.03x / ng-10 2.66x / ng-12 4.77x @ SH120 存档对偶)。
        self.dp_mode = str(dp_mode)
        # [UB 组件 LARR: 载荷感知航线重排] 默认关 = 零行为变化。载荷依赖弧成本使
        # 访问顺序成为一阶决策; 2-opt 是置换局部搜索, 找不到载荷序全局最优。
        # 对给定客户集用子集 DP 求精确最优序 (m<=larr_max), 可证不劣 (候选含原序)。
        self.larr = bool(larr)
        self.larr_max = 16
        self.larr_calls = 0
        self.larr_gain = 0.0
        # [LARR-cert] 终局后置重排内核: 搜索结束后对最终 incumbent 做一次保序重排。
        # 搜索期代码路径与 larr=False 逐位相同 (零轨迹扰动) ⇒ UB 单调不增、
        # LB_tree 逐位不变 ⇒ gap 可证单调不增 (后置重排命题1)。
        # 与搜索内 larr 幂等 (同 argmin), 采纳格叠加运行时 δ=0, 无副作用。
        self.larr_cert = bool(larr_cert)
        self.larr_cert_gain = 0.0
        # [GF-dec] 证书-定价网格解耦: cert_grid>0 时定价网格 (pricer.G_FINE) 保持
        # dp_grid 不变, 仅 Farley 认证 μ̂ 调用位点 (mu8 末期窗口 / Tier-2a) 临时升档。
        # 合法性: μ̂ 只需 ≤ 真值 (网格保守性引理), 与对偶来源无关; 粗网格定价产出的
        # 列仍是可行列, RMP 界不依赖列的生成方式 ⇒ 证书与定价可独立取档。
        self.cert_grid = int(cert_grid)
        self.cert_calls = 0
        # [GF-dec 生产限流] 0 = 无限制 (det 仪器口径, 与价值地图腿逐位一致);
        # >0 = 生产形态: 仅根节点 + 末期窗口(<120s) + 全局次数帽 (见 _cert_ok)。
        self.cert_max_calls = int(cert_max_calls)
        # [后置解耦 v2] 位点录制开关 (默认关 = 零行为变化)。开启时在根节点
        # Farley 更新位点被动追加 (z, π, σ, remain) 快照 —— 只 append, 不读时钟、
        # 不碰 RNG/列池/对偶, 对搜索轨迹零扰动。录制流供 solve() 结束后的
        # certify_post() 重放认证 (组件 A 后置形态) 与 mono_kernel() (组件 B
        # 后置形态) 使用: 四臂消融共享同一条基础轨迹, 排序按构造成立。
        self.record_sites = bool(record_sites)
        self.site_log: List[dict] = []
        self._root_bctx = None
        self.cert_post_lb = -math.inf
        self.cert_post_gain = 0.0
        self.cert_post_calls = 0
        self.kernel_gain = 0.0
        # [LB 组件 COD: 认证预言机双职定价] 默认关 = 零行为变化。需配合
        # insert_dp_routes/measure_dp_routes 打开 (frontier 路径回收机)。
        self.cod_refine = bool(cod_refine)
        # [AEB] A* 穷尽真值成本证书界组合: DP 需求下取整 ⇒ 载荷/成本低估 ⇒ μ̂ 保守 ~2 kJ;
        # tier2b A* 穷尽 (certified) 时 mu_A* 有效且更紧, 与 mu4 取 max (合法组合算术)。
        self.astar_bound = bool(astar_bound)
        # [列吞吐] tier2b 穷尽搜索但交回被截断到 max_cols (a_star return cols[:max_cols]);
        # SG120 det_200 实测单一对偶态存在 20529 条负初等列, 引擎每次仅收 40 条。
        self.t2b_max_cols = int(t2b_max_cols)
        # [Exp-D2] DP 证书网格分辨率: 默认 126 (零行为变化)。加密到 1020 使 μ̂ 捕获 97.5%
        # 的保守 slack (实测 -12.8286 vs A\* 穷尽真值 -12.7797, 7.6s/次; 机理=需求 floor
        # ×载荷-成本斜率, Exp-D0 逐弧闭合)。运行时膨胀温和 (expansions 被 cap 钉住)。
        self.dp_grid = int(dp_grid)
        if self.dp_grid != 126:
            self.pricer.G_FINE = self.dp_grid
            self.pricer._fine_M = None
        # [LB 组件 DFP: 对偶面证书抛光] 默认关 = 零行为变化。引擎交付的 RMP 最优对偶
        # ≠ 认证最优对偶 (对偶面退化): P2 探针实测同一最优面上 L1 最近投影点把
        # μ̂ 从 -16.73 收紧到 -7.17 (Farley +105.25 kJ, 超预注册门 21 倍)。机制:
        # 节点 CG 出口对终末 (池, z, π, σ) 建最优面 LP → L1 最近投影候选 → 穷尽 DP
        # 复认证 → 棘轮只收认证值。证书环节隔离: 不碰交付定价对偶/列池/UB 管线
        # (det UB 逐位不变是组件死刑门)。合法性: ACB 任意对偶有效 (白皮书定理 4.3)。
        # 见对偶面证书抛光推导（项目纪要）。
        self.dual_polish = bool(dual_polish)
        self.polish_calls = 0
        self.polish_gain = 0.0
        # [确定性度量模式] det_iters>0 时根 CG 跑恰好 det_iters 个迭代, 所有控制决策
        # (late_phase/DP 计划/DP 预算/a_star 时限) 改为迭代序号驱动, 使 LB 成为迭代数
        # 的确定函数 (消除时间自适应引擎的调度彩票; 默认 0 = 零行为变化)。
        self.det_iters = int(det_iters)
        self._it_local = 0
        self._dp_calls_node = 0
        # [Z 组件: 工作预算制 tier 调度] 把 a_star 两层的时限从绝对秒数改为仅 pop 帽。
        # 动机(G5): 时间预算 tier 会吸收其他组件释放的墙钟(条件触发时更频繁),
        # 使固定墙钟内迭代数不增。改成 pop 帽后释放的墙钟真正变成迭代(默认关)。
        self.tier_pop_bounded = bool(tier_pop_bounded)
        self.stabilize = stabilize
        self.T_MAX = time_budget_s
        self.verbose = verbose
        self.rng = random.Random(seed)
        # 运行态
        self.ub = math.inf
        self.ub_routes: List[List[int]] = []
        self.nodes_total = 0
        self.nodes_certified = 0
        self.nodes_pruned = 0
        self.nodes_integral = 0
        self.columns_total = 0
        self.cg_iterations = 0
        self.farley_history: List[float] = []
        self.root_farley = -math.inf
        self.enable_ap_elb = enable_ap_elb
        self.enable_throttle = enable_throttle
        self.enable_exp3 = enable_exp3
        self.enable_two_pass = enable_two_pass
        self.two_pass_early_s = 60.0
        self.two_pass_early_iters = 0
        self.trace_labels = False
        self.label_traces = []
        self.collect_dp = False
        self.collect_kicks = False
        self.dp_rows = []
        self.kick_rows = []
        self.measure_dp_routes = False
        self.insert_dp_routes = False
        self.best_relocate = False
        self.route_stats = {"negative": 0, "recovered": 0, "value_ok": 0,
                            "elementary": 0, "poolable": 0}
        self._pi_prev = None
        self._z_prev = None
        self.exp3_arms = (0.25, 0.5, 0.75, 0.9)
        self._exp3_w = np.ones(len(self.exp3_arms))
        self._exp3_last_arm = None
        self._exp3_reward_updates = 0
        self.node_time_caps: List[float] = []
        self.lb_ap = self._compute_ap_elb() if enable_ap_elb else -math.inf
        if enable_ap_elb:
            self.root_farley = max(self.root_farley, self.lb_ap)
        self._last_mu4 = None
        self.cuts: List[Tuple[int, int]] = []       # 割池保持为空；圆整容量割不再分离
        self.n_cuts_added = 0
        self._cert_nb = None
        self.sris: List[int] = []              # 全局 SRI-3 割池
        self.sri_keys: set = set()

    # ---------------- 日志 ----------------
    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    # ---------------- UB 更新 ----------------
    def _larr_route(self, body: List[int]) -> List[int]:
        """载荷感知精确重排 (LARR): 子集 DP 求给定客户集的最优访问顺序。

        成本约定同 ArcCosts.route_cost: 弧 u->v 的载荷 = 未服务客户需求之和
        (含 v 包裹, pre-release)。m>larr_max 时原序返回 (保持不劣)。
        """
        m = len(body)
        if m <= 2 or m > self.larr_max:
            return [0] + list(body) + [0]
        cost, dem = self.costs.cost, self.demands_g
        tot = sum(int(dem[u]) for u in body)
        full = (1 << m) - 1
        INF = math.inf
        dp = [[INF] * m for _ in range(1 << m)]
        par = [[-1] * m for _ in range(1 << m)]
        for i in range(m):
            dp[1 << i][i] = cost(0, body[i], tot)
        for mask in range(1, full + 1):
            load = sum(int(dem[body[i]]) for i in range(m) if not (mask >> i) & 1)
            row = dp[mask]
            for i in range(m):
                base = row[i]
                if base == INF:
                    continue
                ci = body[i]
                for j in range(m):
                    if (mask >> j) & 1:
                        continue
                    nm = mask | (1 << j)
                    v = base + cost(ci, body[j], load)
                    if v < dp[nm][j]:
                        dp[nm][j] = v
                        par[nm][j] = i
        best, bi = INF, -1
        for i in range(m):
            v = dp[full][i] + cost(body[i], 0, 0)
            if v < best:
                best, bi = v, i
        seq, mask, i = [], full, bi
        while i >= 0:
            seq.append(body[i])
            p = par[mask][i]
            mask ^= (1 << i)
            i = p
        seq.reverse()
        return [0] + seq + [0]

    def _larr_routes(self, routes: List[List[int]]) -> List[List[int]]:
        out, gain = [], 0.0
        for r in routes:
            body = [u for u in r if u != 0]
            nr = self._larr_route(body)
            gain += self.costs.route_cost(list(r)) - self.costs.route_cost(nr)
            out.append(nr)
        self.larr_calls += 1
        self.larr_gain += gain
        return out

    def _try_update_ub(self, routes: List[List[int]]):
        assert_partition(routes, self.N, self.K, self.Q, self.demands_g,
                         costs=self.costs)
        if self.larr:
            routes = self._larr_routes(routes)
        c = sum(self.costs.route_cost(r) for r in routes)
        if c < self.ub - 1e-6:
            self.ub = c
            self.ub_routes = [list(r) for r in routes]
            self._log(f"    [UB] 更新至 {c/1e3:.2f} kJ ({len(routes)} 航线)")
        assert self.ub >= self.root_farley - 1e-3, "UB 击穿下界: 证书链断裂!"

    def _round_heuristic(self, cols: List[dict], lam: np.ndarray):
        """λ 贪心取整 + 未覆盖 FFD 修复 + 局部搜索。"""
        order = np.argsort(-lam)
        chosen: List[List[int]] = []
        covered = set()
        for j in order:
            if lam[j] < 1e-6:
                break
            r = cols[j]["route"]
            s = set(r[1:-1])
            if s & covered or len(chosen) >= self.K:
                continue
            chosen.append(list(r))
            covered |= s
        leftover = [u for u in range(1, self.N + 1) if u not in covered]
        # 剩余客户贪心插入或开新航线
        for u in leftover:
            qu = int(self.demands_g[u])
            best, best_d = None, math.inf
            for r in chosen:
                load = sum(int(self.demands_g[x]) for x in r[1:-1])
                if load + qu > self.Q:
                    continue
                for p in range(1, len(r)):
                    cand = r[:p] + [u] + r[p:]
                    cc = self.costs.route_cost(cand)
                    # 电量硬约束: 插入后航线能耗不得超限
                    if cc <= E_MAX_J + 1e-6 and cc - self.costs.route_cost(r) < best_d:
                        best, best_d = (r, cand), cc - self.costs.route_cost(r)
            if best is not None:
                best[0][:] = best[1]
            else:
                if len(chosen) >= self.K:
                    return None     # 修复失败（不应发生：有可行种子在池）
                chosen.append([0, u, 0])
        refined = fleet_local_search(self.costs, chosen, self.Q)
        try:
            assert_partition(refined, self.N, self.K, self.Q, self.demands_g,
                             costs=self.costs)
        except AssertionError:
            return None
        self._try_update_ub(refined)

    def _pool_ip(self, cols: List[dict], time_cap: float = 4.0):
        """列池 0-1 集合分划 IP（原问题侧原始启发，OR-Tools SCIP）。"""
        try:
            from ortools.linear_solver import pywraplp
        except ImportError:
            return
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if solver is None:
            return
        n = len(cols)
        if n > 4000:
            cols = sorted(cols, key=lambda c: c["rc"])[:4000]
            n = len(cols)
        lam = [solver.BoolVar(f"l{j}") for j in range(n)]
        for i in range(self.N):
            ct = solver.Constraint(1.0, 1.0, f"cov{i}")
            for j, col in enumerate(cols):
                if col["a"][i] > 0.5:
                    ct.SetCoefficient(lam[j], 1.0)
        ct = solver.Constraint(0.0, float(self.K), "fleet")
        for j in range(n):
            ct.SetCoefficient(lam[j], 1.0)
        obj = solver.Objective()
        for j, col in enumerate(cols):
            obj.SetCoefficient(lam[j], float(col["cost"]))
        obj.SetMinimization()
        solver.set_time_limit(int(time_cap * 1000))
        st = solver.Solve()
        if st not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return
        routes = [list(cols[j]["route"]) for j in range(n)
                  if lam[j].solution_value() > 0.5]
        try:
            assert_partition(routes, self.N, self.K, self.Q, self.demands_g,
                             costs=self.costs)
        except AssertionError:
            return
        refined = fleet_local_search(self.costs, routes, self.Q)
        self._try_update_ub(refined)

    def _exact_mu(self, pi, sigma, bctx, n_bits=None, nb_override=None):
        """[DP-F] 精算 μ̂ 预言机分发: legacy 后向满网格 / frontier 前向支配标号.

        二者在真实实例上逐位等价 (dev ~ 1e-11 J, SH60 ng-8 / SH120 ng-9/10/12 均验过),
        故切换不破证书链。返回 (mu_hat, certified)。
        """
        if self.dp_mode != "frontier":
            return self.pricer.exact_ng4_dp(pi, sigma, bctx, n_bits=n_bits,
                                             nb_override=nb_override)
        n_eff = self.pricer._n_mem_bits if n_bits is None else int(n_bits)
        do_rec = bool(self.measure_dp_routes or self.insert_dp_routes)
        if do_rec and getattr(self, "cod_refine", False) and getattr(self, "_late_phase", True) \
                and getattr(self, "_cod_window", True):
            mu, _info = frontier_mu_cod(self, pi, sigma, bctx, n_eff, cap=200)
        else:
            mu, _stats = frontier_mu_numba(self, pi, sigma, bctx, n_eff, cap=200,
                                           recover=do_rec)
            if not do_rec:
                self.pricer._last_recovery = None
        return float(mu), bool(mu >= -1e-3)

    def _cert_mu(self, pi, sigma, bctx, n_bits=None, nb_override=None):
        """[GF-dec] 证书专用网格 μ̂: 仅本调用内把 pricer 网格换到 cert_grid。

        μ̂ parity: cert_grid==当前网格时逐位走 _exact_mu 原路径; 换档调用与
        直接设 pricer.G_FINE=cert_grid 的耦合形态走完全相同的 DP 代码
        (legacy 分支 exact_ng4_dp 读 G_FINE), 故同一对偶 ⇒ 同一 μ̂ (逐位)。
        _fine_M 按网格分键缓存, 避免每次换档重建插值矩阵。恢复用 try/finally,
        异常时不把细网格状态泄漏给定价。
        """
        if not self.cert_grid or self.cert_grid == self.pricer.G_FINE:
            return self._exact_mu(pi, sigma, bctx, n_bits, nb_override)
        p = self.pricer
        if not hasattr(p, "_fine_M_by_G"):
            p._fine_M_by_G = {}
        g0, m0 = p.G_FINE, getattr(p, "_fine_M", None)
        p.G_FINE = self.cert_grid
        p._fine_M = p._fine_M_by_G.get(self.cert_grid)
        self.cert_calls += 1
        try:
            out = self._exact_mu(pi, sigma, bctx, n_bits, nb_override)
        finally:
            if getattr(p, "_fine_M", None) is not None:
                p._fine_M_by_G[self.cert_grid] = p._fine_M
            p.G_FINE, p._fine_M = g0, m0
        return out

    def _cert_ok(self, is_root: bool, t_deadline: float) -> bool:
        """[GF-dec 生产限流] 细网格认证触发门。

        det 仪器 (det_iters>0): 无窗无帽 —— 与 6 格价值地图腿逐位一致。
        生产: 仅根节点 (子节点 18s 帽扛不住 28-90s 细 DP; 根 Farley 经祖先
        链继承惠及全树证书) + 根末期窗口 <120s + cert_max_calls 全局帽。
        生产实测依据: det 腿细认证墙钟膨胀 ~3x, 不限流将重演墙钟通道的
        迭代坍缩 (耦合形态同源病灶)。
        """
        if self.cert_grid <= 0:
            return False
        if self.cert_max_calls <= 0 or self.det_iters > 0:
            return True
        return (is_root and self._det_remain(t_deadline) < 120.0
                and self.cert_calls < self.cert_max_calls)

    def _dual_face_polish(self, cols, z_rmp, pi_del, sigma_del, bctx):
        """[DFP] 对偶面证书抛光: 最优面 L1 最近投影候选 + 穷尽 DP 复认证。

        诊断 (P2 探针, SG120 det_200 冻结池): 引擎交付的 RMP 最优对偶在认证意义上
        不紧 —— 同一最优面上存在 L1 最近投影点把 μ̂ 从 -16.73 收紧到 -7.17
        (Farley +105.25 kJ, 超预注册门 +5 的 21 倍)。

        合法性: ACB 在任意 (π, σ≤0) 有效 (白皮书定理 4.3); μ̂ 只出自穷尽 DP
        (cert_nb_from_duals 认证邻域, 与引擎末期认证同轨); LB_cand 的 D 从候选
        向量直接重算 (不沿用 z_RMP)。证书环节隔离: 候选对偶不交付定价、不碰列池,
        只进 Farley 棘轮。返回 (lb_cand [J] or None, info)。
        """
        from scipy.optimize import linprog
        N = self.N
        n_pool = len(cols)
        if n_pool == 0 or z_rmp is None:
            return None, {}
        C = np.array([col["cost"] for col in cols])
        A = np.array([col["a"] for col in cols], dtype=np.float64).T  # N × n_pool
        # 变量: [π (N, free), σ (1, ≤0), t (N, ≥0)]; L1 最近: min Σt, -t ≤ π-π_del ≤ t
        nv = N + 1 + N
        c_obj = np.zeros(nv)
        c_obj[N + 1:] = 1.0
        A_ub = np.zeros((2 * N + n_pool, nv))
        b_ub = np.zeros(2 * N + n_pool)
        for i in range(N):
            A_ub[i, i] = 1.0
            A_ub[i, N + 1 + i] = -1.0
            b_ub[i] = float(pi_del[i])
            A_ub[N + i, i] = -1.0
            A_ub[N + i, N + 1 + i] = -1.0
            b_ub[N + i] = -float(pi_del[i])
        for j in range(n_pool):
            A_ub[2 * N + j, :N] = A[:, j]
            A_ub[2 * N + j, N] = 1.0
            b_ub[2 * N + j] = float(C[j])
        A_eq = np.zeros((1, nv))
        A_eq[0, :N] = 1.0
        A_eq[0, N] = float(self.K)
        b_eq = [float(z_rmp)]
        bounds = [(None, None)] * N + [(None, 0.0)] + [(0, None)] * N
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if not res.success:
            return None, {"lp": "failed", "status": res.status}
        pi_c = res.x[:N]
        sig_c = float(res.x[N])
        D = float(pi_c.sum() + self.K * sig_c)
        _t = time.time()
        mu_c, _ = self._exact_mu(pi_c, sig_c, bctx, n_bits=self.ng_endgame)
        self._dp_time_node += time.time() - _t
        lb_cand = D + self.K * min(0.0, float(mu_c))
        return float(lb_cand), {"D": D, "mu": float(mu_c),
                                "dev": abs(D - z_rmp), "lp_cols": n_pool}

    def _dp_allowed(self, t_deadline: float, t_node0: float) -> bool:
        """[重工程地基] 精算 DP 时间份额门。dp_time_share=0 时恒 True (逐位不变)。

        [解耦 2026-09-28] 预算基准取 T_MAX (总量) 而非节点墙钟 (t_deadline - t_node0):
        原式 = root_share × T_MAX, 导致调 root_share 会同步收缩 DP 预算
        (0.77->0.55 时 DP 预算 -29%) —— 这与 SG120 已归档的"调度彩票"同源
        (晚相窗口按 deadline 等比缩放)。改锚 T_MAX 后, DP 预算与 root_share 解耦。
        注: 本门只管"是否启动一次 DP 调用", 单次调用仍受节点 deadline 硬截断,
        故即便 root_share 很小, 也不会超发。
        """
        if self.dp_time_share <= 0.0:
            return True
        if self.det_iters > 0:
            return self._dp_calls_node < self.dp_time_share * self.det_iters
        return self._dp_time_node < self.dp_time_share * self.T_MAX

    def _det_continue(self, t_deadline: float) -> bool:
        """[确定性模式] 循环延续判定: det 模式下按迭代数, 否则按墙钟。"""
        if self.det_iters > 0:
            return self._it_local < self.det_iters
        return time.time() < t_deadline

    def _det_remain(self, t_deadline: float) -> float:
        """[确定性模式] 虚拟剩余量: 把 det_iters 线性映到 T_MAX 秒量纲,
        使 pricing_round_plan 的 remain_s<60 阈值在最后 60/T_MAX 比例处触发。"""
        if self.det_iters > 0:
            return (self.det_iters - self._it_local) * (self.T_MAX / max(1, self.det_iters))
        return t_deadline - time.time()

    def _det_tlimit(self, t_deadline: float, lo: float, hi: float) -> float:
        """a_star 时限: det 模式/工作预算制下无限 (仅 pop 帽约束), 保证确定性与无吸收。"""
        if self.det_iters > 0 or self.tier_pop_bounded:
            return 1e9
        return max(lo, min(hi, t_deadline - time.time()))

    def _consume_dp_route(self, pi, sigma, cols, mu):
        if not (self.measure_dp_routes or self.insert_dp_routes):
            return
        if mu < -EPS_RC:
            self.route_stats["negative"] += 1
        info = getattr(self.pricer, "_last_recovery", None)
        if not info:
            return
        self.route_stats["recovered"] += 1
        ok = abs(info["dp_rc"] - info["mu"]) <= 1e-4 * max(1.0, abs(info["mu"]))
        if ok:
            self.route_stats["value_ok"] += 1
        if mu >= -EPS_RC or not (ok and info["elementary"]):
            return
        self.route_stats["elementary"] += 1
        route = info["route"]
        if sum(int(self.demands_g[u]) for u in route[1:-1]) > self.Q:
            return
        if not energy_feasible(self.costs, route):   # [路线 2] 电量门
            return
        if reduced_cost(self.costs, route, pi, sigma) >= -EPS_RC:
            return
        self.route_stats["poolable"] += 1
        if not self.insert_dp_routes:
            return
        key = tuple(route)
        if any(tuple(col["route"]) == key for col in cols):
            return
        cols.append(make_col(self.costs, route, pi, sigma))
        self.columns_total += 1

    def _perturbation_ub(self, seconds: float):
        """扰动-重插-局部搜索循环: 随机拆除 8 客户 → 最省插入修复 → VND。"""
        t0 = time.time()
        best = [list(r) for r in self.ub_routes]
        if not best:
            return
        if self.best_relocate:
            while time.time() - t0 < seconds and best:
                nxt, improved = best_cross_moves(self.costs, best, self.Q, self.demands_g)
                if not improved:
                    break
                try:
                    assert_partition(nxt, self.N, self.K, self.Q, self.demands_g,
                                     costs=self.costs)
                except AssertionError:
                    break
                best = nxt
            self._try_update_ub(best)
            return
        while time.time() - t0 < seconds:
            incumbent = [list(r) for r in best]
            kick = self.rng.sample(range(1, self.N + 1),
                                   min(8, self.N // 4))
            refined = destroy_repair(self.costs, incumbent, kick, self.Q, self.K, self.demands_g)
            improved = False
            if refined is not None:
                c = sum(self.costs.route_cost(r) for r in refined)
                cb = sum(self.costs.route_cost(r) for r in incumbent)
                improved = c < cb - 1e-6
                if improved:
                    best = [list(r) for r in refined]
            if self.collect_kicks:
                self.kick_rows.append({
                    "incumbent": incumbent,
                    "kick": [int(u) for u in kick],
                    "improved": int(improved),
                })
        self._try_update_ub(best)

    # ---------------- 节点列生成 ----------------
    def _node_cg(self, cols: List[dict], bctx: BranchCtx, t_deadline: float,
                 is_root: bool) -> Tuple[float, bool, List[dict],
                                         np.ndarray, float, np.ndarray]:
        """单节点 CG + RCI 割分离：Tier-0 贪心 / Tier-1 平滑 A* / Tier-2 认证定价。

        圆整容量割全局有效；割对偶按 Θ_h = Σ_{S∋h} θ_S 可加分解后并入定价对偶
        π' = π + Θ，定价三层结构与 U 表无需任何结构性改动。
        返回 (z_rmp, certified, 列池, π(裸), σ, λ)。
        """
        pi_c, sig_c = None, 0.0
        alpha = 0.5 if self.stabilize else 0.0
        node_farley = -math.inf
        self._pi_box_prev = None   # [组件一] 对偶盒步长参考态 (逐节点重置)
        self._dp_time_node = 0.0   # [重工程地基] 本节点精算 DP 累计耗时
        self._last_mu4 = None      # [跨节点陈旧 μ̂ 守卫] 节点入口复位:
                                    #  本节点首次 DP 结算之前不更新棘轮 (配对
                                    #  纪律见论文 §4.2/定理 1(b): μ̂ 须同节点内
                                    #  结算; 节点内 z 单调不增 ⇒ 复用界仍有效)
        certified = False
        z = pi = sigma = lam = None
        t_node0 = time.time()
        self._t_node0 = t_node0
        self._it_local = 0
        self._dp_calls_node = 0
        _aeb_boost_used = False   # [AEB 生产化] 每 node 至多一次末期时限提额
        while self._det_continue(t_deadline):
            self.cg_iterations += 1
            self._it_local += 1
            _obs = A_STAR_STATS_ON
            _t_rmp = time.time() if _obs else 0.0
            z, pi, sigma, lam, slack, theta, sri_pen = solve_rmp(
                self.costs, cols, self.K, self.cuts, self.sris)
            if _obs:
                NODE_TIER_STATS["cg_iterations"] += 1
                NODE_TIER_STATS["rmp_time_s"] += time.time() - _t_rmp
            if slack:
                self._log("    [CG] 警告: 覆盖松弛激活 (定价自愈中)")
            # SRI-3 分离 [Slot A 重测：干净负结果，2026-09-25，已关闭]
            # 已按 Theorem 5.1/5.2 修正：定价/认证用裸 π，割罚仅在列结算时按完整
            # bitmask 精确计入（reduced_cost_cut）。结果：
            #   激进(每3迭代,≤8割,帽40): SG60n 150s LB=0 vs 基线961；
            #   温和(每10迭代,≤3割,帽12): SG60n 500s LB=867.71 vs 默认1004.02。
            # 机理：割增广 LP 的最优对偶使 |π|、|θ| 巨大(实测 max|π|≈76kJ,
            # Σθ≈-2.9MJ)，而 Theorem 5.2 的 cut-free μ̂ 随之崩溃(μ̂≈-618kJ)，
            # Farley = z+K·min(0,μ̂) → 0。半权折入 π 与裸 π 两种实现同样崩溃，
            # 故死因不是半权，而是 cut-free μ̂ 在大割对偶下失效。
            # 重启条件：需 cut-augmented μ̂（closure 精确割罚）才可能转正；
            # 基础设（separate_sri / reduced_cost_cut）保留供审计复现。
            if False and len(self.sris) < 40 and self.cg_iterations % 3 == 0 and float(np.max(np.minimum(lam, 1.0 - lam))) > 1e-6:
                new_sris = separate_sri(self.costs, cols, lam, self.sri_keys)
                if new_sris:
                    self.sris.extend(new_sris)
                    self.n_cuts_added += len(new_sris)
                    continue          # 先以新割行重解 LP
            # 圆整容量割不分离：等式覆盖下 x(S)=|S|≥ceil(d(S)/Q)。
            # [Theorem 5.1/5.2] 认证/定价路径使用裸 π; 割对偶不折入 U 表/DP。
            # 证书仍有效: RC_cut = RC_free + (−θ'γ) ≥ RC_free (θ≤0), 故裸 π 的
            # no_neg ⇒ 割增广 no_neg; μ̂_free ≥ μ̂_half 且仍 ≤ min_elem RC_cut。
            pi_free = pi
            late_phase = (self._it_local >= 0.82 * self.det_iters) if self.det_iters > 0 \
                else ((t_deadline - time.time()) < 0.18 * (t_deadline - self._t_node0))
            self._late_phase = late_phase   # 供 COD 增广门 (末期才开, 避免 3x DP 代价饥饿 CG)
            # 每 25 迭代以当前对偶重建认证邻域 (阻断套利循环)
            if self.cg_iterations % 25 == 0:
                self._cert_nb = self.pricer.cert_nb_from_duals(
                    pi_free, self.pricer._n_mem_bits)
            nb_use = getattr(self, "_cert_nb", None)
            # ng 精确 DP 认证预言机: 紧致 μ̂ (Farley) + 无预算认证判据
            # [防污染守卫] 任一松弛列激活时 z 含 Big-M, 禁用 Farley/认证
            # 组件 E: 节流采样 exact_ng4_dp，释放 50%+ 算力给列生成 (受 enable_throttle 控制)
            early = (self.enable_two_pass
                     and (time.time() - t_node0) < self.two_pass_early_s
                     and (t_deadline - time.time()) > 8.0)
            if early:
                self.two_pass_early_iters += 1
            run_dp, update_farley = pricing_round_plan(
                self.enable_two_pass, self.enable_throttle, early,
                self.cg_iterations, late_phase, self._det_remain(t_deadline))
            feat = None
            farley_before = node_farley
            if self.collect_dp and not slack:
                feat = dp_iteration_features(
                    pi_free, self._pi_prev, lam, cols, z, self._z_prev, self.cg_iterations)
                self._pi_prev = np.array(pi_free, copy=True)
                self._z_prev = float(z)
                run_dp, update_farley = True, True

            if run_dp and self._dp_allowed(t_deadline, t_node0):
                self._dp_calls_node += 1
                _t_dp = time.time() if _obs else 0.0
                _t_dp_real = time.time()
                # [COD 增广门] 只在"μ̂ 真正决定 Farley 界且处于末期精算认证窗口"的调用上增广;
                # 增广轮次已计入 _dp_time_node (下一行整调用计时) ⇒ 不额外挤占预算。
                self._cod_window = bool(
                    update_farley and self._det_remain(t_deadline) < 60.0)
                mu4, no_neg = self._exact_mu(pi_free, sigma, bctx, nb_override=nb_use)
                self._dp_time_node += time.time() - _t_dp_real
                if _obs:
                    NODE_TIER_STATS["dp_time_s"] += time.time() - _t_dp
                self._last_mu4 = mu4
                self._consume_dp_route(pi_free, sigma, cols, mu4)
            else:
                mu4 = self._last_mu4
                no_neg = False
            if not slack and update_farley and mu4 is not None:
                node_farley = max(node_farley, max(0.0, z + self.K * min(0.0, mu4)))
                if is_root:
                    self.root_farley = max(self.root_farley, node_farley)
                self.farley_history.append(node_farley)
            else:
                no_neg = False
            if feat is not None:
                scale = self.ub if math.isfinite(self.ub) else z
                self.dp_rows.append((feat, int(node_farley > farley_before + 0.001 * max(scale, 1.0))))
            if self.verbose and self.cg_iterations % 20 == 0:
                self._log(f"    [CG {self.cg_iterations:3d}] z={z/1e3:8.2f} "
                          f"mu4={mu4/1e3:7.2f} farley={node_farley/1e3:8.2f} "
                          f"cols={len(cols)}")
            if pi_c is None:
                pi_c, sig_c = pi_free.copy(), sigma
            new_cols: List[dict] = []
            pi_t = sig_t = None
            if self.enable_exp3 and self.stabilize:
                K_exp = len(self.exp3_arms)
                eta = math.sqrt(2.0 * math.log(K_exp) / (2600.0 * K_exp))
                p_dist = self._exp3_w / self._exp3_w.sum()
                p_dist = (1.0 - 0.06) * p_dist + 0.06 / K_exp
                arm_idx = int(self.rng.choices(range(K_exp), weights=p_dist)[0])
                self._exp3_last_arm = (arm_idx, float(p_dist[arm_idx]), eta)
                alpha = self.exp3_arms[arm_idx]
            if alpha > 0.0:
                pi_t = alpha * pi_c + (1 - alpha) * pi_free
                sig_t = alpha * sig_c + (1 - alpha) * sigma
                if self.dual_box:
                    # [组件一: 间隙自收缩对偶盒] 步长盒 L_i = q_i·gap/Q (推导见
                    # 间隙自收缩对偶盒推导 §2, 无自由参数, 自退火)。
                    # 仅作用于交付定价对偶 π̃; 裸 π_free 的认证路径不变 (§4 隔离)。
                    # σ 不盒: Farley 中性 (§3)。默认关, 零行为变化。
                    _gap_box = (self.ub - node_farley
                                if math.isfinite(self.ub) and math.isfinite(node_farley)
                                else math.inf)
                    if math.isfinite(_gap_box) and _gap_box > 0.0:
                        L_box = self.demands_g[1:] * (_gap_box / self.Q)
                        if self._pi_box_prev is not None:
                            pi_t = np.clip(pi_t,
                                           self._pi_box_prev - L_box,
                                           self._pi_box_prev + L_box)
                    self._pi_box_prev = pi_t.copy()
                # Tier-0: 贪心插入 (构造性初等, 平滑对偶引导; 末期试次加倍)
                _t_g = time.time() if _obs else 0.0
                _trials = 20 if late_phase else 14
                g_cols = self.pricer.greedy_insertion(
                    pi_t, sig_t, bctx, self.rng, trials=_trials)
                if _obs:
                    NODE_TIER_STATS["greedy_time_s"] += time.time() - _t_g
                    NODE_TIER_STATS["greedy_cols"] += len(g_cols)
                for col in g_cols:
                    col["rc"] = reduced_cost_cut(self.costs, col["route"],
                                                 pi_free, sigma, self.sris,
                                                 sri_pen)
                    if col["rc"] < -EPS_RC:
                        new_cols.append(col)
                # Tier-1: 平滑 A* (小预算补充)
                if len(new_cols) < 6:
                    _t_t1 = time.time() if _obs else 0.0
                    U_t, _ = self.pricer.u_table(pi_t, sig_t, bctx)
                    if _obs:
                        NODE_TIER_STATS["u_table_time_s"] += time.time() - _t_t1
                        NODE_TIER_STATS["tier1_triggered"] += 1
                    trace = {"paths": [], "features": []} if self.trace_labels else None
                    if _obs:
                        A_STAR_STATS["call_tier"] = "tier1"
                    a_cols, _, _ = self.pricer.a_star(
                        pi_t, sig_t, bctx, U_t, max_pops=2500,
                        time_limit=self._det_tlimit(t_deadline, 0.3, 1.5),
                        trace=trace)
                    if _obs:
                        NODE_TIER_STATS["tier1_time_s"] += time.time() - _t_t1
                        NODE_TIER_STATS["tier1_cols"] += len(a_cols)
                    if trace is not None and "X" in trace:
                        self.label_traces.append(trace)
                    # [Step-1 标签真值层] 同状态重跑深搜索 (20k pops),
                    # 以深搜索产出作为“导向负列”标签真值;
                    # 仅在 trace_labels 观测模式下运行, 不改变主行为。
                    if self.trace_labels:
                        if _obs:
                            A_STAR_STATS["call_tier"] = "tier1deep"
                        deep_cols, _, _ = self.pricer.a_star(
                            pi_t, sig_t, bctx, U_t, max_pops=100000,
                            time_limit=30.0)
                        if _obs:
                            A_STAR_STATS["call_tier"] = "tier1"
                        if deep_cols and trace is not None and trace["paths"]:
                            accepted_deep = [tuple(c["route"][1:]) for c in deep_cols]
                            _, y_deep, _ = mark_ancestor_labels(
                                trace["paths"], trace["features"], accepted_deep)
                            trace["y_deep"] = y_deep
                            trace["deep_cols"] = len(deep_cols)
                            trace["shallow_cols"] = len(a_cols)
                            # 状态级列质量真值: 状态路径 → 导向列的最负 RC
                            best_rc = {}
                            for c in deep_cols:
                                rc_c = float(c["rc"])
                                path = tuple(c["route"][1:])
                                for i in range(len(path)):
                                    sfx = path[i:]
                                    if rc_c < best_rc.get(sfx, float("inf")):
                                        best_rc[sfx] = rc_c
                            trace["state_best_rc"] = [best_rc.get(p) for p in trace["paths"]]
                    for col in a_cols:
                        col["rc"] = reduced_cost_cut(self.costs, col["route"],
                                                     pi_free, sigma, self.sris,
                                                     sri_pen)
                        if col["rc"] < -EPS_RC:
                            new_cols.append(col)
            if not new_cols and no_neg:
                certified = True            # ng-4-DP 穷尽证明无负检验数列
                break
            # 末期 (最后 35% 时间) 每 5 迭代 ng-8 + 对偶邻域精算 μ̂
            if self.T_MAX >= 120.0 and late_phase and self.cg_iterations % 7 == 0 \
                    and self._dp_allowed(t_deadline, t_node0):
                    nb8 = self.pricer.cert_nb_from_duals(pi_free, self.ng_endgame)
                    _t_dp8 = time.time()
                    mu8, no_neg8 = (self._cert_mu if self._cert_ok(is_root, t_deadline)
                                    else self._exact_mu)(
                        pi_free, sigma, bctx, n_bits=self.ng_endgame, nb_override=nb8)
                    self._dp_time_node += time.time() - _t_dp8
                    self._consume_dp_route(pi_free, sigma, cols, mu8)
                    if not slack:
                        node_farley = max(node_farley,
                                          max(0.0, z + self.K * min(0.0, mu8)))
                        if is_root:
                            self.root_farley = max(self.root_farley, node_farley)
                        if self.record_sites and is_root:
                            self._record_site(z, pi_free, sigma, t_deadline,
                                              "mu8", no_neg8)
                        if no_neg8:
                            certified = True
                            break
            if not new_cols:
                # Tier-2a: ng-8 精算 μ̂ (Farley 冲刺, 只在长预算启用)
                if self.T_MAX >= 120.0 and self._dp_allowed(t_deadline, t_node0):
                    _t_dp2a = time.time()
                    mu8, no_neg8 = (self._cert_mu if self._cert_ok(is_root, t_deadline)
                                    else self._exact_mu)(
                        pi_free, sigma, bctx, n_bits=self.ng_endgame)
                    self._dp_time_node += time.time() - _t_dp2a
                    self._consume_dp_route(pi_free, sigma, cols, mu8)
                    if not slack:
                        node_farley = max(node_farley,
                                          max(0.0, z + self.K * min(0.0, mu8)))
                        if is_root:
                            self.root_farley = max(self.root_farley, node_farley)
                        self.farley_history.append(node_farley)
                        if self.record_sites and is_root:
                            self._record_site(z, pi_free, sigma, t_deadline,
                                              "t2a", no_neg8)
                        if no_neg8 and not slack:
                            certified = True
                            break
                # Tier-2b: 真实对偶 + 小邻域 ng 认证级定价 (列提取器)
                _t_t2b = time.time() if _obs else 0.0
                U_true, _ = self.pricer.u_table(pi_free, sigma, bctx)
                if _obs:
                    A_STAR_STATS["call_tier"] = "tier2b"
                # [AEB 生产化] 末期认证窗口 (剩余<90s) 内首次 tier2b 时限 15→50s,
                # 让 A* 穷尽 (实测 25-48s) ⇒ 真值成本证书生效; 每 node 至多一次提额。
                _aeb_win = (self.astar_bound and self.det_iters <= 0
                            and self._det_remain(t_deadline) < 90.0
                            and not _aeb_boost_used)
                if _aeb_win:
                    _aeb_boost_used = True
                exact_cols, astar_cert, astar_mu = self.pricer.a_star(
                    pi_free, sigma, bctx, U_true, max_pops=500000,
                    time_limit=self._det_tlimit(t_deadline, 1.0,
                                                50.0 if _aeb_win else 15.0),
                    ng_masks=self.pricer._ng_masks_cert,
                    max_cols=self.t2b_max_cols,
                    stop_on_max_cols=False)
                # [AEB 界组合] A* 穷尽 (certified) ⇒ mu_A* 是有效下界; 与 DP 的 mu4 取 max。
                # 两个独立有效下界取 max 是合法组合算术 (LLM4MIP bound-validity §7)。
                # 实测 SG120 det_200: mu_DP=-14.7637 (细网格) vs mu_A*=-12.7797 (真值成本)。
                if self.astar_bound and astar_cert and not slack and update_farley:
                    mu_best = min(0.0, max(mu4, float(astar_mu)))
                    node_farley = max(node_farley, max(0.0, z + self.K * mu_best))
                    if is_root:
                        self.root_farley = max(self.root_farley, node_farley)
                    self.farley_history.append(node_farley)
                if _obs:
                    NODE_TIER_STATS["tier2b_time_s"] += time.time() - _t_t2b
                    NODE_TIER_STATS["tier2b_triggered"] += 1
                    NODE_TIER_STATS["tier2b_cols"] += len(exact_cols)
                new_cols = exact_cols
            seen = {tuple(c["route"]) for c in cols}
            added = 0
            for col in new_cols:
                # [Theorem 5.1] 列入池前以精确割增广检验数结算 (exact_cols 亦覆盖)
                col["rc"] = reduced_cost_cut(self.costs, col["route"], pi_free,
                                             sigma, self.sris, sri_pen)
                if col["rc"] >= -EPS_RC:
                    continue
                # [路线 2] 列准入全局能量门: 任何来源 (贪心/A*/DP 回收) 的列
                # 必须满足逐航线电量硬约束 (col["cost"] = 精确 route_cost = 能耗)
                if col["cost"] > E_MAX_J + 1e-6:
                    continue
                k = tuple(col["route"])
                if k not in seen:
                    cols.append(col)
                    seen.add(k)
                    added += 1
            self.columns_total += added
            # C1: EXP3 权重更新
            if self.enable_exp3 and self._exp3_last_arm is not None:
                arm_idx, p_arm, eta = self._exp3_last_arm
                r = min(1.0, added / 10.0)
                r_hat = r / max(p_arm, 1e-9)
                self._exp3_w[arm_idx] *= math.exp(eta * r_hat)
                mx = float(self._exp3_w.max())
                if mx > 1e30:
                    self._exp3_w /= mx
            # Wentges 自适应: 有效产出 → 中心游走 + α 上调; 停滞 → α 减半
            if added > 0:
                if pi_t is not None and self.stabilize:
                    pi_c = 0.5 * pi_c + 0.5 * pi_t
                    sig_c = 0.5 * sig_c + 0.5 * sig_t
                    alpha = min(0.9, alpha + 0.05)
            else:
                if certified:
                    break           # 认证收敛: 真对偶下无负检验数列
                if alpha > 0.0:
                    alpha = 0.0 if alpha <= 0.15 else alpha * 0.5
                    continue        # 以更接近真对偶的 π̃ 重试
                break               # 预算耗尽 (未认证, 回退 Farley 继承)
            # 列池裁剪（防止 LP 规模膨胀; 认证逻辑不受影响: 定价在全列空间进行）
            if len(cols) > 2600:
                active = [cols[j2] for j2, lam2 in enumerate(lam) if lam2 > 1e-6]
                cols.sort(key=lambda c: c.get("rc", 0.0))
                keep = cols[:2000]
                seenk = {tuple(c["route"]) for c in keep}
                for c in active:
                    if tuple(c["route"]) not in seenk:
                        keep.append(c)
                        seenk.add(tuple(c["route"]))
                cols = keep
        z, pi, sigma, lam, slack_exit, _, _ = solve_rmp(self.costs, cols, self.K,
                                                         self.cuts, self.sris)
        # [DFP] 对偶面证书抛光 (节点出口, 每节点至多一次): 终末池 + 终末对偶
        # 建最优面 LP → L1 最近投影候选 → 穷尽 DP 复认证 → 棘轮只收认证值。
        # 证书环节隔离: 候选对偶不交付定价, 列池/UB 管线分毫不动 (det UB 逐位不变)。
        if self.dual_polish and cols and not slack_exit:
            lb_pol, _df = self._dual_face_polish(cols, z, pi, sigma, bctx)
            self.polish_calls += 1
            if lb_pol is not None and lb_pol > node_farley:
                self.polish_gain += lb_pol - node_farley
                node_farley = max(0.0, lb_pol)
                if is_root:
                    self.root_farley = max(self.root_farley, node_farley)
                self.farley_history.append(node_farley)
        if is_root:
            self.root_farley = max(self.root_farley, node_farley)
        self._node_farley = node_farley
        return z, certified, cols, pi, sigma, lam

    # ---------------- 主循环 ----------------
    def _record_site(self, z, pi, sigma, t_deadline, tier, no_neg):
        """[后置解耦 v2] 被动录制根节点 Farley 位点 (只 append, 零轨迹扰动)。"""
        self.site_log.append({
            "z": float(z), "pi": np.array(pi, dtype=np.float64, copy=True),
            "sigma": float(sigma), "remain": float(self._det_remain(t_deadline)),
            "it": int(self.cg_iterations), "tier": tier, "no_neg": bool(no_neg)})

    def certify_post(self, lb_tree: float, ub: float, cert_grid: int = 1020,
                     window_s: float = 120.0, max_calls: int = 8) -> float:
        """[组件 A|后置] 共享轨迹证书后处理器: 对录制的基础对偶流重放细网格认证。

        定理 (后置解耦单调性, 单调递减要求的构造性保证):
          位点全部取自同一次基础运行; 重放只读 (z, π, σ, 根 bctx), 不触碰列池/
          UB/RNG/树 ⇒ UB 逐位不变。每位点的 Farley 候选 z + K·min(0, μ̂_cert) 是
          根松弛的有效全局下界 (根子树 = 全体可行解), no_neg 位点的 z 本身即
          ng 松弛 LP 最优 (有效下界); LB' = max(lb_tree, 棘轮) 且 LB' ≤ UB ⇒
          gap(M2) ≤ gap(M1) 与 gap(M4) ≤ gap(M3) 按构造逐位成立, 零墙钟耦合。
        位点选择复刻生产门 _cert_ok 语义: 根 + 末期窗口 (remain < window_s) +
          次数帽, 取窗口内最末 max_calls 个 (对偶最收敛)。
        """
        if not self.site_log:
            self.cert_post_lb = float(lb_tree)
            return float(lb_tree)
        sites = [s for s in self.site_log if s["remain"] < window_s]
        if max_calls and max_calls > 0:
            sites = sites[-max_calls:]
        if not sites:                      # 极短预算: 无窗口位点, 退化为末 1 个
            sites = self.site_log[-1:]
        old_grid, self.cert_grid = self.cert_grid, int(cert_grid)
        lb = float(lb_tree)
        try:
            for s in sites:
                nb = self.pricer.cert_nb_from_duals(s["pi"], self.ng_endgame)
                mu, no_neg = self._cert_mu(s["pi"], s["sigma"], self._root_bctx,
                                           n_bits=self.ng_endgame, nb_override=nb)
                cand = max(0.0, s["z"] + self.K * min(0.0, mu))
                if no_neg:
                    cand = max(cand, s["z"])     # ng 穷尽 ⇒ z 为有效全局下界
                cand = min(cand, float(ub))      # 棘轮不越 UB (证书链守卫)
                lb = max(lb, cand)
                self.cert_post_calls += 1
        finally:
            self.cert_grid = old_grid
        self.cert_post_lb = lb
        self.cert_post_gain = lb - float(lb_tree)
        assert ub >= lb - 1e-3, "post 证书越 UB: 证书链断裂!"
        return lb

    def mono_kernel(self, seconds: float = None, share: float = 0.25) -> float:
        """[组件 B|后置] 终局单调内核 v3: 两段只接受改进的 LNS (严格段 + 平台游走段)。

        定理 (单调性): 搜索已结束, 内核以最终 incumbent 为起点, 两段共享全局
        best-ever (段 1 严格改进; 段 2 接受等价移动做平台游走但只有严格更优才
        进入 best), 末尾划分断言失败回退原 incumbent, 全程不触碰列池/对偶/树 ⇒
        UB' ≤ UB, LB 逐位不变 ⇒ gap' ≤ gap 按构造成立 (旧搜索期 larr shell 会
        扰动 LNS/列池轨迹造成跨臂墙钟彩票, 已从部署串移除, 本内核为其单调化替代)。
        段 1 (窗口 share×T): destroy-repair + 子集 DP 保序重排 + 接受时交叉移动
        抛光, 仅严格改进进入 best。段 2 (窗口 share×T): 1-1 交换算子初始化
        (swap_pass, 基础 VND 未覆盖邻域) + 平台侧向游走 + 周期重启到 best +
        混合扰动尺度 (6..16)。两段算子互补: 严格段收割浅层改进, 游走段逃离
        局部最优 (离线对照: 单段各 116.6 / 112.7 kJ, 逐格取优 133.1 kJ)。
        通用窗口: 每段 share × T_MAX (随预算规模自适应, 零城市分支)。
        """
        if not self.ub_routes:
            return self.ub
        if seconds is None:
            seconds = share * self.T_MAX
        t0 = time.time()
        best = [list(r) for r in self.ub_routes]
        c_best = sum(self.costs.route_cost(r) for r in best)
        # ---- 段 1: 严格改进 LNS (重排 + 交叉抛光) ----
        while time.time() - t0 < seconds:
            kick = self.rng.sample(range(1, self.N + 1), min(8, self.N // 4))
            cand = destroy_repair(self.costs, [list(r) for r in best], kick,
                                  self.Q, self.K, self.demands_g)
            if cand is None:
                continue
            cand = [self._larr_route([u for u in r if u != 0]) for r in cand]
            c = sum(self.costs.route_cost(r) for r in cand)
            if c < c_best - 1e-6:
                nxt, improved = best_cross_moves(self.costs, cand, self.Q,
                                                 self.demands_g)
                if improved:
                    c2 = sum(self.costs.route_cost(r) for r in nxt)
                    if c2 < c - 1e-6:
                        cand, c = nxt, c2
                best, c_best = [list(r) for r in cand], c
        # ---- 段 2: 交换初始化 + 平台侧向游走 + 周期重启 (共享 best) ----
        t2 = time.time()
        cur = fleet_local_search(self.costs,
                                 swap_pass(self.costs, best, self.Q,
                                           self.demands_g),
                                 self.Q)
        cur = [self._larr_route([u for u in r if u != 0]) for r in cur]
        c_cur = sum(self.costs.route_cost(r) for r in cur)
        if c_cur < c_best - 1e-6:
            best, c_best = [list(r) for r in cur], c_cur
        tries = 0
        while time.time() - t2 < seconds:
            kick_n = self.rng.randint(6, 16)          # 混合扰动尺度
            kick = self.rng.sample(range(1, self.N + 1), min(kick_n, self.N // 4))
            cand = destroy_repair(self.costs, [list(r) for r in cur], kick,
                                  self.Q, self.K, self.demands_g)
            tries += 1
            if cand is None:
                continue
            cand = [self._larr_route([u for u in r if u != 0]) for r in cand]
            c = sum(self.costs.route_cost(r) for r in cand)
            if c < c_cur - 1e-6:                      # 严格改进: 交叉移动再抛光
                nxt, improved = best_cross_moves(self.costs, cand, self.Q,
                                                 self.demands_g)
                if improved:
                    c2 = sum(self.costs.route_cost(r) for r in nxt)
                    if c2 < c - 1e-6:
                        cand, c = nxt, c2
                cur, c_cur = cand, c
                if c < c_best - 1e-6:
                    best, c_best = [list(r) for r in cand], c
            elif c < c_cur + 1e-6:                    # 平台侧向游走 (best-ever 保护)
                cur, c_cur = cand, c
            if tries % 300 == 0:                      # 周期重启到 best, 防漂移失控
                cur, c_cur = [list(r) for r in best], c_best
        try:
            assert_partition(best, self.N, self.K, self.Q, self.demands_g,
                             costs=self.costs)
        except AssertionError:                        # 防御: 回退原 incumbent (不劣)
            best, c_best = [list(r) for r in self.ub_routes], sum(
                self.costs.route_cost(r) for r in self.ub_routes)
        if c_best < self.ub - 1e-6:
            self.kernel_gain = self.ub - c_best
            self._log(f"    [mono-kernel] UB {self.ub/1e3:.2f} -> "
                      f"{c_best/1e3:.2f} kJ (-{self.kernel_gain/1e3:.2f})")
            self.ub, self.ub_routes = c_best, [list(r) for r in best]
        assert self.ub >= self.root_farley - 1e-3, "UB 击穿下界: 证书链断裂!"
        return self.ub

    def solve(self, warm_routes: Optional[List[List[int]]] = None) -> Dict:
        t0 = time.time()
        self.pricer.recover_routes = self.measure_dp_routes or self.insert_dp_routes
        # 初始列池：单客户列 + 贪心机队种子（保证 RMP 无 Big-M 可行）
        cols: List[dict] = []
        seen = set()
        for i in range(1, self.N + 1):
            col = make_col(self.costs, [0, i, 0])
            # 电量门: 九算例单客户航线最大 267.8 kJ < 360 (数值验证), 本门
            # 恒放行; 在此明示以使 "初始列池 ⊆ Ω_E" 在代码层无例外成立。
            if col["cost"] > E_MAX_J + 1e-6:
                continue
            cols.append(col)
            seen.add(tuple(col["route"]))
        seed_routes = greedy_fleet(self.costs, self.K, self.Q, None)
        # 电量硬约束: 种子/热身航线只有能量可行者才准入列池;
        # 整组可行时才作 UB 种子 (贪心短视时交由松弛定价自愈, 不崩溃)。
        seeds_ok = all(energy_feasible(self.costs, r) for r in seed_routes)
        if seeds_ok:
            self._try_update_ub([list(r) for r in seed_routes])
        for r in list(seed_routes) + list(warm_routes or []):
            k = tuple(r)
            if k not in seen and energy_feasible(self.costs, r):
                cols.append(make_col(self.costs, r))
                seen.add(k)
        if warm_routes and all(energy_feasible(self.costs, r) for r in warm_routes):
            self._try_update_ub([list(r) for r in warm_routes])
        self.columns_total = len(cols)

        heap: List[Tuple[float, int, BranchCtx, List[dict], int]] = [
            (0.0, 0, BranchCtx(self.N), cols, 0)]
        self._root_bctx = heap[0][2]   # [后置解耦 v2] certify_post 重放的根上下文
        lb_valid = {0: -math.inf}     # seq -> 子树有效下界（认证链继承）
        seq_counter = 0
        lb_tree, gap = -math.inf, math.inf

        def _tree_gap() -> Tuple[float, float]:
            lbs = list(lb_valid.values())
            lb = min([self.ub] + lbs) if lbs else self.ub
            if self.enable_ap_elb:
                lb = max(lb, self.lb_ap)
            return lb, (self.ub - lb) / self.ub

        while heap:
            if self.det_iters > 0:
                if self.nodes_total >= 1:
                    break
            elif time.time() - t0 >= self.T_MAX:
                break
            _, seq, bctx, pool, depth = heapq.heappop(heap)
            lb_inherited = lb_valid.pop(seq)
            if lb_inherited >= self.ub - 1e-6:
                self.nodes_pruned += 1
                continue
            self.nodes_total += 1
            remain = self.T_MAX - (time.time() - t0)
            node_cap = max(232.0, self.T_MAX * self.root_share) if self.nodes_total == 1 else 18.0
            node_cap = min(remain, node_cap)
            self.node_time_caps.append(node_cap)
            deadline = time.time() + node_cap
            z, certified, pool, pi, sigma, lam = self._node_cg(
                pool, bctx, deadline, is_root=(self.nodes_total == 1))
            node_farley = getattr(self, "_node_farley", -math.inf)
            node_lb = max(lb_inherited, node_farley)
            if certified:
                node_lb = max(node_lb, z)
                self.nodes_certified += 1
            frac = float(np.max(np.minimum(lam, 1.0 - lam))) if len(lam) else 0.0
            # 原始启发：取整 + 根节点列池 IP
            self._round_heuristic(pool, lam)
            if self.nodes_total == 1:
                self._pool_ip(pool, time_cap=15.0 if self.T_MAX > 300 else 10.0)
                if self.det_iters <= 0:               # 确定性模式: 跳过时间驱动的 LNS
                    if self.lns_time_share > 0.0:
                        # 份额上限 = lns_time_share × T_MAX, 再被物理可用量 (remain-36, 留给树) 夹住
                        _lns_cap = min(self.lns_time_share * self.T_MAX, remain - 36.0)
                    else:
                        _lns_cap = min(60.0 if self.T_MAX > 300 else 30.0, remain - 36.0)
                    self._perturbation_ub(_lns_cap)
            if certified and z >= self.ub - 1e-6:
                self.nodes_pruned += 1
                lb_tree, gap = _tree_gap()
                if gap <= GAP_TARGET - 0.001:
                    break
                continue
            if frac < 1e-6:
                if certified:
                    # 池 LP 整数最优 + 认证定价 ⇒ 子树 IP 最优 = z, 节点闭合
                    self.nodes_integral += 1
                    lb_tree, gap = _tree_gap()
                    if gap <= GAP_TARGET - 0.001:
                        break
                    continue
                # 未认证的整数 LP: 重入队继续定价, 绝不静默丢弃子树 (证书链完整)
                seq_counter += 1
                lb_valid[seq_counter] = node_lb
                heapq.heappush(heap, (z, seq_counter, bctx, pool, depth))
                self._log(f"  [node {self.nodes_total:3d}] LP 整数但未认证, "
                          f"重入队 (z={z/1e3:.2f} kJ)")
                continue
            arc, ok = self._select_branch_arc(pool, lam)
            if not ok:
                # 定理（见 THEORETICAL_PROOF.md 命题 B-3）：去重初等列池上
                # λ 分数 ⇒ 必存在分数弧流；若触发本断言说明实现缺陷。
                raise AssertionError(
                    f"分数 λ 但无分数弧流: frac={frac}, 节点 {self.nodes_total}")
            u, v = arc
            base_lb = node_lb
            # SAME 子节点（R1/R2 拦截编译 + 列池清洗 + 种子注入）
            c_same = bctx.child_same(u, v)
            if c_same is not None:
                seq_counter += 1
                pool_same = [c for c in pool if c_same.col_ok(c["route"])]
                for ic in self._inject_columns(c_same):
                    kt = tuple(ic["route"])
                    if kt not in {tuple(c["route"]) for c in pool_same}:
                        pool_same.append(ic)
                lb_valid[seq_counter] = base_lb
                heapq.heappush(heap, (z, seq_counter, c_same, pool_same,
                                      depth + 1))
            # DIFFER 子节点（删弧 + 列池清洗）
            c_diff = bctx.child_differ(u, v)
            seq_counter += 1
            pool_diff = [c for c in pool
                         if (u, v) not in _arcs_of(c["route"])]
            for ic in self._inject_columns(c_diff):
                kt = tuple(ic["route"])
                if kt not in {tuple(c["route"]) for c in pool_diff}:
                    pool_diff.append(ic)
            lb_valid[seq_counter] = base_lb
            heapq.heappush(heap, (z, seq_counter, c_diff, pool_diff, depth + 1))
            lb_tree, gap = _tree_gap()
            self._log(f"  [node {self.nodes_total:3d}] d={depth} "
                      f"z={z/1e3:8.2f}kJ cert={int(certified)} "
                      f"UB={self.ub/1e3:8.2f} LBtree={lb_tree/1e3:8.2f}kJ "
                      f"gap={gap*100:5.2f}% cols={len(pool)} heap={len(heap)}")
            if gap <= GAP_TARGET - 0.001:
                break

        # [LARR-cert] 终局后置重排: 搜索零扰动 ⇒ UB 单调不增、LB_tree 逐位不变。
        # 重排解仍是可行划分 (同客户集同需求, 保序 DP 不劣含原序), 故 UB' ≥ LB_tree
        # 由证书链自保证 (下方 assert 与 _try_update_ub 同一守卫)。
        if self.larr_cert and self.ub_routes:
            rr = [self._larr_route([u for u in r if u != 0])
                  for r in self.ub_routes]
            c2 = sum(self.costs.route_cost(r) for r in rr)
            if c2 < self.ub - 1e-6:
                self.larr_cert_gain = self.ub - c2
                self._log(f"    [LARR-cert] 终局重排 UB {self.ub/1e3:.2f} -> "
                          f"{c2/1e3:.2f} kJ (-{self.larr_cert_gain/1e3:.2f})")
                self.ub, self.ub_routes = c2, [list(r) for r in rr]
                # 终局能量断言 (同 route_payload 口径; 子集 DP 含原序 ⇒ 逐航线
                # 能耗不增, 生产恒放行, 此处为不变量文档化)
                assert_partition(self.ub_routes, self.N, self.K, self.Q,
                                 self.demands_g, costs=self.costs)
            assert self.ub >= self.root_farley - 1e-3, "UB 击穿下界: 证书链断裂!"

        # 终局树下界
        if heap:
            lb_tree, gap = _tree_gap()
        else:
            lb_tree, gap = self.ub, 0.0   # 全树闭合 ⇒ 证明最优
        elapsed = time.time() - t0
        certified_gap = gap <= GAP_TARGET
        # 极低预算下可能尚无可行 incumbent (greedy 种子能量不可行且 LNS 未及
        # 找到解): 此时无解可导, 由调用方按 ub_routes 为空判定。
        routes_out = self.routes_payload() if self.ub_routes else []
        return {
            "ub_kJ": self.ub / 1e3,
            "lb_tree_kJ": lb_tree / 1e3,
            "gap_pct": gap * 100.0,
            "gap_certified_le_5pct": bool(certified_gap),
            "nodes_total": self.nodes_total,
            "nodes_certified": self.nodes_certified,
            "nodes_pruned_bound": self.nodes_pruned,
            "nodes_integral": self.nodes_integral,
            "columns_total": self.columns_total,
            "larr_cert_gain_kJ": self.larr_cert_gain / 1e3,
            "cert_mu_calls": self.cert_calls,
            "rci_cuts_total": self.n_cuts_added,
            "cg_iterations": self.cg_iterations,
            "exp3_reward_updates": self._exp3_reward_updates,
            "enable_two_pass": self.enable_two_pass,
            "two_pass_early_iters": self.two_pass_early_iters,
            "node_time_caps_s": [round(cap, 3) for cap in self.node_time_caps],
            "time_s": elapsed,
            "time_budget_s": self.T_MAX,
            "routes": routes_out,
        }

    def routes_payload(self) -> List[dict]:
        """[后置解耦 v2] 当前 incumbent 的航线导出 (solve 与后处理器共用)。"""
        routes_out = []
        for r in self.ub_routes:
            load = 0
            legs = []
            for u in r[1:-1]:
                load += int(self.demands_g[u])
            cur = load
            for t in range(len(r) - 1):
                u, v = r[t], r[t + 1]
                legs.append({"u": int(u), "v": int(v), "load_g": int(cur)})
                if v != 0:
                    cur -= int(self.demands_g[v])
            routes_out.append({
                "sequence": [int(x) for x in r],
                "load_g": int(load),
                "cost_kJ": round(self.costs.route_cost(r) / 1e3, 3),
                "legs": legs,
            })
        assert_partition(self.ub_routes, self.N, self.K, self.Q, self.demands_g,
                         costs=self.costs)   # 终局能量可行性硬断言 (路线 2)
        return routes_out

    # ---------------- 分支弧选择 ----------------
    def _select_branch_arc(self, cols: List[dict], lam: np.ndarray
                           ) -> Tuple[Optional[Tuple[int, int]], bool]:
        """γ_uv 最接近 1/2 的分数弧（客户弧优先）。"""
        gamma: Dict[Tuple[int, int], float] = {}
        for j, col in enumerate(cols):
            if lam[j] < 1e-6:
                continue
            r = col["route"]
            for t in range(len(r) - 1):
                a = (r[t], r[t + 1])
                gamma[a] = gamma.get(a, 0.0) + lam[j]
        best, best_key = None, None
        for a, g in gamma.items():
            if 1e-6 < g < 1 - 1e-6:
                key = (abs(g - 0.5), a[0] == 0 or a[1] == 0)
                if best_key is None or key < best_key:
                    best, best_key = a, key
        return best, best is not None

    def _inject_columns(self, bctx: BranchCtx) -> List[dict]:
        """注入分支兼容的种子列（单客户 + SAME 链段），防 Big-M 激活。"""
        out = []
        for i in range(1, self.N + 1):
            r = [0, i, 0]
            if bctx.col_ok(r) and energy_feasible(self.costs, r):
                out.append(make_col(self.costs, r))
        for u, v in bctx.succ.items():
            if u == 0:
                continue        # (0,v): 首客户约束由单客户列自然兼容
            seg = [u]
            x = v
            while x != 0 and x in bctx.succ:
                seg.append(x)
                x = bctx.succ[x]
            if x != 0:
                seg.append(x)
            r = [0] + seg + [0]
            load = sum(int(self.demands_g[i2]) for i2 in seg)
            # [路线 2] 注入列同样过电量门: 保证列池 ⊆ Ω_E 无例外
            if (load <= self.Q and bctx.col_ok(r)
                    and energy_feasible(self.costs, r)):
                out.append(make_col(self.costs, r))
        return out


def _arcs_of(route: List[int]) -> set:
    return {(route[t], route[t + 1]) for t in range(len(route) - 1)}


def _col_ok(bctx: BranchCtx, col: dict) -> bool:
    return bctx.col_ok(col["route"])


# ==============================================================================
# 7. 入口冒烟测试
# ==============================================================================
if __name__ == "__main__":
    import pathlib
    gis = pathlib.Path(GIS_DIR)
    data = json.load(open(gis / "shenzhen_futian_60nodes.json", encoding="utf-8"))
    tensor_file = gis / "arc_phys_table_shenzhen_futian_opt.npy"
    if tensor_file.exists():
        T = np.load(tensor_file)
        print(f"物理张量已就绪: {T.shape}")
    else:
        legacy = gis / "arc_e_table_shenzhen_futian.npy"
        print("物理张量未就绪，暂用旧 LB 张量做机制冒烟（非认证数值）")
        T = np.load(legacy)
    eng = BPEngine(T, data["customers"], K=5, time_budget_s=120.0)
    res = eng.solve()
    print(json.dumps({k: v for k, v in res.items() if k != "routes"},
                     indent=2, ensure_ascii=False))


# ===== 前向 Pareto 支配 DP 内核 (原 _frontier_numba.py, 2026-10-02 并入) =====
@njit(cache=True, fastmath=False)
def _kernel_elem(Mt, TRANS, pi, sigma, dem_g, arc_ok, close_allow, end_allow,
                N, NS, Gf, cap, mi_self, insert_cap, work_cap,
                top_idx, KE):
    """Pecin 式部分初等 ng-DP: top-KE 高|π|用户全局禁重访 (状态维 e ∈ 2^KE).

    e 的位 = top_idx 里的序号。加客户 j: 若 j 在 top 集内, 需对应 e 位=0, 置 1。
    支配: (g升序, c严格降) 且 e1 ⊇ e2 (访问更多 = 更受限 = 更强)。
    [路线 2 注] 本内核为诊断/奇偶校验入口, 维持能量不感知 (超集松弛, 界仍
    有效); 与 _kernel_v2 的逐位 parity 仅在 E_max=inf 时成立。生产认证走
    _kernel_v2 (带电量剪枝)。
    """
    nE = 1 << KE
    fg = np.full((N + 1, NS, nE, cap), -1, dtype=np.int32)
    fc = np.zeros((N + 1, NS, nE, cap), dtype=np.float64)
    flen = np.zeros((N + 1, NS, nE), dtype=np.int32)
    dex = np.zeros((N + 1, NS, nE), dtype=np.int32)
    touched = np.zeros(N + 1, dtype=np.bool_)
    tg = np.zeros(cap + 2, dtype=np.int32)
    tc = np.zeros(cap + 2, dtype=np.float64)
    work_h = np.zeros(work_cap, dtype=np.int32)
    work_m = np.zeros(work_cap, dtype=np.int32)
    work_e = np.zeros(work_cap, dtype=np.int32)
    head = 0
    tail = 0
    wsize = 0
    inserts = 0
    expansions = 0
    overflow = 0
    aborted = 0
    maxlen = 0
    # 客户 j -> top 位序号 (不在 top 集则为 -1)
    top_pos = np.full(N + 1, -1, dtype=np.int32)
    for t in range(KE):
        top_pos[top_idx[t]] = t

    for j in range(1, N + 1):
        if not end_allow[j]:
            continue
        a0 = Mt[0, 0, j]
        if not np.isfinite(a0):
            continue
        g0 = dem_g[j]
        if g0 > Gf:
            continue
        mi = mi_self[j]
        pj = top_pos[j]
        e0 = 0 if pj < 0 else (1 << pj)
        fg[j, mi, e0, 0] = g0
        fc[j, mi, e0, 0] = a0 - pi[j - 1]
        flen[j, mi, e0] = 1
        touched[j] = True
        inserts += 1
        work_h[tail] = j
        work_m[tail] = mi
        work_e[tail] = e0
        tail = (tail + 1) % work_cap
        wsize += 1

    while wsize > 0:
        if inserts > insert_cap:
            aborted = 1
            break
        h = work_h[head]
        mi = work_m[head]
        e = work_e[head]
        head = (head + 1) % work_cap
        wsize -= 1
        n = flen[h, mi, e]
        st = dex[h, mi, e]
        if st >= n:
            continue
        dex[h, mi, e] = n
        for idx in range(st, n):
            g = fg[h, mi, e, idx]
            c = fc[h, mi, e, idx]
            expansions += 1
            for k in range(1, N + 1):
                if not arc_ok[h, k]:
                    continue
                pk = top_pos[k]
                if pk >= 0 and (e >> pk) & 1:
                    continue                     # 重要用户已访问 => 禁重访
                e2 = e
                if pk >= 0:
                    e2 = e | (1 << pk)
                mi2 = TRANS[h, mi, k]
                if mi2 >= NS:
                    continue
                g3 = g + dem_g[k]
                if g3 > Gf:
                    continue
                a = Mt[h, g, k]
                if not np.isfinite(a):
                    continue
                c3 = c + a - pi[k - 1]
                ln = flen[k, mi2, e2]
                dom = False
                for t in range(ln):
                    if fg[k, mi2, e2, t] <= g3 and fc[k, mi2, e2, t] <= c3:
                        dom = True
                        break
                if dom:
                    continue
                cnt = 0
                for t in range(ln):
                    gg = fg[k, mi2, e2, t]
                    cc = fc[k, mi2, e2, t]
                    if gg >= g3 and cc >= c3:
                        continue
                    tg[cnt] = gg
                    tc[cnt] = cc
                    cnt += 1
                if cnt + 1 > cap:
                    overflow += 1
                    continue
                pos = cnt
                for t in range(cnt):
                    if tg[t] > g3:
                        pos = t
                        break
                for t in range(cnt - 1, pos - 1, -1):
                    tg[t + 1] = tg[t]
                    tc[t + 1] = tc[t]
                tg[pos] = g3
                tc[pos] = c3
                cnt += 1
                for t in range(cnt):
                    fg[k, mi2, e2, t] = tg[t]
                    fc[k, mi2, e2, t] = tc[t]
                flen[k, mi2, e2] = cnt
                if cnt > maxlen:
                    maxlen = cnt
                if pos < dex[k, mi2, e2]:
                    dex[k, mi2, e2] = pos
                touched[k] = True
                inserts += 1
                if wsize < work_cap:
                    work_h[tail] = k
                    work_m[tail] = mi2
                    work_e[tail] = e2
                    tail = (tail + 1) % work_cap
                    wsize += 1

    mu = np.inf
    states = 0
    for h in range(1, N + 1):
        if not touched[h]:
            continue
        if not close_allow[h]:
            continue
        for mi in range(NS):
            for e in range(nE):
                ln = flen[h, mi, e]
                if ln == 0:
                    continue
                states += 1
                for t in range(ln):
                    a = Mt[h, fg[h, mi, e, t], 0]
                    if np.isfinite(a):
                        v = fc[h, mi, e, t] + a - sigma
                        if v < mu:
                            mu = v
    return mu, states, inserts, expansions, overflow, aborted, maxlen


@njit(cache=True, fastmath=False)
def _kernel_v2(Mt, TRANS, pi, sigma, dem_g, arc_ok, close_allow, end_allow,
               N, NS, Gf, cap, mi_self, insert_cap, work_cap, E_max):
    """[路线 2] 前向 Pareto 标号 DP, 标签三维 (g, rc, E): E 为沿途真实弧成本
    (=能耗, 细网格插值 ≤ 精确值 ⇒ 能量剪枝不丢真可行走线), E > E_max 即剪枝,
    支配判据与闭合收口均带 E 维。E_max = inf 时逐位退化为能量不感知版。"""
    fg = np.full((N + 1, NS, cap), -1, dtype=np.int32)
    fc = np.zeros((N + 1, NS, cap), dtype=np.float64)
    fe = np.zeros((N + 1, NS, cap), dtype=np.float64)
    flen = np.zeros((N + 1, NS), dtype=np.int32)
    dex = np.zeros((N + 1, NS), dtype=np.int32)
    touched = np.zeros(N + 1, dtype=np.bool_)
    tg = np.zeros(cap + 2, dtype=np.int32)
    tc = np.zeros(cap + 2, dtype=np.float64)
    te = np.zeros(cap + 2, dtype=np.float64)
    work_h = np.zeros(work_cap, dtype=np.int32)
    work_m = np.zeros(work_cap, dtype=np.int32)
    head = 0
    tail = 0
    wsize = 0
    inserts = 0
    expansions = 0
    overflow = 0
    aborted = 0
    maxlen = 0

    for j in range(1, N + 1):
        if not end_allow[j]:
            continue
        a0 = Mt[0, 0, j]
        if not np.isfinite(a0):
            continue
        g0 = dem_g[j]
        if g0 > Gf:
            continue
        mi = mi_self[j]
        fg[j, mi, 0] = g0
        fc[j, mi, 0] = a0 - pi[j - 1]
        fe[j, mi, 0] = a0
        flen[j, mi] = 1
        touched[j] = True
        inserts += 1
        work_h[tail] = j
        work_m[tail] = mi
        tail = (tail + 1) % work_cap
        wsize += 1

    while wsize > 0:
        if inserts > insert_cap:
            aborted = 1
            break
        h = work_h[head]
        mi = work_m[head]
        head = (head + 1) % work_cap
        wsize -= 1
        n = flen[h, mi]
        st = dex[h, mi]
        if st >= n:
            continue
        dex[h, mi] = n
        do_close = close_allow[h]
        for idx in range(st, n):
            g = fg[h, mi, idx]
            c = fc[h, mi, idx]
            e = fe[h, mi, idx]
            expansions += 1
            for k in range(1, N + 1):
                if not arc_ok[h, k]:
                    continue
                mi2 = TRANS[h, mi, k]
                if mi2 >= NS:
                    continue
                g3 = g + dem_g[k]
                if g3 > Gf:
                    continue
                a = Mt[h, g, k]
                if not np.isfinite(a):
                    continue
                e3 = e + a
                if e3 > E_max:
                    continue              # [路线 2] 电量剪枝 (能量单调增长)
                c3 = c + a - pi[k - 1]
                # ---- 支配判定: 存在 (gg<=g3 且 cc<=c3 且 ee<=e3) 则丢弃 ----
                ln = flen[k, mi2]
                dom = False
                for t in range(ln):
                    if (fg[k, mi2, t] <= g3 and fc[k, mi2, t] <= c3
                            and fe[k, mi2, t] <= e3):
                        dom = True
                        break
                if dom:
                    continue
                # ---- 局部缓冲重建: 剔除被新标签支配者, 再按 g 升序插入 ----
                cnt = 0
                for t in range(ln):
                    gg = fg[k, mi2, t]
                    cc = fc[k, mi2, t]
                    ee = fe[k, mi2, t]
                    if gg >= g3 and cc >= c3 and ee >= e3:
                        continue          # 被新标签支配
                    tg[cnt] = gg
                    tc[cnt] = cc
                    te[cnt] = ee
                    cnt += 1
                if cnt + 1 > cap:
                    overflow += 1
                    continue
                pos = cnt
                for t in range(cnt):
                    if tg[t] > g3:
                        pos = t
                        break
                for t in range(cnt - 1, pos - 1, -1):
                    tg[t + 1] = tg[t]
                    tc[t + 1] = tc[t]
                    te[t + 1] = te[t]
                tg[pos] = g3
                tc[pos] = c3
                te[pos] = e3
                cnt += 1
                for t in range(cnt):
                    fg[k, mi2, t] = tg[t]
                    fc[k, mi2, t] = tc[t]
                    fe[k, mi2, t] = te[t]
                flen[k, mi2] = cnt
                if cnt > maxlen:
                    maxlen = cnt
                if pos < dex[k, mi2]:
                    dex[k, mi2] = pos
                touched[k] = True
                inserts += 1
                if wsize < work_cap:
                    work_h[tail] = k
                    work_m[tail] = mi2
                    tail = (tail + 1) % work_cap
                    wsize += 1

    mu = np.inf
    states = 0
    ch = 0
    cmi = 0
    ct = 0
    for h in range(1, N + 1):
        if not touched[h]:
            continue
        do_close = close_allow[h]
        for mi in range(NS):
            ln = flen[h, mi]
            if ln == 0:
                continue
            states += 1
            if not do_close:
                continue
            for t in range(ln):
                a = Mt[h, fg[h, mi, t], 0]
                if np.isfinite(a):
                    # [路线 2] 闭合能量门: 回场弧并入后总能耗超限的走线
                    # 不进入 μ̂ 候选 (细网格能耗 ≤ 精确值 ⇒ 不丢真可行走线)
                    if fe[h, mi, t] + a > E_max:
                        continue
                    v = fc[h, mi, t] + a - sigma
                    if v < mu:
                        mu = v
                        ch = h
                        cmi = mi
                        ct = t
    return (mu, states, inserts, expansions, overflow, aborted, maxlen,
            fg, fc, flen, ch, cmi, ct)


def frontier_mu_elem(eng, pi, sigma, bctx, n_bits, KE=6, cap=48,
                     work_cap=4_000_000, insert_cap=20_000_000):
    """Pecin 式部分初等 ng-DP: top-KE 高|π|用户全局禁重访.

    KE=0 退化为 _kernel_v2 (parity 自检入口)。
    """
    pricer = eng.pricer
    N, GF = eng.N, pricer.G_FINE
    Mx = pricer._fine_cost_table()
    nb = pricer.cert_nb_from_duals(pi, n_bits)
    TRANS, mi_self, NS = pricer._dp_tables(n_bits, nb)
    b = bctx
    close_allow = np.array([b.close_ok(h) for h in range(N + 1)], dtype=np.bool_)
    end_allow = np.zeros(N + 1, dtype=np.bool_)
    end_allow[1:] = np.array([b.end_ok(j) for j in range(1, N + 1)], dtype=np.bool_)
    step_g = pricer.Q / GF
    dem_g = np.minimum((pricer.dem // step_g).astype(int), GF).astype(np.int64)
    Mt = np.ascontiguousarray(np.ascontiguousarray(
        np.transpose(Mx, (1, 2, 0))).astype(np.float64))
    top_idx = (np.argsort(-np.abs(pi))[:KE] + 1).astype(np.int64)
    out = _kernel_elem(Mt, np.ascontiguousarray(TRANS.astype(np.int64)),
                       np.ascontiguousarray(pi.astype(np.float64)), float(sigma),
                       dem_g, np.ascontiguousarray(b.arc_mask_matrix()),
                       close_allow, end_allow, N, NS, GF, cap,
                       np.ascontiguousarray(mi_self.astype(np.int64)),
                       insert_cap, work_cap,
                       np.ascontiguousarray(top_idx), int(KE))
    mu, states, inserts, expansions, overflow, aborted, maxlen = out
    return mu, dict(states=int(states), inserts=int(inserts),
                    expansions=int(expansions), overflow=int(overflow),
                    aborted=int(aborted), maxlen=int(maxlen), cap=cap, KE=int(KE))


def _recover_route(pi, sigma, bctx, TRANS, NS, dem_g, Mt, fg, fc, flen, ch, cmi, ct):
    """从 frontier 内核的最终 Pareto 数组向后重放 argmin 关闭状态, 回收见证走线。

    不存父指针: 在桶 (hp, mp) 中按 (gprev, cprev) 反查父标签 (g 为整数精确;
    c 允许 1e-9 相对容差补偿浮点累积)。链断则返回 None (best-effort)。
    返回 route=[0,...,0] 或 None; 走线可为非初等 (由调用方判定)。
    """
    N = Mt.shape[0] - 1
    arc_ok = bctx.arc_mask_matrix()
    pi = np.asarray(pi, dtype=np.float64)
    if ch == 0:
        return None
    h, mi = int(ch), int(cmi)
    g, c = int(fg[h, mi, ct]), float(fc[h, mi, ct])
    rev = [h]
    for _ in range(N + 2):
        # 种子: 直接由车场出发 (0 -> h)
        if arc_ok[0, h] and np.isfinite(Mt[0, 0, h]) and g == int(dem_g[h]):
            seed_c = float(Mt[0, 0, h]) - pi[h - 1]
            if abs(seed_c - c) <= 1e-9 * max(1.0, abs(c)):
                # 内核按弧向反向建走线 (Mt[h,g,k]=弧 k->h), 故 rev 即真实航线
                return [0] + rev + [0]
        gprev = g - int(dem_g[h])
        if gprev < 0:
            return None
        cprev = c + pi[h - 1]
        nxt = None
        for hp in range(1, N + 1):
            if hp == h or not arc_ok[hp, h]:
                continue
            a = Mt[hp, gprev, h]
            if not np.isfinite(a):
                continue
            need = cprev - a
            for mp in np.nonzero(TRANS[hp, :, h] == mi)[0]:
                ln = int(flen[hp, mp])
                if ln == 0:
                    continue
                for t in np.nonzero(fg[hp, mp, :ln] == gprev)[0]:
                    if abs(float(fc[hp, mp, t]) - need) <= 1e-9 * max(1.0, abs(need)):
                        nxt = (int(hp), int(mp), need)
                        break
                if nxt is not None:
                    break
            if nxt is not None:
                break
        if nxt is None:
            return None
        h, mi, c = nxt
        g = gprev
        rev.append(h)
    return None


def _route_rc(Mx, route, dem_g, pi, sigma):
    """真实航线方向的检验数: 弧 u->v 载重 = v 及其后缀需求 (含 v 包裹, pre-release)。

    Mx 为细网格成本表, 源先索引 Mx[u, v, load] (与 costs.cost 同约定)。
    """
    pi = np.asarray(pi, dtype=np.float64)
    body = route[1:-1]
    m = len(body)
    suffix = [0] * (m + 1)
    for t in range(m - 1, -1, -1):
        suffix[t] = suffix[t + 1] + int(dem_g[body[t]])
    rc = 0.0
    prev = 0
    load = suffix[0]
    for t, u in enumerate(body):
        rc += float(Mx[prev, u, load]) - pi[u - 1]
        load = suffix[t + 1]
        prev = u
    rc += float(Mx[prev, 0, 0]) - float(sigma)
    return rc


def _max_members(nb, N):
    return max(bin(int(nb[h])).count("1") for h in range(1, N + 1))


def _breach_reinforce(nb, walk, N):
    """ng 记忆逐步遗忘 (mi2=(mi∩N(k))∪{k}) ⇒ 堵长程重访须把重访客户 c 加固进
    中间**所有**节点 walk[t1+1 .. t2-1] 的邻域 (只加直接前驱无效)。返回加固位数。"""
    seen = {}
    changed = 0
    for t, u in enumerate(walk):
        if u == 0:
            continue
        if u in seen:
            for p in walk[seen[u] + 1:t]:
                if p == 0:
                    continue
                if not (int(nb[p]) >> (u - 1)) & 1:
                    nb[p] = int(nb[p]) | (1 << (u - 1))
                    changed += 1
        else:
            seen[u] = t
    return changed


def frontier_mu_cod(eng, pi, sigma, bctx, n_bits, max_rounds=3, cap=200,
                    work_cap=4_000_000, insert_cap=20_000_000, n_bits_max=12):
    """认证预言机双职定价 (COD): 见证回收 + 破口驱动 ng-邻域最小增广。

    每轮重解得到单调不降的 μ̂ (加固=收缩可行走线集), 直到见证走线初等或
    邻域位数顶棚 (n_bits_max, 受 2^n_bits 状态空间内存约束)。
    返回 (mu, info); mu 始终是有效 Farley 证书 (ng 超集论证对任何邻域成立)。
    """
    N = eng.N
    pricer = eng.pricer
    nb = [int(x) for x in pricer.cert_nb_from_duals(pi, n_bits)]
    n_eff = int(n_bits)
    mu = None
    info = None
    rounds = 0
    for _ in range(max_rounds + 1):
        mu, _stats = frontier_mu_numba(eng, pi, sigma, bctx, n_eff, cap=cap,
                                       work_cap=work_cap, insert_cap=insert_cap,
                                       nb_override=np.array(nb, dtype=object),
                                       recover=True)
        info = pricer._last_recovery
        if info is None or info["elementary"]:
            break
        if _breach_reinforce(nb, info["route"], N) == 0:
            break
        mx = _max_members(nb, N)
        if mx > n_bits_max:
            break
        n_eff = max(n_eff, mx)
        rounds += 1
    if info is not None:
        info["cod_rounds"] = rounds
        info["cod_n_bits"] = n_eff
    return mu, info


def frontier_mu_numba(eng, pi, sigma, bctx, n_bits, cap=160, work_cap=4_000_000,
                      insert_cap=20_000_000, nb_override=None, recover=False):
    pricer = eng.pricer
    N, GF = eng.N, pricer.G_FINE
    Mx = pricer._fine_cost_table()
    nb = pricer.cert_nb_from_duals(pi, n_bits) if nb_override is None else nb_override
    TRANS, mi_self, NS = pricer._dp_tables(n_bits, nb)
    b = bctx
    close_allow = np.array([b.close_ok(h) for h in range(N + 1)], dtype=np.bool_)
    end_allow = np.zeros(N + 1, dtype=np.bool_)
    end_allow[1:] = np.array([b.end_ok(j) for j in range(1, N + 1)], dtype=np.bool_)
    step_g = pricer.Q / GF
    dem_g = np.minimum((pricer.dem // step_g).astype(int), GF).astype(np.int64)
    Mt = np.ascontiguousarray(np.ascontiguousarray(
        np.transpose(Mx, (1, 2, 0))).astype(np.float64))
    def _run_kernel(cap_x):
        return _kernel_v2(Mt, np.ascontiguousarray(TRANS.astype(np.int64)),
                          np.ascontiguousarray(pi.astype(np.float64)),
                          float(sigma), dem_g,
                          np.ascontiguousarray(b.arc_mask_matrix()),
                          close_allow, end_allow, N, NS, GF, cap_x,
                          np.ascontiguousarray(mi_self.astype(np.int64)),
                          eff_insert_cap, work_cap, E_MAX_J)

    # [路线 2 标签爆炸止损] 能量维使桶内 Pareto 由 (g,rc) 变 (g,rc,E), GF=1020
    # 下三维前沿规模爆炸: sh_90/sz_120 实测单次调用烧满 20M 插入量 (~90-240 s)
    # 才止动, 预算内列生成迭代坍缩至个位数, LB 停留在原始对偶。统一规则
    # (不分城市与规模): 止动阈值 2M (~2-5 s) 快速回退 legacy@126 (能量不感知
    # 超集松弛, 界仍有效; 90 档实测回退主导下 LB 与旧系统持平或更优)。
    eff_insert_cap = 2_000_000
    out = _run_kernel(cap)
    (mu, states, inserts, expansions, overflow, aborted, maxlen,
     fg, fc, flen, ch, cmi, ct) = out
    # [路线 2 桶溢出处置] 能量维削弱桶内 (g,rc) 支配 ⇒ 标签密度上升, 有限桶帽
    # 可能溢出。部分标签集的最小值会高于真值 (μ̂ 下界方向性失效), 绝不可直接
    # 采用。处置链: (1) 桶溢出 ⇒ 4× 升帽重试一次 (内存瞬时 ~125 MB, 可承受);
    # (2) 仍溢出或总插入量止动 ⇒ 回退 legacy 后向 DP (能量不感知超集松弛,
    # μ̂ ≤ min_{Ω_E} rc 方向恒成立, 界偏松但永不失效; SZ90 实测无此处置时
    # Farley 全程归零)。两者皆败才回退 -inf (Farley 棘轮不取该候选)。
    if overflow > 0 and not aborted:
        out = _run_kernel(min(cap * 4, 1600))
        (mu, states, inserts, expansions, overflow, aborted, maxlen,
         fg, fc, flen, ch, cmi, ct) = out
    if overflow > 0 or aborted:
        pricer._last_recovery = None
        try:
            # legacy 回退在 126 粗格执行 (1020 细格单次 30-60 s, 逐迭代回退会
            # 挤占列生成预算, 复现在环耦合病理); 粗格 μ̂ 同为合法下界, 仅偏松。
            g0, m0 = pricer.G_FINE, getattr(pricer, "_fine_M", None)
            if not hasattr(pricer, "_fine_M_by_G"):
                pricer._fine_M_by_G = {}
            pricer.G_FINE = 126
            pricer._fine_M = pricer._fine_M_by_G.get(126)
            try:
                mu_leg, _nn = pricer.exact_ng4_dp(pi, sigma, bctx,
                                                  n_bits=n_bits,
                                                  nb_override=nb_override)
            finally:
                if getattr(pricer, "_fine_M", None) is not None:
                    pricer._fine_M_by_G[126] = pricer._fine_M
                pricer.G_FINE, pricer._fine_M = g0, m0
            return float(mu_leg), dict(
                states=int(states), inserts=int(inserts),
                expansions=int(expansions), overflow=int(overflow),
                aborted=int(aborted), maxlen=int(maxlen), cap=cap,
                fallback="legacy126")
        except Exception:
            return -math.inf, dict(states=int(states), inserts=int(inserts),
                                   expansions=int(expansions),
                                   overflow=int(overflow), aborted=int(aborted),
                                   maxlen=int(maxlen), cap=cap, fused=True)
    if recover:
        route = _recover_route(pi, sigma, b, np.asarray(TRANS), NS, dem_g,
                               Mt, fg, fc, flen, ch, cmi, ct)
        if route is not None:
            body = route[1:-1]
            elem = len(set(body)) == len(body)
            rc = _route_rc(Mx, route, dem_g, pi, sigma)
            pricer._last_recovery = {"route": route, "elementary": bool(elem),
                                     "dp_rc": float(rc), "mu": float(mu)}
        else:
            pricer._last_recovery = None
    return mu, dict(states=int(states), inserts=int(inserts),
                    expansions=int(expansions), overflow=int(overflow),
                    aborted=int(aborted), maxlen=int(maxlen), cap=cap)
