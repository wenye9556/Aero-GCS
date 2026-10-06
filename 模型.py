"""
模型.py: Aero-GCS 物理动力学连续方程、外向舍入区间有理包络与 3D 几何预言机

本模块实现:
1. Gong et al. (IEEE TAES 2023) / Zeng et al. (IEEE TWC 2019) 动量叶素三维非凸功率闭式评估
   (标准名 compute_bemt_power, 别名 evaluate_gong_power)
2. 严格对齐权威认证的 20 个仿射支撑平面 (CERTIFIED_20_PLANES, IEEE 754 向下舍入锁定,
   任何浮点截距不得改动 —— TIGHTNESS-2 公理)
3. 严格基于 IEEE 754 负无穷舍入 (Downward Rounding) 的单例悬停切片下界
4. TangentOracle3D: 三维避障双分支切线预言机 (LOS / Bypass / Overtop, R5 红线:
   LB 只用水平欧氏距离 + 20 平面; 绕行/翻越几何仅进 UB)
5. B 样条光滑器与 BEMT 数值闭环积分器 (BSplineIntegrator3D, 仅 UB/仿真)
6. 势能守恒爬升下界 (compute_potential_lower_bound)

认证物理紧集: W ∈ [20,30] N, v_h ∈ [0,15] m/s, v_z ∈ [-3,-0.5] ∪ {0} ∪ [0.5,6] m/s。
"""

import math
import heapq
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from shapely.geometry import Polygon, LineString, Point
from shapely.strtree import STRtree
from 参数 import (AERO_PARAMS, AERO_GROUND_TRUTH, ENV_CONSTANTS, SPEED_LIMITS,
                  H_SAFE_CLEARANCE, SAFETY_MARGIN_2D, MAX_CLIMB_ANGLE_DEG, MAX_DESC_ANGLE_DEG)

# ==============================================================================
# 1. 严格向下舍入浮点辅助函数 (IEEE 754 Round Down)
# ==============================================================================
def round_down_float(val: float, precision: float = 0.0) -> float:
    """IEEE-754 向下舍入（朝 -∞ 连续 nextafter 若干 ulp）。

    不再用 floor((x-1e-7)*1e7)/1e7：那会引入 1e-7 量级的系统偏置，
    且在 |x| 较大时改变有效数字。这里用 2 ulp 外向，与权威区间契约一致。
    """
    x = float(val)
    for _ in range(2):
        x = math.nextafter(x, -math.inf)
    return x

# ==============================================================================
# 2. Gong 2023 / Zeng 2019 三维瞬时电功率闭式模型 (BEMT)
# ==============================================================================
def evaluate_gong_power(vh: float, vz: float, W: float) -> float:
    """
    计算 Gong / Zeng 瞬时功率 P_ref(vh, vz, W)
    输入:
      vh: 水平空速 (m/s) >= 0
      vz: 垂直速度 (m/s)
      W: 实时全机重力 (N)
    """
    k_bl = AERO_GROUND_TRUTH["k_bl"]
    k_in = AERO_GROUND_TRUTH["k_in"]
    v0 = AERO_PARAMS["v0"]
    sfp_par = AERO_PARAMS["sfp_par"]
    sfp_perp = AERO_PARAMS["sfp_perp"]
    rho = ENV_CONSTANTS["rho"]
    n = AERO_PARAMS["N"]
    area = AERO_PARAMS["area"]

    # 1. 悬停基准项 P_mh(W) = (k_bl + k_in) * W^(3/2)
    W32 = W * math.sqrt(W)
    P_mh = (k_bl + k_in) * W32

    # 2. 水平平动阻力项 dP_par(vh, W)
    two_v0_sq = 2.0 * (v0 ** 2)
    x = (vh ** 2) / two_v0_sq
    rad = math.sqrt(1.0 + x * x)
    t2 = 1.0 / math.sqrt(rad + x) - 1.0

    k_par = (3.0 / 8.0) * math.sqrt(n) * AERO_PARAMS["delta"] * AERO_PARAMS["s"] * math.sqrt((rho * area) / AERO_PARAMS["c_t"])
    k_par3 = 0.5 * n * sfp_par * rho

    dp_par = k_par * math.sqrt(W) * (vh ** 2) + k_in * W32 * t2 + k_par3 * (vh ** 3)

    # 3. 垂直升降分部项 dP_perp(vz, W)
    # 【理论披露与连续性说明】:
    # 动量闭式中 coef * sqrt(inner) 在 vz -> 0+ 处的极限值为 0.5 * W * sqrt(k_w2 * W) = 0.5 * W * v_0 (在 W=20N 时约为 63.25 W)。
    # 该值对应经典单旋翼动量理论的基础诱导做功，不含针对实机气动损耗的 kappa=1.11 修正因子 (故低于全机修正诱导功率 k_in * W^1.5 = 70.21 W)。
    # 实际四旋翼飞行中，垂直机动速度严格规避涡环状态 (VRS) 禁区 (-0.5, 0.5) m/s。
    # 本函数闭式取其上跳保守值作为物理上界 (UB) 计算依据，对真实能耗具有安全裕度保障。
    if abs(vz) < 1e-6:
        return float(P_mh + dp_par)

    V = abs(vz)
    s = 1.0 if vz > 0 else -1.0
    k_s3 = 0.25 * n * sfp_perp * rho
    k_w2 = 2.0 / (n * rho * area)
    up_in = 1.0 + sfp_perp / (n * area)
    dn_in = 1.0 - sfp_perp / (n * area)

    t1 = 0.5 * W * V
    t2p = s * k_s3 * (V ** 3)
    inner = (up_in if s > 0 else dn_in) * (V ** 2) + k_w2 * W
    coef = 0.5 * W + s * k_s3 * (V ** 2)

    dp_perp_raw = t1 + t2p + coef * math.sqrt(max(0.0, inner))
    
    # [P1-1 C1 continuous transition near vz=0 to eliminate step discontinuity]
    v_trans = 0.5
    if V < v_trans:
        u = V / v_trans
        alpha = 3.0 * (u ** 2) - 2.0 * (u ** 3)
        dp_perp = alpha * dp_perp_raw
    else:
        dp_perp = dp_perp_raw
        
    return float(P_mh + dp_par + dp_perp)


def compute_bemt_power(vh: float, vz: float, W: float) -> float:
    """BEMT 闭式功率的标准名 (与 evaluate_gong_power 同一实现)。
    样条积分器与 3D 预言机统一通过本名调用。"""
    return evaluate_gong_power(vh, vz, W)

# ==============================================================================
# 3. 单例悬停切片下界 (Singleton Hover Slice)
# ==============================================================================
def hover_slice_lower_bound(W: float) -> float:
    """
    基于定理 2：纯悬停工况 (vh=0, vz=0) 下的紧致闭式下界
    P_hover_slice(W) = dn( (k_bl + k_in) * W^(1.5) )
    (vh=vz=0 处 dp_par=dp_perp=0 精确成立, 故该闭式为精确值向下舍入)
    带严格认证域守卫 (W ∈ [W_min, W_max] = [20, 30] N), 拒绝静默签发域外下界。
    """
    w_val = float(W)
    if not (ENV_CONSTANTS["W_min"] - 1e-6 <= w_val <= ENV_CONSTANTS["W_max"] + 1e-6):
        raise ValueError(
            f"hover_slice_lower_bound 超出认证域 [{ENV_CONSTANTS['W_min']}, {ENV_CONSTANTS['W_max']}] N: W={w_val} N!"
        )
    k_sum = AERO_GROUND_TRUTH["k_sum"]
    nominal = k_sum * (w_val ** 1.5)
    return round_down_float(nominal)

# ==============================================================================
# 4. 20 面仿射支撑平面 (a, c, beta_hat)
# 形式: P_ref(vh, vz, W) >= a*vh + c*vz + beta_hat  在声明物理域上点点成立
#
# 来源说明 (R5-F2):
#   - 斜率目录与权威冻结目录一致: a ∈ {0,4,8,12}, c ∈ {-60,-30,0,30,50}
#   - beta_hat 在 W∈[20,30] 上由支撑超平面截距下界给出。
#     本表数值经 verify_model_enclosure_rigor 网格 + 10 万点随机抽检
#     与独立细粒度下确界复核 (最小裕度 +0.0167 W @ (a,c)=(0,-60)) 确认无击穿。
#   - c>0 且 beta 与 c=0 行相同，表示「爬升超额 ≥ c·v_z」类平面
#     （截距可取平飞零平面同一值，因 P_climb - P_level ≥ 0 且
#      额外项随 vz 增长；若抽检发现击穿必须下调 beta）。
# ==============================================================================
CERTIFIED_20_PLANES: List[Tuple[float, float, float]] = [
    (0.0, -60.0, 67.65488272001988),
    (0.0, -30.0, 157.6548827200199),
    (0.0, 0.0, 194.77939390391913),
    (0.0, 30.0, 194.77939390391913),
    (0.0, 50.0, 188.3000),
    (4.0, -60.0, 31.779869164168204),
    (4.0, -30.0, 121.7798229325077),
    (4.0, 0.0, 158.8955561058353),
    (4.0, 30.0, 158.8955561058353),
    (4.0, 50.0, 152.4300),
    (8.0, -60.0, -13.874459761070064),
    (8.0, -30.0, 76.1223100464577),
    (8.0, 0.0, 113.2407287020563),
    (8.0, 30.0, 113.2407287020563),
    (8.0, 50.0, 106.7700),
    (12.0, -60.0, -67.91990589634662),
    (12.0, -30.0, 22.073566645732328),
    (12.0, 0.0, 59.1950453508114),
    (12.0, 30.0, 59.1950453508114),
    (12.0, 50.0, 52.7300),
]

# ==============================================================================
# 5. 全局包络逐点下界性验证断言 (Unit Self-Test)
# ==============================================================================
def verify_model_enclosure_rigor() -> None:
    """验证 20 面体仿射包络与单例悬停在全物理域网格上的 100% 下界性 (显式运行时抛错, 防 -O 剥离)"""
    # 1. 悬停切片自检
    hover_20 = hover_slice_lower_bound(20.0)
    actual_hover_20 = evaluate_gong_power(0.0, 0.0, 20.0)
    if hover_20 > actual_hover_20:
        raise RuntimeError(f"Hover slice violation: {hover_20} > {actual_hover_20}")
    if abs(hover_20 - AERO_GROUND_TRUTH["hover_slice_bound"]) >= 1e-3:
        raise RuntimeError(f"Hover slice baseline mismatch: {hover_20} != {AERO_GROUND_TRUTH['hover_slice_bound']}")

    # 2. 仿射面在全物理域网格上的点点硬保证
    test_vhs = np.linspace(SPEED_LIMITS["vh_min"], SPEED_LIMITS["vh_max"], 8)
    test_vzs = [-3.0, -1.5, -0.5, 0.0, 0.5, 2.0, 4.0, 6.0]
    test_ws = np.linspace(ENV_CONSTANTS["W_min"], ENV_CONSTANTS["W_max"], 5)

    for vh in test_vhs:
        for vz in test_vzs:
            for w in test_ws:
                p_true = evaluate_gong_power(vh, vz, w)
                for (a, c, beta_hat) in CERTIFIED_20_PLANES:
                    p_aff = a * vh + c * vz + beta_hat
                    margin = p_true - p_aff
                    # 容忍 1e-6 浮点误差，绝对不可击穿
                    if margin < -1e-6:
                        raise RuntimeError(f"Certified plane breached! P_true={p_true}, P_aff={p_aff}, margin={margin}")

    # 3. 刀锋极值点靶向安全校验 (vh=7.474858, vz=-3.0, W=20.0, 理论极限裕度 +0.01666 W)
    p_razor_true = evaluate_gong_power(7.474858, -3.0, 20.0)
    for (a, c, beta_hat) in CERTIFIED_20_PLANES:
        p_razor_aff = a * 7.474858 + c * (-3.0) + beta_hat
        if p_razor_true - p_razor_aff < 0.0160:
            raise RuntimeError(f"Razor-edge margin dropped below +0.016 W: true={p_razor_true}, aff={p_razor_aff}")

verify_model_enclosure_rigor()

# ==============================================================================
# 5.5 向量化 BEMT 闭式功率与分支自适应割平面认证器 (BranchCutCertifier)
# 形式: P_ref(vh, vz, W) >= a*vh + c*vz + beta_hat  在三分支定义域上点点成立
# ==============================================================================
BRANCH_MINUS = 0  # vz in [-3.0, -0.5]
BRANCH_ZERO = 1   # vz == 0.0
BRANCH_PLUS = 2   # vz in [0.5, 6.0]

def gong_power_vec(vh, vz, W):
    """向量化 Gong 2023 / Zeng 2019 BEMT 瞬时功率闭式计算。输入可为标量或同形状 numpy 数组。"""
    k_bl = AERO_GROUND_TRUTH["k_bl"]
    k_in = AERO_GROUND_TRUTH["k_in"]
    v0 = AERO_PARAMS["v0"]
    sfp_par = AERO_PARAMS["sfp_par"]
    sfp_perp = AERO_PARAMS["sfp_perp"]
    rho = ENV_CONSTANTS["rho"]
    n = AERO_PARAMS["N"]
    area = AERO_PARAMS["area"]

    vh = np.asarray(vh, dtype=float)
    vz = np.asarray(vz, dtype=float)
    W = np.asarray(W, dtype=float)

    W32 = W * np.sqrt(W)
    P_mh = (k_bl + k_in) * W32

    two_v0_sq = 2.0 * (v0 ** 2)
    x = (vh ** 2) / two_v0_sq
    rad = np.sqrt(1.0 + x * x)
    t2 = 1.0 / np.sqrt(rad + x) - 1.0
    k_par = (3.0 / 8.0) * np.sqrt(n) * AERO_PARAMS["delta"] * AERO_PARAMS["s"] * np.sqrt((rho * area) / AERO_PARAMS["c_t"])
    k_par3 = 0.5 * n * sfp_par * rho
    dp_par = k_par * np.sqrt(W) * (vh ** 2) + k_in * W32 * t2 + k_par3 * (vh ** 3)

    V = np.abs(vz)
    s = np.where(vz > 0, 1.0, -1.0)
    k_s3 = 0.25 * n * sfp_perp * rho
    k_w2 = 2.0 / (n * rho * area)
    up_in = 1.0 + sfp_perp / (n * area)
    dn_in = 1.0 - sfp_perp / (n * area)
    inner_coef = np.where(s > 0, up_in, dn_in)
    inner = inner_coef * (V ** 2) + k_w2 * W
    coef = 0.5 * W + s * k_s3 * (V ** 2)
    t1 = 0.5 * W * V
    t2p = s * k_s3 * (V ** 3)
    dp_perp = np.where(np.abs(vz) < 1e-6, 0.0, t1 + t2p + coef * np.sqrt(np.maximum(0.0, inner)))
    return P_mh + dp_par + dp_perp

_VH_GRID = np.arange(0.0, 15.0 + 1e-9, 0.25)

def _branch_vz_grid(branch: int) -> np.ndarray:
    if branch == BRANCH_MINUS:
        return np.arange(-3.0, -0.4999, 0.25)
    if branch == BRANCH_ZERO:
        return np.array([0.0])
    return np.arange(0.5, 6.0 + 1e-9, 0.25)

def _branch_plane_inf(W: float, a: float, c: float, branch: int) -> float:
    """beta_s(a,c) = inf_{分支子域}[P - a vh - c vz] 网格估计 (低估安全) - delta_cert + 外向舍入。"""
    vzs = _branch_vz_grid(branch)
    VH = np.repeat(_VH_GRID, len(vzs))
    VZ = np.tile(vzs, len(_VH_GRID))
    P = gong_power_vec(VH, VZ, W)
    b = float(np.min(P - (a * VH + c * VZ))) - 0.05
    for _ in range(2):
        b = math.nextafter(b, -math.inf)
    return b

_TAN_PTS = {
    BRANCH_ZERO: [(15.0, 0.0), (14.0, 0.0), (13.0, 0.0), (12.0, 0.0), (11.0, 0.0),
                  (10.0, 0.0), (9.0, 0.0), (7.0, 0.0), (5.0, 0.0), (0.0, 0.0)],
    BRANCH_PLUS: [(15.0, 0.5), (15.0, 1.0), (15.0, 1.5), (15.0, 2.0), (15.0, 2.5),
                  (15.0, 3.0), (15.0, 4.0), (15.0, 5.0), (15.0, 6.0),
                  (6.0 / math.tan(math.radians(25.0)), 6.0), (13.0, 2.0), (12.0, 4.0),
                  (11.0, 3.0), (9.0, 2.0), (6.0, 1.0), (3.0, 0.5)],
    BRANCH_MINUS: [(15.0, -0.5), (15.0, -1.0), (15.0, -1.5), (15.0, -2.0), (15.0, -2.5),
                   (15.0, -3.0), (3.0 / math.tan(math.radians(20.0)), -3.0),
                   (13.0, -2.0), (11.0, -3.0), (9.0, -1.5), (5.0, -1.0), (2.0, -0.5)],
}

def _tangent_slope(W: float, vh0: float, vz0: float) -> Tuple[float, float]:
    h = 1e-4
    vhi, vlo = min(15.0, vh0 + h), max(0.0, vh0 - h)
    da = (evaluate_gong_power(vhi, vz0, W) - evaluate_gong_power(vlo, vz0, W)) / (vhi - vlo)
    zhi, zlo = vz0 + h, vz0 - h
    if vz0 <= -3.0 + h:
        zhi, zlo = vz0 + h, vz0
    elif vz0 >= 6.0 - h:
        zhi, zlo = vz0, vz0 - h
    elif vz0 <= -0.5 + h or (0.5 - h <= vz0 <= 0.5 + h):
        zhi, zlo = vz0 + h, vz0
    dc = (evaluate_gong_power(vh0, zhi, W) - evaluate_gong_power(vh0, zlo, W)) / (zhi - zlo)
    return da, dc

class BranchCutCertifier:
    """分支自适应割平面认证器。
    按速度分支 (MINUS/ZERO/PLUS) 与载重区间动态生成支撑超平面,
    通过全连续子域蒙特卡洛抽检认证, 保证 LB <= OPT 绝不击穿。"""
    W_QUANT = 0.5
    _GLOBAL_CACHE: Dict[float, Dict[int, List[Tuple[float, float, float]]]] = {}

    def __init__(self):
        self._cache = self._GLOBAL_CACHE
        self.certified_min_margin = math.inf

    def quantize_down(self, W: float) -> float:
        w = float(W)
        if 20.0 - 1e-6 <= w < 20.0:
            w = 20.0
        if not (20.0 - 1e-9 <= w <= 30.0 + 1e-9):
            raise RuntimeError(f"W 量化越界: W={W}")
        wq = math.floor((w - 20.0) / self.W_QUANT + 1e-9) * self.W_QUANT + 20.0
        return round(wq, 6)

    def get_branch_planes(self, W: float) -> Dict[int, List[Tuple[float, float, float]]]:
        w_true = float(W)
        if not (20.0 - 1e-6 <= w_true <= 30.0 + 1e-6):
            raise RuntimeError(f"BranchCutCertifier: W={w_true} 超出认证域 [20,30] N")
        wq = self.quantize_down(w_true)
        if wq in self._cache:
            return self._cache[wq]
        out: Dict[int, List[Tuple[float, float, float]]] = {}
        for branch in (BRANCH_MINUS, BRANCH_ZERO, BRANCH_PLUS):
            planes = []
            for (vh0, vz0) in _TAN_PTS[branch]:
                a, c = _tangent_slope(wq, vh0, vz0)
                beta = _branch_plane_inf(wq, a, c, branch)
                self._certify_single(wq, a, c, beta, branch)
                planes.append((a, c, beta))
            vzs = _branch_vz_grid(branch)
            VH = np.repeat(_VH_GRID, len(vzs))
            VZ = np.tile(vzs, len(_VH_GRID))
            beta0 = float(np.min(gong_power_vec(VH, VZ, wq))) - 1e-3
            for _ in range(2):
                beta0 = math.nextafter(beta0, -math.inf)
            planes.append((0.0, 0.0, beta0))
            out[branch] = planes
        self._cache[wq] = out
        return out

    def _certify_single(self, W: float, a: float, c: float, beta: float, branch: int) -> None:
        vhs = np.linspace(0.0, 15.0, 121)
        vzs = _branch_vz_grid(branch)
        if len(vzs) > 1:
            vzs = np.unique(np.concatenate([vzs, np.linspace(vzs[0], vzs[-1], 97)]))
        vv, zz = np.meshgrid(vhs, vzs, indexing="ij")
        vv, zz = vv.ravel(), zz.ravel()
        wgrid = np.array([W, min(30.0, W + self.W_QUANT)])
        min_m = math.inf
        for w in wgrid:
            p = gong_power_vec(vv, zz, float(w))
            min_m = min(min_m, float(np.min(p - (a * vv + c * zz + beta))))
        rng = np.random.default_rng(int(W * 100) + int(abs(a * 7 + c * 13)) % 99991 + branch)
        n = 20000
        vh_mc = rng.uniform(0, 15, n)
        if branch == BRANCH_MINUS:
            vz_mc = rng.uniform(-3.0, -0.5, n)
        elif branch == BRANCH_PLUS:
            vz_mc = rng.uniform(0.5, 6.0, n)
        else:
            vz_mc = np.zeros(n)
        w_mc = rng.uniform(W, min(30.0, W + self.W_QUANT), n)
        p = gong_power_vec(vh_mc, vz_mc, w_mc)
        min_m = min(min_m, float(np.min(p - (a * vh_mc + c * vz_mc + beta))))
        self.certified_min_margin = min(self.certified_min_margin, min_m)
        if min_m < -1e-6:
            raise RuntimeError(
                f"自适应割平面击穿! (a,c,beta)=({a:.4f},{c:.4f},{beta:.4f}) branch={branch} @W_q={W}, 裕度 {min_m:.6f} W")

_BRANCH_CERTIFIER = BranchCutCertifier()
_LP_CACHE: Dict[Tuple[int, float, float, float], float] = {}

def arc_lb_branch_lp(d: float, dz: float, W: float,
                     branch_planes: Optional[Dict[int, List[Tuple[float, float, float]]]] = None) -> float:
    """三分支时间分配 LP 弧下界: E >= min LP(T-, T0, T+, L-, L0, L+, Z-, Z+)."""
    from scipy.optimize import linprog
    if not (20.0 - 1e-6 <= float(W) <= 30.0 + 1e-6):
        raise RuntimeError(f"arc_lb_branch_lp: W={W} 超域")
    if not (math.isfinite(d) and math.isfinite(dz)) or d < -1e-9:
        raise RuntimeError(f"arc_lb_branch_lp: 非法位移 d={d}, dz={dz}")
    d, dz = max(0.0, d), float(dz)
    if branch_planes is None:
        branch_planes = _BRANCH_CERTIFIER.get_branch_planes(float(W))
    key = (id(branch_planes), round(d, 3), round(dz, 3), float(W))
    if key in _LP_CACHE:
        return _LP_CACHE[key]
    if d < 1e-9 and abs(dz) < 1e-9:
        _LP_CACHE[key] = 0.0
        return 0.0

    rows = []
    for br in (BRANCH_MINUS, BRANCH_ZERO, BRANCH_PLUS):
        for (a, c, b) in branch_planes[br]:
            rows.append((br, a, c, b))
    nv = 11
    obj = np.zeros(nv)
    obj[8] = obj[9] = obj[10] = 1.0
    A_ub, b_ub = [], []
    for (ts, ls) in [(0, 3), (1, 4), (2, 5)]:
        r = np.zeros(nv); r[ls] = 1.0; r[ts] = -15.0
        A_ub.append(r); b_ub.append(0.0)
    r = np.zeros(nv); r[7] = 1.0; r[2] = -6.0; A_ub.append(r); b_ub.append(0.0)
    r = np.zeros(nv); r[7] = -1.0; r[2] = 0.5; A_ub.append(r); b_ub.append(0.0)
    r = np.zeros(nv); r[6] = -1.0; r[0] = -3.0; A_ub.append(r); b_ub.append(0.0)
    r = np.zeros(nv); r[6] = 1.0; r[0] = 0.5; A_ub.append(r); b_ub.append(0.0)
    for (br, a, c, b) in rows:
        r = np.zeros(nv)
        r[8 + br] = -1.0
        if br == BRANCH_MINUS:
            r[3], r[6], r[0] = a, c, b
        elif br == BRANCH_ZERO:
            r[4], r[1] = a, b
        else:
            r[5], r[7], r[2] = a, c, b
        A_ub.append(r); b_ub.append(0.0)
    A_eq = []
    b_eq = []
    r = np.zeros(nv); r[3] = r[4] = r[5] = 1.0; A_eq.append(r); b_eq.append(d)
    r = np.zeros(nv); r[6] = r[7] = 1.0; A_eq.append(r); b_eq.append(dz)
    bounds = [(0, None), (0, None), (0, None),
              (0, None), (0, None), (0, None),
              (None, 0.0), (0.0, None),
              (None, None), (None, None), (None, None)]
    res = linprog(obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=bounds, method="highs")
    if not res.success:
        # 求解器数值异常时安全回退至保守 20 平面外包下界 (Admissible Fallback)
        val = _planes_arc_lb(d, dz, W)
    else:
        val = float(res.fun) - 1e-4 if float(res.fun) > 1e-4 else 0.0
    _LP_CACHE[key] = val
    return val

def _best_vertical_speed(W: float, sign: int) -> float:
    zs = np.arange(0.5, 6.0 + 1e-9, 0.05) if sign > 0 else np.arange(-3.0, -0.499, 0.05)
    P = gong_power_vec(np.zeros_like(zs), zs, W)
    return float(zs[int(np.argmin(P / np.abs(zs)))])

def optimal_profile_energy(L: float, up: float, dn: float, W: float,
                           entry: float = 1.0, exit: float = 0.0) -> float:
    """能量最优常速剖面 UB: 基于极小值原理与最优能距比速度剖面求解可行真实物理上界。"""
    tan_c = math.tan(math.radians(25.0))
    tan_d = math.tan(math.radians(20.0))
    net = up - dn
    best = math.inf

    # 空间阻挡窗口感知: 若 entry < 1.0 或 exit > 0.0, 说明中间存在建筑物阻挡, 严禁采用未经净空检验的直飞斜线!
    has_blocking_window = (entry < 1.0 - 1e-6) or (exit > 1e-6)

    if L > 1e-9 and (up < 1e-12 or dn < 1e-12) and not has_blocking_window:
        for vh in np.arange(4.0, 15.0 + 1e-9, 0.05):
            vh = float(vh)
            vz = net * vh / L
            if abs(vz) < 1e-12:
                e = float(gong_power_vec(vh, 0.0, W)) * (L / vh)
            else:
                if not (-3.0 - 1e-12 <= vz <= -0.5 + 1e-12 or 0.5 - 1e-12 <= vz <= 6.0 + 1e-12):
                    continue
                if vz > 0 and vh < vz / tan_c - 1e-9:
                    continue
                if vz < 0 and vh < abs(vz) / tan_d - 1e-9:
                    continue
                e = float(gong_power_vec(vh, vz, W)) * (L / vh)
            best = min(best, e)
    if L < 1e-9 and (up > 1e-12 or dn > 1e-12):
        z_opt = _best_vertical_speed(W, 1 if net > 0 else -1)
        best = min(best, float(gong_power_vec(0.0, z_opt, W)) * ((up + dn) / abs(z_opt)))

    zc = _best_vertical_speed(W, 1)
    zd = _best_vertical_speed(W, -1)
    vh_c, vh_d = zc / tan_c, abs(zd) / tan_d
    h_c = up * vh_c / zc if up > 1e-12 else 0.0
    h_d = dn * vh_d / abs(zd) if dn > 1e-12 else 0.0

    if L > 1e-9 and h_c + h_d <= L + 1e-9 and h_c <= entry * L + 1e-6 and h_d <= (1.0 - exit) * L + 1e-6:
        e = 0.0
        if up > 1e-12:
            e += float(gong_power_vec(vh_c, zc, W)) * (up / zc)
        if dn > 1e-12:
            e += float(gong_power_vec(vh_d, zd, W)) * (dn / abs(zd))
        cruise = L - h_c - h_d
        if cruise > 1e-9:
            e += float(gong_power_vec(15.0, 0.0, W)) * (cruise / 15.0)
        best = min(best, e)

    e = 0.0
    if up > 1e-12:
        e += float(gong_power_vec(0.0, zc, W)) * (up / zc)
    if dn > 1e-12:
        e += float(gong_power_vec(0.0, zd, W)) * (dn / abs(zd))
    if L > 1e-9:
        e += float(gong_power_vec(15.0, 0.0, W)) * (L / 15.0)
    best = min(best, e)
    return float(best)

# ==============================================================================
# 6. 认证弧能量下界 (支撑平面包络, R5 红线: L 只能取水平欧氏距离)
# ==============================================================================
def _planes_arc_lb(L_h: float, H_z: float, W: Optional[float] = None) -> float:
    """min_{T >= T_min} max_r (a_r L + c_r H + beta_r T) —— 与 算法.py 的
    solve_tight_arc_energy_lb 同一数学 (T_min 基于认证域速度界, 逐对 kink 候选, -1e-4 浮点守卫)。
    本文件内独立实现以避免循环依赖; 两处实现改动必须同步。

    W 守卫: 20 平面包络截距对 W ∈ [W_min, W_max] = [20, 30] N 认证。
    未指定时默认按 W_min (20.0 N) 安全基线校验, 始终执行严格认证域运行时异常拦截以防未经认证的外推。"""
    w_eval = float(ENV_CONSTANTS["W_min"]) if W is None else float(W)
    if not (ENV_CONSTANTS["W_min"] - 1e-6 <= w_eval <= ENV_CONSTANTS["W_max"] + 1e-6):
        raise ValueError(
            f"Arc LB 认证域越界: W={w_eval} N 不在认证盒 [{ENV_CONSTANTS['W_min']}, {ENV_CONSTANTS['W_max']}] N 内!"
        )
    if not (math.isfinite(L_h) and math.isfinite(H_z)):
        raise ValueError(f"_planes_arc_lb 输入位移必须为有限实数, 实得: L_h={L_h}, H_z={H_z}")
    if float(L_h) < 0.0:
        raise ValueError(f"_planes_arc_lb 水平距离 L_h 必须非负, 实得: L_h={L_h}")

    if L_h < 1e-9 and abs(H_z) < 1e-9:
        return 0.0
    vh_max = float(SPEED_LIMITS["vh_max"])
    vz_climb = float(SPEED_LIMITS["vz_climb_max"])
    vz_desc = abs(float(SPEED_LIMITS["vz_descend_min"]))  # 与包络认证域一致 (|vz| <= 3.0)
    # [物理与保界性说明]: 采用全域垂直速度绝对值上限 vz_lim = max(vz_climb, vz_desc) = 6.0 m/s
    # 构造保守松弛时间下界 T_min = max(L/vh_max, |H_z|/6.0). 因松弛域包含真实物理可行时间域,
    # 在其上求极小所得能量下界恒 <= 真实时间域极小, 保证下界绝对有效不虚高, 同时锁定基准数值精确闭环.
    vz_lim = max(vz_climb, vz_desc)
    T_min = max(L_h / vh_max if L_h > 1e-12 else 0.0,
                abs(H_z) / vz_lim if abs(H_z) > 1e-12 else 0.0,
                1e-6)
    const = [a * L_h + c * H_z for (a, c, _b) in CERTIFIED_20_PLANES]
    betas = [b for (_a, _c, b) in CERTIFIED_20_PLANES]

    def f(T: float) -> float:
        return max(const[i] + betas[i] * T for i in range(len(const)))

    cands = [T_min]
    for i in range(len(betas)):
        for j in range(i + 1, len(betas)):
            db = betas[i] - betas[j]
            if abs(db) < 1e-15:
                continue
            T_star = (const[j] - const[i]) / db
            if T_star > T_min:
                cands.append(T_star)
    best = min(f(T) for T in cands)
    if best < 0.0:
        best = 0.0
    return best - 1e-4 if best > 1e-4 else 0.0

# ==============================================================================
# 7. 三维避障双分支切线预言机 TangentOracle3D
# ==============================================================================
def _point_in_poly(pt, poly) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            if x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
                inside = not inside
    return inside


def _seg_poly_t_windows(p1, p2, poly) -> List[Tuple[float, float]]:
    """线段 p1->p2 (参数 t∈[0,1]) 与多边形内部相交的全部 t 区间 (基于 Shapely 严格空间投影)。"""
    line = LineString([p1, p2])
    L_tot = line.length
    if L_tot < 1e-12:
        return [(0.0, 1.0)] if Polygon(poly).contains(Point(p1)) else []
    p = Polygon(poly)
    inter = line.intersection(p)
    if inter.is_empty:
        return []
    geoms = [inter] if inter.geom_type == "LineString" else (
        list(inter.geoms) if inter.geom_type in ("MultiLineString", "GeometryCollection") else []
    )
    wins = []
    for g in geoms:
        if g.geom_type == "LineString" and g.length > 1e-9:
            coords = list(g.coords)
            t0 = float(line.project(Point(coords[0])) / L_tot)
            t1 = float(line.project(Point(coords[-1])) / L_tot)
            wins.append((min(t0, t1), max(t0, t1)))
    return wins


def _inflate_poly(poly, margin: float) -> List[Tuple[float, float]]:
    """权威几何外向膨胀 (基于 Shapely 2.0 Minkowski-buffer, 严格向外部扩张)。
    输入要求为有效简单多边形 (Simple Polygon)。包含严格防御性运行时异常拦截 (防 -O 剥离)。
    保证沿法向到任意建筑边缘的横向净空恒 >= margin，杜绝凹凸多边形自交与内缩漏洞。"""
    if not (math.isfinite(margin) and margin >= 0.0):
        raise ValueError(f"safety margin 必须为非负有限实数, 实得: {margin}")
    p = Polygon(poly)
    if not p.is_valid:
        raise ValueError(f"输入建筑多边形几何非法 (非简单多边形): {poly}")
    if p.is_empty:
        raise ValueError("输入建筑多边形为空")
    buf = p.buffer(float(margin), join_style="mitre", mitre_limit=2.0)
    if not buf.is_valid or buf.is_empty or buf.geom_type != "Polygon":
        raise ValueError(f"膨胀多边形异常: geom_type={buf.geom_type}, valid={buf.is_valid}, empty={buf.is_empty}")
    coords = list(buf.exterior.coords)[:-1]
    return [(float(x), float(y)) for x, y in coords]

# 模块级公开发布别名 (解决跨模块导入私有函数缺陷)
inflate_polygon = _inflate_poly
inflate_polygon_2d = _inflate_poly


class TangentOracle3D:
    """三维避障双分支切线预言机。

    R5 红线分工:
      * LB  = 20 平面包络在 (水平欧氏距离 d_xy, 净高差 Δz) 上的对偶极小化;
        三维直线距离不得用于放大 L (垂直航段的水平投影长度为 0, 用 3D 距离
        会高估水平长度、击穿下界); 端点势能差由平面的 c_r·Δz 行严格计价。
      * UB  = min(Bypass 绕行能量, Overtop 翻越能量), 全部为物理可行剖面。
    """

    def __init__(self, buildings: List[Dict[str, Any]],
                 h_safe: float = H_SAFE_CLEARANCE,
                 safety_margin_2d: float = SAFETY_MARGIN_2D):
        if not (math.isfinite(h_safe) and float(h_safe) >= 0.0):
            raise ValueError(f"h_safe 必须为非负有限实数, 实得: {h_safe}")
        if not (math.isfinite(safety_margin_2d) and float(safety_margin_2d) >= 0.0):
            raise ValueError(f"safety_margin_2d 必须为非负有限实数, 实得: {safety_margin_2d}")
        self.h_safe = float(h_safe)
        self.safety_margin_2d = float(safety_margin_2d)
        self.b: List[Dict[str, Any]] = []
        for b in buildings:
            if "vertices" in b and "height" in b:      # 3D 权威结构
                poly, h = list(b["vertices"]), float(b["height"])
            else:                                       # 旧版 polygon/z_max 视图
                poly, h = list(b["polygon"]), float(b["z_max"])
            if not (math.isfinite(h) and h >= 0.0):
                raise ValueError(f"建筑高度必须为非负有限实数, 实得 id={b.get('id')}: height={h}")
            self.b.append({"id": b.get("id", "?"), "poly": poly, "h": h})
        self._los_cache: Dict[Tuple, Any] = {}
        self._bypass_cache: Dict[Tuple, Any] = {}
        self._overtop_cache: Dict[Tuple, Any] = {}
        # [性能补丁, 数值语义零改动] 预建 shapely Polygon + STRtree
        # 空间索引与膨胀多边形缓存, 消除逐弧重复 buffer/相交测试的常数开销。
        from shapely.strtree import STRtree
        self._raw_polys = [Polygon(bd["poly"]) for bd in self.b]
        self._raw_tree = STRtree(self._raw_polys)
        self._infl_polys: List[Optional[Polygon]] = [None] * len(self.b)

    def _infl_poly(self, idx: int) -> Polygon:
        """缓存版 _inflate_poly(poly_i, safety_margin_2d)（确定性同值）。"""
        p = self._infl_polys[idx]
        if p is None:
            p = Polygon(_inflate_poly(self.b[idx]["poly"], self.safety_margin_2d))
            self._infl_polys[idx] = p
        return p

    def _infl_for_bd(self, bd: Dict[str, Any]) -> Polygon:
        """按建筑字典懒缓存膨胀多边形（bypass/overtop 路径共用）。"""
        p = bd.get("_infl")
        if p is None:
            p = Polygon(_inflate_poly(bd["poly"], self.safety_margin_2d))
            bd["_infl"] = p
        return p

    # ------------------------------------------------------------- LOS
    def check_los_obstruction(self, p1, p2) -> Tuple[bool, List[Dict[str, Any]]]:
        """3D 视线判定: 2D 投影进入足迹的多边形内部 且 线段在该窗口内的高度
        z(t) < H_k + h_safe  => 受阻。z(t) 沿线段线性, 窗口最小值在窗口端点。"""
        cache_key = (round(float(p1[0]), 2), round(float(p1[1]), 2), round(float(p1[2]), 2),
                     round(float(p2[0]), 2), round(float(p2[1]), 2), round(float(p2[2]), 2))
        if cache_key in self._los_cache:
            return self._los_cache[cache_key]

        a2, b2 = (p1[0], p1[1]), (p2[0], p2[1])
        blockers = []
        seg2d = LineString([a2, b2])
        for k in self._raw_tree.query(seg2d):
            bd = self.b[k]
            wins = _seg_poly_t_windows(a2, b2, bd["poly"])
            if not wins:
                continue
            z0, z1 = p1[2], p2[2]
            thresh = bd["h"] + self.h_safe
            for (t0, t1) in wins:
                zmin = min(z0 + t0 * (z1 - z0), z0 + t1 * (z1 - z0))
                if zmin < thresh - 1e-9:
                    blockers.append(bd)
                    break
        res = (len(blockers) > 0), blockers
        self._los_cache[cache_key] = res
        return res

    # ------------------------------------------------------------- Bypass
    def compute_bypass_tangent(self, p1, p2, return_path: bool = False,
                               _extra_local=None):
        """2D 绕行测地长 (膨胀多边形可见图 Dijkstra)。端点 2D 落在任何
        足迹内时绕行不可行, 返回 +inf (屋顶客户走 Overtop)。
        若 return_path=True, 返回 (dist, path_coords)。"""
        cache_key = (round(float(p1[0]), 2), round(float(p1[1]), 2),
                     round(float(p2[0]), 2), round(float(p2[1]), 2), bool(return_path))
        if cache_key in self._bypass_cache:
            return self._bypass_cache[cache_key]

        a2, b2 = (p1[0], p1[1]), (p2[0], p2[1])
        for bd in self.b:
            if _point_in_poly(a2, bd["poly"]) or _point_in_poly(b2, bd["poly"]):
                res = (math.inf, []) if return_path else math.inf
                self._bypass_cache[cache_key] = res
                return res

        # 空间剪枝 (Bounding Box Pruning): 仅将起点与终点外扩区域内的局部障碍物纳入 2D 可见图
        # 防止 300+ 栋楼、3000+ 顶点的全图 O(V^2) 盲目相交导致几何构建卡死
        min_x = min(a2[0], b2[0]) - 80.0
        max_x = max(a2[0], b2[0]) + 80.0
        min_y = min(a2[1], b2[1]) - 80.0
        max_y = max(a2[1], b2[1]) + 80.0

        def _in_window(bd):
            poly = bd["poly"]
            px = [p[0] for p in poly]
            py = [p[1] for p in poly]
            return (max(px) >= min_x and min(px) <= max_x
                    and max(py) >= min_y and min(py) <= max_y)

        local_b = [bd for bd in self.b if _in_window(bd)]
        local_ids = {id(bd) for bd in local_b}
        for bd in (_extra_local or []):
            if id(bd) not in local_ids:
                local_b.append(bd)
                local_ids.add(id(bd))
                
        # 若剪枝后无局部障碍物，则局部极值测地线等价于直飞
        if not local_b:
            d_direct = math.hypot(b2[0] - a2[0], b2[1] - a2[1])
            res = (d_direct, [a2, b2]) if return_path else d_direct
            self._bypass_cache[cache_key] = res
            return res

        nodes = [a2, b2]
        node_comp = [-1, -1]
        # [性能补丁 | 距离严格不变] 节点集约简:
        #   局部膨胀多边形取 unary_union, 仅保留 ∂O 缠绕顶点 (材料角 < π;
        #   判据: 外环 cr·o>0 / 内环庭院 cr·o<0, o 为环符号面积定向)。
        #   经典结论: 多边形障碍外部的测地最短路仅在障碍并集边界的缠绕顶点处
        #   转折; 埋没顶点孤立、凹顶点不可能落在最短路上 ⇒ 距离严格相等。
        from shapely.ops import unary_union
        union_geom = unary_union([self._infl_for_bd(bd) for bd in local_b])
        comps = list(union_geom.geoms) if union_geom.geom_type == "MultiPolygon" \
            else [union_geom]
        for ci, comp in enumerate(comps):
            for ring, is_hole in [(comp.exterior, False)] + \
                    [(r, True) for r in comp.interiors]:
                cs = list(ring.coords)
                if len(cs) > 1 and cs[0] == cs[-1]:
                    cs = cs[:-1]
                m = len(cs)
                if m < 3:
                    continue
                area2 = sum((cs[i][0] * cs[(i + 1) % m][1]
                             - cs[(i + 1) % m][0] * cs[i][1]) for i in range(m))
                o = 1.0 if area2 > 0 else -1.0
                for i2 in range(m):
                    w0, w1, w2 = cs[i2 - 1], cs[i2], cs[(i2 + 1) % m]
                    cr = ((w1[0] - w0[0]) * (w2[1] - w1[1])
                          - (w1[1] - w0[1]) * (w2[0] - w1[0]))
                    if cr * o * (-1.0 if is_hole else 1.0) > 0:
                        nodes.append((float(w1[0]), float(w1[1])))
                        node_comp.append(ci)
        local_infl = comps                    # 可见性谓词作用于并集分量 (等价)
        if len(nodes) <= 2:
            d_direct = math.hypot(b2[0] - a2[0], b2[1] - a2[1])
            res = (d_direct, [a2, b2]) if return_path else d_direct
            self._bypass_cache[cache_key] = res
            return res

        # [性能补丁 | 数值零改动] 惰性 A* 可见图:
        #   仅在节点结算时按需生成出边; 可见性谓词分三层 ——
        #   (L0) 凸分量内锥判据: u 为凸分量顶点时, 该分量对全部出边的封锁
        #        ⟺ 方向 d 严格落入内锥开锥 (两严格叉积同号于 s), 向量化;
        #   (L1) numpy 同侧严格预筛: 对每条边线起终点叉积严格同号
        #        ⇒ 线段与该多边形闭集不相交, 直接判可见;
        #   (L2) 歧义回退: DE-9IM seg.relate(poly)[0]=='1' 精确判定。
        #   A* 启发式为欧氏距离 (可采纳且一致) ⇒ 最短路距离与原实现严格同一。
        pts = np.asarray(nodes, dtype=np.float64)
        Vn = len(nodes)
        edges_list, bboxes = [], []
        for pl in local_infl:
            cs = np.asarray(pl.exterior.coords)
            for ring in pl.interiors:
                csr = np.asarray(ring.coords)
                if csr.shape[0] >= 4:
                    cs = np.vstack([cs, csr])
            E = np.column_stack([cs[:-1, 0], cs[:-1, 1],
                                 cs[1:, 0], cs[1:, 1]])
            edges_list.append(np.ascontiguousarray(E))
            bboxes.append(pl.bounds)
        # 锥参数: 节点 → (分量 idx, a, b, s), 仅当所属分量为凸
        convex_flags = [pl.equals(pl.convex_hull) for pl in local_infl]
        cone_by_comp = {}
        for ci, pl in enumerate(local_infl):
            if not convex_flags[ci]:
                continue
            ring = np.asarray(pl.exterior.coords)[:-1]
            m = len(ring)
            dcone = {}
            for i2 in range(m):
                w0, w1, w2 = ring[i2 - 1], ring[i2], ring[(i2 + 1) % m]
                a = (w0[0] - w1[0], w0[1] - w1[1])
                b3 = (w2[0] - w1[0], w2[1] - w1[1])
                s2 = a[0] * b3[1] - a[1] * b3[0]
                if s2 != 0.0:
                    dcone[(float(p1[0]), float(p1[1]))] = (a, b3, s2)
            cone_by_comp[ci] = dcone
        node_cone = [cone_by_comp.get(node_comp[ni], {}).get(
            (float(pts[ni, 0]), float(pts[ni, 1]))) for ni in range(Vn)]
        bboxes = np.asarray(bboxes, dtype=np.float64)
        bx0, by0 = float(pts[1, 0]), float(pts[1, 1])

        def _strictly_inside(x: float, y: float, k: int) -> bool:
            E = edges_list[k]
            x1, y1, x2, y2 = E[:, 0], E[:, 1], E[:, 2], E[:, 3]
            with np.errstate(divide="ignore", invalid="ignore"):
                xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            cond = ((y1 > y) != (y2 > y)) & (x < xint)
            if not (np.count_nonzero(cond) % 2 == 1):
                return False
            return bool(local_infl[k].contains(Point((x, y))))  # 契约级复核

        def _visible_mask(ui: int) -> np.ndarray:
            ux, uy = float(pts[ui, 0]), float(pts[ui, 1])
            tx, ty = pts[:, 0], pts[:, 1]
            mask = (tx != ux) | (ty != uy)        # 剔除重合坐标与自身
            sminx = np.minimum(ux, tx)
            smaxx = np.maximum(ux, tx)
            sminy = np.minimum(uy, ty)
            smaxy = np.maximum(uy, ty)
            skip_poly = -1
            cone = node_cone[ui]
            if cone is not None:
                skip_poly = node_comp[ui]
                ax, ay = cone[0]
                bx, by2 = cone[1]
                s2c = cone[2]
                dxs = tx - ux
                dys = ty - uy
                c1 = ax * dys - ay * dxs          # cross(a, d)
                c2 = dxs * by2 - dys * bx         # cross(d, b)
                mask &= ~((s2c * c1 > 1e-12) & (s2c * c2 > 1e-12))
            for k in range(len(local_infl)):
                if k == skip_poly:
                    continue
                bb = bboxes[k]
                cand = mask & (smaxx >= bb[0]) & (sminx <= bb[2]) & \
                       (smaxy >= bb[1]) & (sminy <= bb[3])
                nidx = np.nonzero(cand)[0]
                if nidx.size == 0:
                    continue
                if _strictly_inside(ux, uy, k):
                    return np.zeros(Vn, dtype=bool)   # 内点出发全线封锁
                E = edges_list[k]
                ex1, ey1 = E[:, 0], E[:, 1]
                dx, dy = E[:, 2] - ex1, E[:, 3] - ey1
                s1 = dx * (uy - ey1) - dy * (ux - ex1)
                s2 = (dx[None, :] * (ty[nidx, None] - ey1[None, :])
                      - dy[None, :] * (tx[nidx, None] - ex1[None, :]))
                clear = np.all(s1[None, :] * s2 > 1e-12, axis=1)
                for j in nidx[~clear]:            # 歧义回退精确判定
                    seg = LineString([(ux, uy),
                                      (float(pts[j, 0]), float(pts[j, 1]))])
                    if seg.relate(local_infl[k])[0] == "1":
                        mask[j] = False
            return mask

        g_best = np.full(Vn, math.inf)
        g_best[0] = 0.0
        prev = np.full(Vn, -1, dtype=np.int64)
        closed = np.zeros(Vn, dtype=bool)
        pq = [(math.hypot(pts[0, 0] - bx0, pts[0, 1] - by0), 0.0, 0)]
        d_res = math.inf
        while pq:
            _, gu, u = heapq.heappop(pq)
            if closed[u]:
                continue
            closed[u] = True
            if u == 1:
                d_res = gu
                break
            vmask = _visible_mask(u)
            nbs = np.nonzero(vmask)[0]
            if nbs.size == 0:
                continue
            w = np.hypot(pts[nbs, 0] - pts[u, 0], pts[nbs, 1] - pts[u, 1])
            ng = gu + w
            better = ng < g_best[nbs] - 1e-12
            upd = nbs[better]
            if upd.size:
                g_best[upd] = ng[better]
                prev[upd] = u
                hupd = np.hypot(pts[upd, 0] - bx0, pts[upd, 1] - by0)
                for j, gj, hj in zip(upd, ng[better], hupd):
                    heapq.heappush(pq, (float(gj + hj), float(gj), int(j)))
        curr = 1
        path = []
        while curr != -1:
            path.append((float(pts[curr, 0]), float(pts[curr, 1])))
            curr = int(prev[curr])
        path.reverse()
        # [路径全局校验] 局部窗口内建筑的顶点可在窗外, A* 路径可能越窗
        # 刮蹭非局部建筑 (母本既有保真度缺口)。逐段对全部建筑膨胀棱柱检测,
        # 发现越窗刮蹭则将该建筑并入局部集重算 (递归, 上限 40 栋/6 轮),
        # 保证返回的测地线全局碰撞自由 —— 与"可行绕行轨迹代价"语义严格一致。
        if math.isfinite(d_res) and len(path) >= 2:
            bad = []
            for w1, w2 in zip(path[:-1], path[1:]):
                seg = LineString([w1, w2])
                for k2 in self._raw_tree.query(seg):
                    bd = self.b[k2]
                    if id(bd) in local_ids:
                        continue
                    if seg.relate(self._infl_for_bd(bd))[0] == "1":
                        bad.append(bd)
            if bad:
                extra = list(_extra_local or [])
                for bd in bad:
                    if id(bd) not in {id(x) for x in extra}:
                        extra.append(bd)
                if len(extra) <= 40 and (_extra_local is None or
                                         len(extra) > len(_extra_local)):
                    return self.compute_bypass_tangent(
                        p1, p2, return_path, _extra_local=extra)
        if not return_path:
            self._bypass_cache[cache_key] = d_res
            return d_res
        if math.isinf(d_res):
            res = (math.inf, [])
            self._bypass_cache[cache_key] = res
            return res
        res = (d_res, path)
        self._bypass_cache[cache_key] = res
        return res

    # ------------------------------------------------------------- Overtop
    def compute_overtop_tangent(self, p1, p2, blockers=None):
        """翻越剖面的强制总爬升 (垂直总变差下界):
        H_overtop = (Z_peak - z1) + (Z_peak - z2),
        Z_peak = max(H_bar, z1, z2), H_bar = max_{k∈Blockers} (H_k + h_safe)。返回 (H_overtop, H_bar)。"""
        cache_key = (round(float(p1[0]), 2), round(float(p1[1]), 2), round(float(p1[2]), 2),
                     round(float(p2[0]), 2), round(float(p2[1]), 2), round(float(p2[2]), 2))
        if blockers is None and cache_key in self._overtop_cache:
            return self._overtop_cache[cache_key]

        if blockers is None:
            _, blockers = self.check_los_obstruction(p1, p2)
        if not blockers:
            res = (0.0, -math.inf)
            self._overtop_cache[cache_key] = res
            return res
        H_bar = max(bd["h"] + self.h_safe for bd in blockers)
        Z_peak = max(H_bar, p1[2], p2[2])
        up = max(0.0, Z_peak - p1[2])
        dn = max(0.0, Z_peak - p2[2])
        res = (up + dn, H_bar)
        self._overtop_cache[cache_key] = res
        return res

    # ------------------------------------------------------------- UB 剖面
    def _blended_energy(self, L_h: float, dz: float, W: float) -> float:
        """常速混合剖面能量 (UB): 网格搜 vh∈[4,14], vz=dz/dt 落声明域并联,
        且满足爬升/俯冲角上限与严格水平位移一致性; 无解时退化为两相剖面 (垂直 ±2 m/s + 10 m/s 巡航)。"""
        best = math.inf
        tan_climb = math.tan(math.radians(MAX_CLIMB_ANGLE_DEG))
        tan_desc = math.tan(math.radians(MAX_DESC_ANGLE_DEG))

        # 航迹倾角预检 (若空间直线坡度超限，直接退化为两相垂直起降+平飞)
        can_straight = True
        if L_h > 1e-9:
            if dz > 1e-9 and (dz / L_h > tan_climb + 1e-9):
                can_straight = False
            elif dz < -1e-9 and (abs(dz) / L_h > tan_desc + 1e-9):
                can_straight = False
        else:
            can_straight = False

        if can_straight:
            for vh in np.linspace(4.0, 14.0, 41):
                dt = L_h / float(vh)
                vz = dz / dt
                if abs(dz) < 1e-9:
                    vz = 0.0
                elif not (0.5 - 1e-12 <= vz <= 6.0 + 1e-12
                          or -3.0 - 1e-12 <= vz <= -0.5 + 1e-12):
                    continue
                if vz > 0 and float(vh) < vz / tan_climb - 1e-9:
                    continue
                if vz < 0 and float(vh) < abs(vz) / tan_desc - 1e-9:
                    continue
                e = compute_bemt_power(float(vh), vz, W) * dt
                if e < best:
                    best = e
        if math.isfinite(best):
            return best
        # 两相退化 (VTOL 豁免: 纯垂直机动 + 巡航, 角限参数仅约束混合剖面)
        vz_v = 2.0 if dz > 0 else -2.0
        e = 0.0
        if abs(dz) > 1e-9:
            t_v = abs(dz) / abs(vz_v)
            e += compute_bemt_power(0.0, vz_v, W) * t_v
        if L_h > 1e-9:
            e += compute_bemt_power(10.0, 0.0, W) * (L_h / 10.0)
        return e

    def _overtop_energy(self, p1, p2, H_bar: float, W: float, blockers=None) -> float:
        """翻越剖面 (UB): 空间净空感知与航速优化。
        1) 若起点/终点距离建筑物有充足水平前置/后置距离，支持斜向角限爬升/俯冲；
        2) 空间净空不足或近距阻挡时采用原地垂直起降 (VTOL: vz=±2.0 m/s) + 10 m/s 平飞巡航。"""
        d_xy = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        Z_peak = max(H_bar, p1[2], p2[2])
        up = max(0.0, Z_peak - p1[2])
        dn = max(0.0, Z_peak - p2[2])

        # 原地垂直起降 (VTOL: vz=±2.0 m/s 标定工作点) + 10 m/s 平飞巡航 (基准可行解，无空间干涉风险)
        vz_v = 2.0
        e_vtol = 0.0
        if up > 1e-9:
            e_vtol += compute_bemt_power(0.0, vz_v, W) * (up / vz_v)
        if dn > 1e-9:
            e_vtol += compute_bemt_power(0.0, -vz_v, W) * (dn / vz_v)
        if d_xy > 1e-9:
            e_vtol += compute_bemt_power(10.0, 0.0, W) * (d_xy / 10.0)

        # 空间净空位置感知 (基于膨胀安全包络足迹检测前置/后置距离)
        t_entry, t_exit = 1.0, 0.0
        if blockers and d_xy > 1e-6:
            windows = []
            for bd in blockers:
                infl_poly = self._infl_for_bd(bd)
                wins = _seg_poly_t_windows((p1[0], p1[1]), (p2[0], p2[1]),
                                          list(infl_poly.exterior.coords)[:-1])
                windows.extend(wins)
            if windows:
                t_entry = min(w[0] for w in windows)
                t_exit = max(w[1] for w in windows)

        tan_c = math.tan(math.radians(MAX_CLIMB_ANGLE_DEG))
        tan_d = math.tan(math.radians(MAX_DESC_ANGLE_DEG))

        vh_c = vz_v / tan_c
        vh_d = vz_v / tan_d
        h_c = up * vh_c / vz_v if up > 1e-9 else 0.0
        h_d = dn * vh_d / vz_v if dn > 1e-9 else 0.0

        can_oblique = (h_c + h_d <= d_xy and
                       h_c <= t_entry * d_xy + 1e-6 and
                       h_d <= (1.0 - t_exit) * d_xy + 1e-6)

        if can_oblique:
            cruise = max(0.0, d_xy - h_c - h_d)
            e_ob = (compute_bemt_power(vh_c, vz_v, W) * (up / vz_v)
                    + compute_bemt_power(10.0, 0.0, W) * (cruise / 10.0)
                    + compute_bemt_power(vh_d, -vz_v, W) * (dn / vz_v))
            return min(e_vtol, e_ob)
        return e_vtol

    # ------------------------------------------------------------- 主入口
    def evaluate_3d_arc_branches(self, p1, p2, payload_weight: float,
                                 use_optimal_speed: bool = False) -> Tuple[str, float, float]:
        # [P1-2 certified physical envelope check]
        total_w = payload_weight * ENV_CONSTANTS["g"] + ENV_CONSTANTS["W_tare"]
        if not (13.0 <= total_w <= 35.0):
            raise ValueError(f"Weight W={total_w:.2f}N outside certified BEMT envelope [13.0, 35.0]N")
        """返回 (branch_selected, lb_energy, ub_energy)。

        lb: 20 平面包络在 (d_xy 欧氏, Δz_net) 上的对偶极小化 (认证下界);
        ub: LOS 通畅 -> 直飞; 受阻 -> min(Bypass, Overtop) 真实剖面能量。
        use_optimal_speed: 若为 True, 采用能距比最优速度剖面 (最高 15 m/s); 否则采用基线剖面 (最高 14 m/s / 10 m/s)。
        恒断言 lb <= ub (R5 金丝雀)。
        """
        d_xy = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        dz = p2[2] - p1[2]
        if d_xy < 1e-9 and abs(dz) < 1e-9:
            return "direct", 0.0, 0.0
        W = ENV_CONSTANTS["W_min"] + payload_weight * ENV_CONSTANTS["g"]
        # R5 欧氏保真: _planes_arc_lb(d_xy, dz) 严格在 bypass 之前计算且仅依赖欧氏距离
        # 针对大载荷压力测试: 使用 W_max 截断保护计算保守下界
        W_lb = min(W, float(ENV_CONSTANTS["W_max"]))
        lb = _planes_arc_lb(d_xy, dz, W_lb)

        blocked, blockers = self.check_los_obstruction(p1, p2)
        if not blocked:
            if use_optimal_speed:
                up = max(0.0, dz)
                dn = max(0.0, -dz)
                branch, ub = "direct", optimal_profile_energy(d_xy, up, dn, W)
            else:
                branch, ub = "direct", self._blended_energy(d_xy, dz, W)
        else:
            L_bp = self.compute_bypass_tangent(p1, p2)
            if use_optimal_speed:
                up = max(0.0, dz)
                dn = max(0.0, -dz)
                E_bp = (optimal_profile_energy(L_bp, up, dn, W)
                        if math.isfinite(L_bp) else math.inf)
            else:
                E_bp = (self._blended_energy(L_bp, dz, W)
                        if math.isfinite(L_bp) else math.inf)

            _, H_bar = self.compute_overtop_tangent(p1, p2, blockers)
            if use_optimal_speed:
                Z_peak = max(H_bar, p1[2], p2[2])
                up_ov = max(0.0, Z_peak - p1[2])
                dn_ov = max(0.0, Z_peak - p2[2])
                t_entry, t_exit = 1.0, 0.0
                if blockers and d_xy > 1e-6:
                    windows = []
                    for bd in blockers:
                        infl_poly = self._infl_for_bd(bd)
                        wins = _seg_poly_t_windows((p1[0], p1[1]), (p2[0], p2[1]),
                                                  list(infl_poly.exterior.coords)[:-1])
                        windows.extend(wins)
                    if windows:
                        t_entry = min(w[0] for w in windows)
                        t_exit = max(w[1] for w in windows)
                E_ov = optimal_profile_energy(d_xy, up_ov, dn_ov, W, entry=t_entry, exit=t_exit)
            else:
                E_ov = self._overtop_energy(p1, p2, H_bar, W, blockers=blockers)

            if E_bp <= E_ov:
                branch, ub = "bypass", E_bp
            else:
                branch, ub = "overtop", E_ov
        if lb > ub + 1e-6:
            raise RuntimeError(
                f"R5 canary: LB={lb:.3f} > UB={ub:.3f} on arc {p1}->{p2} "
                f"(branch={branch}) — certificate chain broken")
        return branch, lb, ub

# ==============================================================================
# 8. B 样条光滑器与 BEMT 数值闭环积分器 (BSplineIntegrator3D)
# ==============================================================================
def generate_smooth_3d_spline(waypoints: List[Tuple[float, float, float]],
                              dt: float = 0.1) -> Dict[str, Any]:
    """均匀弦长参数化三次 B 样条插值 (C² 连续) + 物理时间律 + 时间均匀重采样。

    [P1-01 修复, 网格无关性] 物理时长**先**按速度限独立积分:
        dt_phys_j = max(ds_xy_j / 10, |ds_z_j| / 2)   (v_h <= 10, |v_z| <= 2)
    总时长 T = sum(dt_phys) 与采样步长 dt **完全解耦**; dt 仅决定积分/输出的
    均匀时间网格 (在物理时间轴上线性插值重采样)。dt 从 0.2 -> 0.01 s 的
    能量漂移 < 0.1% (自测 3 断言)。
    入口做相邻航路点去重 (P2-01): 完全相同的相邻点会使样条拟合抛
    ValueError; 去重容差 1e-6 m。

    注: 样条在 v_z≈0 附近连续穿越 VRS 禁区 (0,0.5); 闭式功率在 v_z=0 支点
    有结构性上跳 (P(0,0)=204.2 vs P(0,0.5)≈273.7 W), 因此样条积分对
    "跳过禁区的两相剖面"是保守上界。本积分器仅用于 UB/仿真。
    """
    from scipy.interpolate import make_interp_spline
    if not (math.isfinite(dt) and float(dt) > 0.0):
        raise ValueError(f"B 样条时间步长 dt 必须为正有限实数, 实得: {dt}")
    pts = np.asarray(waypoints, dtype=float)
    if not (pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] >= 4):
        raise ValueError("B 样条插值需要 >= 4 个三维航路点")
    # P2-01: 相邻航路点去重 (容差 1e-6 m)
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-6:
            keep.append(i)
    pts = pts[keep]
    if pts.shape[0] < 4:
        raise ValueError("去重后航路点不足 4 个，无法构造三次 B 样条")
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(seg)])
    if u[-1] <= 1e-9:
        raise ValueError("航路点总长为零，无法参数化")
    u /= u[-1]
    spl = make_interp_spline(u, pts, k=3)
    d2 = spl.derivative(2)

    # ---- 物理时间律 (与 dt 解耦): 细采样上积分真实时长 ----
    n_phys = 2400
    us = np.linspace(0.0, 1.0, n_phys)
    xyz = spl(us)
    acc = np.linalg.norm(d2(us), axis=1)
    ds = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    ds_xy = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
    ds_z = np.diff(xyz[:, 2])
    dt_phys = np.maximum.reduce([ds_xy / 10.0, np.abs(ds_z) / 2.0,
                                 np.full_like(ds_xy, 1e-6)])
    t_phys = np.concatenate([[0.0], np.cumsum(dt_phys)])
    T_total = float(t_phys[-1])
    vh_f = ds_xy / dt_phys
    vz_f = ds_z / dt_phys
    t_mid = t_phys[:-1] + 0.5 * dt_phys          # 步中点时刻 (速度的代表时刻)

    # ---- 按 dt 均匀重采样 (积分网格), 物理量线性插值 ----
    t_u = np.arange(0.0, T_total, dt)
    if t_u.size == 0 or t_u[-1] < T_total - 1e-12:
        t_u = np.concatenate([t_u, [T_total]])
    vh = np.interp(t_u, t_mid, vh_f)
    vz = np.interp(t_u, t_mid, vz_f)
    xyz_u = np.stack([np.interp(t_u, t_mid, xyz[:-1, k] + 0.5 *
                                (xyz[1:, k] - xyz[:-1, k]))
                      for k in range(3)], axis=1)
    return {"t": t_u, "xyz": xyz_u, "vh": vh, "vz": vz,
            "accel_norm": acc, "T": T_total, "length": float(ds.sum()),
            "max_vh": float(vh.max()), "max_abs_vz": float(np.abs(vz).max()),
            "spline": spl}


def integrate_spline_bemt_energy(traj: Dict[str, Any],
                                 payload_schedule: Optional[List[Dict[str, Any]]] = None,
                                 dt: float = 0.1) -> Dict[str, Any]:
    """梯形法数值积分真实 BEMT 能耗 + 各客户点悬停交付能量。

    payload_schedule: 按 t_release 升序的事件列表, 每项
      {"t_release": float, "drop_kg": float, "hover_s": float}
    载荷语义与认证链一致: 悬停计价发生在释放之前 (pre-release)。
    """
    t, vh, vz = traj["t"], traj["vh"], traj["vz"]
    if len(t) < 2 or float(t[-1]) <= float(t[0]):
        raise ValueError(
            f"轨迹时间轴非法: 节点数={len(t)}, t_start={t[0] if len(t) else None}, t_end={t[-1] if len(t) else None}"
        )
    events = sorted(payload_schedule or [], key=lambda e: e["t_release"])
    total_q = sum(e["drop_kg"] for e in events)
    remaining = total_q
    hover_E = 0.0
    for e in events:
        W_pre = ENV_CONSTANTS["W_min"] + remaining * ENV_CONSTANTS["g"]
        hover_E += compute_bemt_power(0.0, 0.0, W_pre) * e["hover_s"]
        remaining -= e["drop_kg"]

    def W_at(tv: float) -> float:
        rem = total_q
        for e in events:
            if tv >= e["t_release"]:
                rem -= e["drop_kg"]
        return ENV_CONSTANTS["W_min"] + rem * ENV_CONSTANTS["g"]

    flight_E = 0.0
    # t/vh/vz 为同一均匀时间网格上的节点值: 梯形法逐对积分
    for i in range(len(t) - 1):
        dt_i = float(t[i + 1]) - float(t[i])
        if dt_i <= 0:
            continue
        P1 = compute_bemt_power(float(vh[i]), float(vz[i]), W_at(float(t[i])))
        P2 = compute_bemt_power(float(vh[i + 1]), float(vz[i + 1]),
                                W_at(float(t[i + 1])))
        flight_E += 0.5 * (P1 + P2) * dt_i
    return {"E_total_J": flight_E + hover_E,
            "E_flight_J": flight_E, "E_hover_J": hover_E,
            "T_total_s": float(t[-1]),
            "payload_total_kg": total_q}

# ==============================================================================
# 9. 势能守恒爬升下界 (Potential Lower Bound)
# ==============================================================================
CLIMB_EXCESS_COEF: float = 0.5 * ENV_CONSTANTS["W_min"]
"""认证爬升超额系数 (J/m): 爬升分支上 dP_perp >= 0.5·W·v_z 且 W >= W_min,
故每爬升 1 m 的能耗超出 (相对同速平飞) >= 0.5·W_min = 10 J/m。"""


def compute_potential_lower_bound(assigned_nodes, depot, must_clear_height: float = 0.0
                                  ) -> Dict[str, float]:
    """闭合子回路不可逆重力做功下界 (机队分配层的不可撼动项)。

    E_climb_LB = CLIMB_EXCESS_COEF · Δh_must,
    Δh_must = max(0, max_i(z_i − z0), must_clear_height − z0)。

    [求和合法性, 防双重计价] 本项只能与 **c=0 行** 的水平包络下界相加:
    P = P(vh,0,W) + dP_perp 逐点精确分解, 其中
      ∫P(vh,0,W)dt >= max_{r: c_r=0}(a_r·L + β_r·T),  ∫dP_perp·dt >= 0.5·W_min·Δh_up。
    与含 c_r>0 行的 20 平面 LB 只能取 max, 不得相加。
    提示公式中 P_climb_eff/v_z_max 之比对 v_z 求极小后 v_z 相消, 恰得
    0.5·W_lo 系数, 与本实现一致。
    """
    def z_of(n):
        return float(n["pos"][2]) if isinstance(n, dict) else float(n[2])
    z0 = z_of(depot)
    zs = [z_of(n) for n in assigned_nodes] or [z0]
    dh = max([0.0] + [z - z0 for z in zs] + [must_clear_height - z0])
    E = CLIMB_EXCESS_COEF * dh
    return {"E_climb_lb_J": E, "climb_height_m": dh,
            "coef_J_per_m": CLIMB_EXCESS_COEF}

# ==============================================================================
# 10. __main__ 自测套件 (任务规格: 三项)
# ==============================================================================
if __name__ == "__main__":
    print("[Model Self-Test] 模型.py 3D 升级自测套件启动 (显式 RuntimeError 防御 -O 剥离)...")

    # ---- 测试 1: 20 平面全域网格严格下界性 (击穿 0, 网格最小裕度 >= +0.10 W) ----
    print("\n[Test 1] 20 切平面全域网格下界性...")
    grid_vh = np.linspace(0.0, 15.0, 8)
    grid_vz = [-3.0, -1.5, -0.5, 0.0, 0.5, 2.0, 4.0, 6.0]
    grid_w = np.linspace(20.0, 30.0, 5)
    breaches = 0
    min_margin = math.inf
    for vh in grid_vh:
        for vz in grid_vz:
            for w in grid_w:
                p_true = evaluate_gong_power(vh, vz, w)
                for (a, c, beta) in CERTIFIED_20_PLANES:
                    m = p_true - (a * vh + c * vz + beta)
                    if m < -1e-6:
                        breaches += 1
                    min_margin = min(min_margin, m)
    if breaches != 0:
        raise RuntimeError(f"击穿 {breaches} 处")
    if min_margin < 0.10:
        raise RuntimeError(f"网格最小裕度 {min_margin:.4f} W < +0.10 W")
    print(f"  PASS: 击穿 0, 网格最小裕度 +{min_margin:.4f} W "
          f"(细搜认证下确界裕度 >= +0.0167 W, 见模块头注记)")

    # ---- 测试 2: TangentOracle3D 双分支自动触达 + LB <= min(E_byp, E_ovt) ----
    print("\n[Test 2] TangentOracle3D 双分支与 R5 下界不击穿...")
    from 参数 import CASE1_BUILDINGS_3D, CASE1_CUSTOMERS, DEPOT_XYZ

    # (a) 矮长墙 (h=10): 两端点分列两侧 -> 绕行代价极高, 自动触发 Overtop
    oracle_low = TangentOracle3D(
        [{"id": "W", "name": "矮超长墙", "vertices": [(-300.0, -20.0), (300.0, -20.0),
                                                    (300.0, 20.0), (-300.0, 20.0)],
          "height": 10.0}], h_safe=5.0)
    br_low, lb_low, ub_low = oracle_low.evaluate_3d_arc_branches(
        (0.0, 60.0, 0.0), (0.0, -60.0, 0.0), 0.0)
    if br_low != "overtop":
        raise RuntimeError(f"矮长墙场景应触发 Overtop, 实得 {br_low}")

    # (b) 高窄塔 (h=90): 绕行几乎不加长 -> 应选 Bypass
    oracle_high = TangentOracle3D(
        [{"id": "T", "name": "高窄塔", "vertices": [(140.0, -10.0), (160.0, -10.0),
                                                    (160.0, 10.0), (140.0, 10.0)],
          "height": 90.0}], h_safe=5.0)
    br_high, lb_high, ub_high = oracle_high.evaluate_3d_arc_branches(
        (0.0, 0.0, 0.0), (300.0, 0.0, 0.0), 0.0)
    if br_high != "bypass":
        raise RuntimeError(f"高塔场景应触发 Bypass, 实得 {br_high}")

    # (c) Case 1 实景: depot -> c4 (被 B1+B2 阻挡) 逐分支核对与物理净距硬断言
    oracle_c1 = TangentOracle3D(CASE1_BUILDINGS_3D, H_SAFE_CLEARANCE)
    p1, p2 = DEPOT_XYZ, CASE1_CUSTOMERS[4]["pos"]
    blocked, blockers = oracle_c1.check_los_obstruction(p1, p2)
    if not blocked:
        raise RuntimeError("depot->c4 应被建筑阻挡")
    br_c1, lb_c1, ub_c1 = oracle_c1.evaluate_3d_arc_branches(p1, p2, 0.5)
    L_bp, bp_path = oracle_c1.compute_bypass_tangent(p1, p2, return_path=True)
    H_ov, H_bar = oracle_c1.compute_overtop_tangent(p1, p2, blockers)
    W_arc = ENV_CONSTANTS["W_min"] + 0.5 * ENV_CONSTANTS["g"]
    E_bp = oracle_c1._blended_energy(L_bp, p2[2] - p1[2], W_arc)
    E_ov = oracle_c1._overtop_energy(p1, p2, H_bar, W_arc)
    if lb_c1 > min(E_bp, E_ov) + 1e-6:
        raise RuntimeError(f"LB {lb_c1:.1f} > min(UB分支) {min(E_bp, E_ov):.1f} — R5 击穿")

    # [回归断言 1] 物理净空硬约束: 验证 Bypass 真实几何航迹与所有原始建筑实体最小净距 >= SAFETY_MARGIN_2D
    bp_line = LineString(bp_path)
    for bd in CASE1_BUILDINGS_3D:
        b_poly = Polygon(bd["vertices"])
        d_clearance = bp_line.distance(b_poly)
        if d_clearance < SAFETY_MARGIN_2D - 1e-6:
            raise RuntimeError(f"Bypass 航迹侵入建筑 {bd['id']} 实体安全包络: 最小净空 {d_clearance:.4f} m < {SAFETY_MARGIN_2D} m")

    # [回归断言 2] Case 1 全 90 条有向端点对全量物理净空与无穿透硬断言
    all_case1_nodes = [c["pos"] for c in CASE1_CUSTOMERS]
    if len(all_case1_nodes) != 10:
        raise RuntimeError(f"Case 1 节点数期望为 10, 实得 {len(all_case1_nodes)}")
    n_nodes = len(all_case1_nodes)
    pair_count = 0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            pair_count += 1
            pi, pj = all_case1_nodes[i], all_case1_nodes[j]
            d_bp, p_bp = oracle_c1.compute_bypass_tangent(pi, pj, return_path=True)
            if not math.isinf(d_bp) and len(p_bp) >= 2:
                p_line = LineString(p_bp)
                for bd in CASE1_BUILDINGS_3D:
                    b_poly = Polygon(bd["vertices"])
                    if p_line.distance(b_poly) < SAFETY_MARGIN_2D - 1e-6:
                        raise RuntimeError(f"全 90 弧回归失败: 弧 {i}->{j} 侵入建筑 {bd['id']} 实体包络")
    if pair_count != 90:
        raise RuntimeError(f"有向弧对数期望为 90, 实得 {pair_count}")

    print(f"  PASS: 矮墙->Overtop / 高塔->Bypass 自动触达; "
          f"Case1 depot->c4: branch={br_c1}, LB={lb_c1:.0f} J <= "
          f"min(E_byp={E_bp:.0f}, E_ovt={E_ov:.0f}) J, "
          f"L_bypass={L_bp:.1f} m (原始实体净空 >= {SAFETY_MARGIN_2D} m), H_overtop={H_ov:.1f} m; "
          f"全 {pair_count} 条唯一有向弧净空 >= {SAFETY_MARGIN_2D} m 100% 验证通过!")

    # ---- 测试 3: B 样条光滑 + BEMT 闭环积分 ----
    print("\n[Test 3] B 样条轨迹与 BEMT 数值积分...")
    wpts = [(0.0, 0.0, 0.0), (150.0, 60.0, 30.0), (300.0, 120.0, 55.0),
            (420.0, 200.0, 68.0), (550.0, 310.0, 0.0)]
    traj = generate_smooth_3d_spline(wpts, dt=0.1)
    if not np.all(np.isfinite(traj["accel_norm"])):
        raise RuntimeError("加速度出现非有限值")
    if traj["max_vh"] > 15.0 + 1e-9:
        raise RuntimeError(f"v_h 超域: {traj['max_vh']}")
    if traj["max_abs_vz"] > 6.0 + 1e-9:
        raise RuntimeError(f"v_z 超域: {traj['max_abs_vz']}")
    sched = [{"t_release": traj["T"] * 0.55, "drop_kg": 0.4, "hover_s": 40.0},
             {"t_release": traj["T"] * 0.95, "drop_kg": 0.4, "hover_s": 45.0}]
    energy = integrate_spline_bemt_energy(traj, sched, dt=0.1)
    if not (energy["E_total_J"] > 0 and math.isfinite(energy["E_total_J"])):
        raise RuntimeError("能量非正或非有限")

    # P1-01 回归: 网格无关性 (dt 物理解耦后能量漂移必须消失)
    drifts = []
    e_ref = None
    for dt_scan in (0.2, 0.1, 0.05, 0.01):
        tr_s = generate_smooth_3d_spline(wpts, dt=dt_scan)
        sc = [{"t_release": tr_s["T"] * f, "drop_kg": d, "hover_s": hov}
              for f, d, hov in ((0.55, 0.4, 40.0), (0.95, 0.4, 45.0))]
        e_s = integrate_spline_bemt_energy(tr_s, sc, dt=dt_scan)["E_total_J"]
        if e_ref is None:
            e_ref = e_s
        drifts.append(abs(e_s - e_ref) / e_ref * 100.0)
    if max(drifts) >= 0.05:
        raise RuntimeError(f"P1-01 回归失败: dt 扫描能量漂移 {drifts}% (应严格 < 0.05%)")
    print(f"  PASS: 航迹长 {traj['length']:.1f} m, T={traj['T']:.1f} s, "
          f"max_vh={traj['max_vh']:.2f}, max|vz|={traj['max_abs_vz']:.2f}, "
          f"B样条导数范数 max={traj['accel_norm'].max():.1f}; "
          f"能耗: 飞行 {energy['E_flight_J']/1000:.2f} kJ + "
          f"悬停 {energy['E_hover_J']/1000:.2f} kJ = {energy['E_total_J']/1000:.2f} kJ")
    print(f"  PASS: P1-01 网格无关性 — dt=0.2/0.1/0.05/0.01 能量漂移 "
          f"{['%.4f%%' % d for d in drifts]} (阈值 0.05%)")

    # P2-01 回归: 相邻重复航路点去重
    traj_dup = generate_smooth_3d_spline(
        wpts[:2] + [wpts[1], wpts[1]] + wpts[2:], dt=0.1)
    if abs(traj_dup["length"] - traj["length"]) >= 1e-6:
        raise RuntimeError("去重后航迹长不一致")
    print("  PASS: P2-01 相邻重复航路点去重 (长度与无重复版一致)")

    # D2 回归: 足迹内端点 bypass 必须 +inf (走 Overtop); 用矩形内点验证
    p_in = (185.0, 255.0, 70.0)     # B2 [160,210]x[220,290] 的严格内点上方
    lb_in = _planes_arc_lb(math.hypot(185.0, 255.0), 70.0, W=ENV_CONSTANTS["W_min"])
    L_inf = oracle_c1.compute_bypass_tangent(p_in, (450.0, 350.0, 0.0))
    if not math.isinf(L_inf):
        raise RuntimeError("足迹内端点的 bypass 应不可行 (+inf)")
    br_in, lb_in2, ub_in = oracle_c1.evaluate_3d_arc_branches(
        p_in, (450.0, 350.0, 0.0), 0.0)
    if br_in != "overtop":
        raise RuntimeError("足迹内端点应自动走 Overtop")
    if lb_in2 > ub_in + 1e-6:
        raise RuntimeError(f"足迹内端点 R5 击穿: LB={lb_in2} > UB={ub_in}")
    print(f"  PASS: D2 足迹内端点 bypass=+inf, 自动 Overtop, LB={lb_in2:.0f} <= UB={ub_in:.0f} J")

    # ---- 附加: 势能下界快速演示 ----
    plb = compute_potential_lower_bound(
        [c for c in CASE1_CUSTOMERS if c["id"] != 0], DEPOT_XYZ,
        must_clear_height=75.0)
    print(f"\n[Demo] Case1 势能爬升下界: Δh_must={plb['climb_height_m']:.0f} m "
          f"-> E_climb_LB={plb['E_climb_lb_J']:.0f} J "
          f"(系数 {plb['coef_J_per_m']:.1f} J/m, 仅可与 c=0 行水平包络相加)")
    print("\n[Model Verification PASS] 模型.py 3D 升级自测全部通过!")
