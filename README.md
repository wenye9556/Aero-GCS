# Aero-GCS

**Certified global lower-bound optimization for multi-UAV parcel delivery with
BEMT physics, per-route battery constraints, and real 3D city GIS.** 中文说明见下文。

Aero-GCS solves energy-optimal multi-drone routing on real city instances
(Singapore Marina Bay / Shenzhen Futian / Shanghai Lujiazui × 60/90/120-node
instances, i.e. 59/89/119 customers + 1 depot) with a branch-and-price engine that produces **certified gaps**:
every reported solution ships with a mathematical proof that it is within the
stated % of the true optimum — and every route it outputs satisfies a
**per-route battery hard constraint** (E_route ≤ 360 kJ) *by construction*.

## Highlights

- **Physics**: rotor energy from a BEMT model with payload-dependent arc costs
  (full load costs ~1.7× empty), 3D building geometry, line-of-sight
  clearance, per-arc optimal cruise altitude. The arc-cost tensor *is* the
  energy model, so the battery constraint is a per-route cost cap — one
  accounting, zero bookkeeping drift.
- **Per-route battery hard constraint** (energy feasible by construction):
  pricing labels carry an energy dimension (prune on E > E_usable, closure
  gate on the complete route), every neighborhood move in the primal side is
  energy-gated, and a terminal assertion re-checks every route of the final
  solution. Across the 9 instances all 73 routes respect the cap (per-route
  energy 131.7–358.7 kJ against the 360 kJ usable budget, max utilization
  99.6%; enabling the constraint moves the feasible UB by only −2.6% to +2.9%
  versus the unconstrained historical baseline, and by −1.5% to +2.9% on the
  four originally-over-limit cells).
- **Certificate chain**: set-partitioning master over the energy-feasible
  column space Ω_E + ng-route pricing, Farley anytime bound
  `z + K·min(0, μ̂)` with ancestor-monotone tree certificates — the reported
  gap is provable, not heuristic.
- **Two components, one universal config** (no per-city tuning), both deployed
  as **post-processors of the same base run** (shared-trajectory decoupling):
  - **Post certifier (LB)**: the base run passively records root Farley sites
    (z, π, σ); after it finishes, the last ≤8 dual snapshots in the endgame
    window are re-priced on a fine load grid. UB bitwise unchanged, LB only
    rises ⇒ gap(M2) ≤ gap(M1) by construction.
  - **Monotone end-of-search kernel (UB)**: two-phase accept-only-improvements
    LNS on the final incumbent (strict phase + swap/sideways-walk phase with
    global best-ever tracking). LB bitwise unchanged, UB only falls ⇒
    gap(M3) ≤ gap(M1) by construction.
  - Hence the 4-arm ordering gap(M1) ≥ gap(M2) ≥ gap(M4), gap(M1) ≥ gap(M3) ≥
    gap(M4) holds **bitwise on every instance** — a constructive guarantee, not
    a statistical one (the earlier in-loop deployment violated it in 8/9 cells
    through wall-clock/column-pool trajectory coupling and was retired).
- **Universal string results** (single config, seed 42, 500/1000/2000 s
  budgets; fleet size K = ⌈Σd/1kg⌉ + max(0, ⌈(n−60)/30⌉)):

| instance | base gap | Aero-GCS gap | base / cert / kernel wall |
|---|---:|---:|---|
| SG60 | 3.65% | **2.49%** | 447* / 103 / 251 s |
| SZ60 | 4.57% | **2.37%** | 460* / 105 / 250 s |
| SH60 | 13.48% | **13.48%** | 504 / 108 / 250 s |
| SG90 | 4.95% | **4.36%** | 1001 / 17 / 500 s |
| SZ90 | 5.40% | **4.51%** | 1005 / 65 / 501 s |
| SH90 | 11.37% | **9.64%** | 1013 / 38 / 502 s |
| SG120 | 5.41% | **4.98%** | 2003 / 35 / 1001 s |
| SZ120 | 5.89% | **5.38%** | 2007 / 55 / 1000 s |
| SH120 | 15.00% | **12.59%** | 2006 / 64 / 1001 s |

\* SG60/SZ60 base hit the 5% anytime target early. The base search budget is
the input (500 s @ 60n, 1000 s @ 90n, 2000 s @ 120n); certifier and kernel are
post-processing stages reported separately. 8/9 improved, 1 tie (SH60), zero
degradations, mean −1.10 pp, M4 mean 6.64%, zero per-city switches, ordering
by construction.

**4-arm ablation** (M1 base / M2 +LB component / M3 +UB component / M4 full,
same budgets, certified gap %):

All four arms are **readings of one shared base run** per instance (seed 42),
so the ordering gap(M1) ≥ gap(M2) ≥ gap(M4), gap(M1) ≥ gap(M3) ≥ gap(M4) holds
bitwise by construction — there is no cross-run scheduling noise:

| instance | M1 | M2 | M3 | M4 | LB lift (kJ) | UB cut (kJ) |
|---|---:|---:|---:|---:|---:|---:|
| SG60 | 3.65 | 2.49 | 3.65 | **2.49** | +12.0 | 0.0 |
| SZ60 | 4.57 | 2.69 | 4.27 | **2.37** | +21.4 | 3.6 |
| SH60 | 13.48 | 13.48 | 13.48 | **13.48** | 0.0 | 0.0 |
| SG90 | 4.95 | 4.95 | 4.36 | **4.36** | 0.0 | 10.8 |
| SZ90 | 5.40 | 5.40 | 4.51 | **4.51** | 0.0 | 16.7 |
| SH90 | 11.37 | 11.37 | 9.64 | **9.64** | 0.0 | 37.3 |
| SG120 | 5.41 | 5.41 | 4.98 | **4.98** | 0.0 | 10.5 |
| SZ120 | 5.89 | 5.89 | 5.38 | **5.38** | 0.0 | 11.7 |
| SH120 | 15.00 | 15.00 | 12.59 | **12.59** | 0.0 | 74.3 |

The certifier lifts the lower bound in 2/9 instances (+12.0/+21.4 kJ — the
legacy-grid 60n cells; at 90/120n pricing already runs on the energy-aware
1020 forward-labeling DP, so the uplift is structurally zero); the kernel cuts
the upper bound in 7/9 instances (3.6 to 74.3 kJ, 164.8 kJ total) and is
provably harmless elsewhere.

## Install & run

```bash
pip install -r requirements.txt   # numpy, scipy, numba, shapely, ortools
python run.py --city shanghai_lujiazui --nodes 60 --budget 500
# LB / UB / certified gap / wall, plus --out routes.json
```

## Repository layout

```
算法.py / 参数.py / 模型.py     engine core (solver + frontier DP / constants / BEMT physics)
实验.py                          unified experiment driver (selfcheck / run / ablation / model / p3 / p4)
run.py                           universal entry point — the single deployed config
可视化.py                        paper figure renderer (reads results/logs only, no hardcoded data)
results/                         gis_data (instances + physics tensors), logs (evidence), charts/paper (F1–F5)
```

Each module has a Chinese companion doc (算法.md etc., kept local, not tracked).

## Reproducing

`实验.py` (selfcheck / run / ablation / model / p3 / p4) drives every
configuration. Key evidence: `results/logs/ablation4_post.json` /
`ablation4_post.log` (4-arm ablation, one shared base run per cell — the
tables above), `det_curve_route2.log` / `a6_cert_tightening_route2.json`
(certificate-quality instruments), `model_experiments_v2.json` (physics
gates P1/P2/P5), `p3_2d3d.json` (2D-vs-3D counterfactual),
`p4_energy_decomposition.json` + `p4_routes/` (four-channel energy
decomposition: hover + direct/overtop/bypass flight, oracle replay per arc).
`可视化.py` re-renders all paper figures (F1–F5) from these logs:
`python 可视化.py`.

## License

MIT — see LICENSE.

---

# 中文说明

**Aero-GCS：面向真实三维城市 GIS 的无人机物流能量最优路由 + 可认证全局下界系统
（含逐航线电量硬约束）。**

- 三城市（新加坡滨海湾 / 深圳福田 / 上海陆家嘴）× 三规模（60/90/120 节点档 = 59/89/119 客户 + 1 仓库）真实算例；
- BEMT 物理能耗（载荷相关弧成本：满载约为空载 1.7 倍）+ 三维建筑避障 + 逐弧最优巡航高度；
  弧代价张量即能耗模型，**逐航线电量硬约束（≤360 kJ）由定价电量维标号、闭合门、
  列准入门与终局断言按构造保证**（九算例 73 条航线全部 ≤358.7 kJ，利用率上限 99.6%）；
- Branch-and-Price + Farley 任意时刻界 + 祖先单调树证书——**报告的 gap 是被证明的**；
- 两个组件、一条通用配置串（零按城调参）：
  **证书-定价解耦**（下界侧）与 **上界单调内核（终局两段 LNS）**（上界侧）；
- 通用串（后置解耦形态）：四臂 = 同一基础运行的四个读数，gap 排序
  M1≥M2≥M4、M1≥M3≥M4 **按构造逐位成立**（详见 算法.md 定理 3）；
- 预算 500/1000/2000 s（60/90/120 档），机队规模
  K = ⌈Σd/1kg⌉ + max(0, ⌈(n−60)/30⌉)。

运行：`pip install -r requirements.txt`，然后
`python run.py --city shanghai_lujiazui --nodes 60 --budget 500`。
