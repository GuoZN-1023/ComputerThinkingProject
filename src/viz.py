# src/viz.py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------
# Color tokens (user-specified)
# ---------------------------
COLOR_BLUE = "#71A7D2"
COLOR_GREEN = "#A6D9C0"
COLOR_RED = "#E699A7"
COLOR_YELLOW = "#FEDD9E"
COLOR_GRID = "#D0D0D0"


# ---------------------------
# Journal-ish default styling
# ---------------------------
def set_journal_style():
    """
    Nature-ish Matplotlib defaults for papers.
    We keep the palette minimal and let plotting functions control key colors.
    """
    mpl.rcParams.update({
        # Output & layout
        "figure.dpi": 160,
        "savefig.dpi": 320,
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight",
        "savefig.transparent": False,

        # Typography (fallback-friendly)
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "axes.titleweight": "regular",
        "axes.labelweight": "regular",

        # Axes geometry
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,

        # Lines & markers
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,

        # Legend
        "legend.fontsize": 9,
        "legend.frameon": False,

        # Grid (kept off by default; enable per-plot when helpful)
        "axes.grid": False,
        "grid.color": COLOR_GRID,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
    })


def _apply_nature_axes(ax, ygrid=False):
    """Small consistent tweaks per axes (Nature-ish)."""
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    if ygrid:
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)


def plot_hourly_demand_bar(hourly_total, title="Hourly demand profile"):
    """
    Plot a clean, publication-style hourly demand bar chart.

    hourly_total: pd.Series indexed by hour (0..23) or array-like length 24.
    """
    set_journal_style()
    if isinstance(hourly_total, pd.Series):
        xs = hourly_total.index.to_numpy()
        ys = hourly_total.values
    else:
        ys = np.asarray(hourly_total)
        xs = np.arange(len(ys))

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.bar(xs, ys, color=COLOR_BLUE, linewidth=0, alpha=0.95)
    _apply_nature_axes(ax, ygrid=True)

    ax.set_xlabel("Hour")
    ax.set_ylabel("Total demand")
    ax.set_title(title)
    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_xlim(-0.6, 23.6)

    # Soft y-limits (avoid bars touching the ceiling)
    if np.isfinite(ys).any():
        ymax = float(np.nanmax(ys))
        ax.set_ylim(0, ymax * 1.08 if ymax > 0 else 1)

    plt.show()



def demand_grid_from_dataframe(demand_df, grid_n=None, hour=8):
    """
    假设 i_id 是按 i = row * grid_n + col 生成，
    把指定 hour 的需求 reshape 成 [grid_n, grid_n] 矩阵。
    """
    df_h = (
        demand_df[demand_df["t"] == hour]
        .sort_values("i_id")
    )
    vals = df_h["demand"].to_numpy()

    if grid_n is None:
        I_count = demand_df['i_id'].max() + 1
        grid_n = int(np.sqrt(I_count))
        if grid_n * grid_n != I_count:
            raise ValueError("无法推断网格大小：i_id 数量不是一个完全平方数。请手动指定 grid_n。")
    if vals.size != grid_n * grid_n:
        raise ValueError(
            f"期望 {grid_n*grid_n} 个 i_id，实际只有 {vals.size} 个，"
            "检查一下 i_id 是否按网格生成。"
        )
    grid = vals.reshape(grid_n, grid_n)
    return grid


def hourly_grids_from_dataframe(demand_df, grid_n=None, hours=None):
    """
    将 demand_df 变成按小时排列的一组 2D 网格，便于统一标尺绘制。
    """
    if hours is None:
        # default: all unique hours sorted
        hours = sorted(demand_df["t"].unique().tolist())

    grids = [demand_grid_from_dataframe(demand_df, grid_n=grid_n, hour=h) for h in hours]
    return grids, hours


def plot_hourly_heatmap_grid(demand_df=None, demand_csv_path=None, grid_n=None,
                             hours=None, ncols=6, clip_q=99):
    """
    在同一张大图上展示多个时段（通常 24 小时）的需求热力图，且 **共享同一色标**。

    - clip_q: 采用全时段的分位数做 vmax 截断（默认 99），避免极端峰值把色标“拉爆”
    """
    set_journal_style()

    if demand_df is None:
        if demand_csv_path is None:
            raise ValueError("需要传入 demand_df 或 demand_csv_path 之一。")
        demand_df = pd.read_csv(demand_csv_path)

    grids, hours = hourly_grids_from_dataframe(demand_df, grid_n=grid_n, hours=hours)

    # Global shared scale
    all_vals = np.concatenate([g.ravel() for g in grids])
    all_vals = all_vals[np.isfinite(all_vals)]
    vmin = 0.0
    vmax = np.percentile(all_vals, clip_q) if all_vals.size else 1.0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    n = len(grids)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.15, nrows * 2.15), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)

    im = None
    for idx in range(nrows * ncols):
        ax = axes.flat[idx]
        if idx < n:
            im = ax.imshow(grids[idx], norm=norm, origin="lower", interpolation="nearest")
            ax.set_title(f"{int(hours[idx]):02d}:00")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86, pad=0.02)
        cbar.set_label("Demand (shared scale)")

    fig.suptitle("Hourly demand heatmaps (shared color scale)", y=1.02)
    plt.show()


def plot_morning_evening_heatmap(demand_csv_path, grid_n=None, morning_h=8, evening_h=18, clip_q=99):
    """
    画出早高峰和晚高峰两个热力图（共享同一色标）。
    """
    set_journal_style()

    demand = pd.read_csv(demand_csv_path)

    grid_m = demand_grid_from_dataframe(demand, grid_n, morning_h)
    grid_e = demand_grid_from_dataframe(demand, grid_n, evening_h)

    all_vals = np.concatenate([grid_m.ravel(), grid_e.ravel()])
    vmin = 0.0
    vmax = np.percentile(all_vals, clip_q) if all_vals.size else 1.0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), sharex=True, sharey=True)

    im0 = axes[0].imshow(grid_m, norm=norm, origin="lower", interpolation="nearest")
    axes[0].set_title(f"{morning_h}:00")
    axes[0].set_xlabel("y")
    axes[0].set_ylabel("x")

    im1 = axes[1].imshow(grid_e, norm=norm, origin="lower", interpolation="nearest")
    axes[1].set_title(f"{evening_h}:00")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("x")

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
    cbar.set_label("Demand (shared scale)")

    fig.suptitle("Demand heatmaps (shared color scale)", y=1.02)
    plt.show()


def _pick_col_case_insensitive(df: pd.DataFrame, candidates):
    lower_to_real = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_to_real:
            return lower_to_real[key]
    return None


def plot_station_layout(stations_df, y=None):
    """
    Plot station coordinates. If y is provided, highlight active sites.

    - Inactive: small circles in COLOR_BLUE
    - Active: small stars in COLOR_GREEN
    """
    set_journal_style()

    # Common column aliases (case-insensitive)
    x_candidates = [
        "lon", "lng", "longitude", "x", "x_coord",
        "station_lon", "station_lng", "station_longitude",
    ]
    y_candidates = [
        "lat", "latitude", "y", "y_coord",
        "station_lat", "station_latitude",
    ]

    xcol = _pick_col_case_insensitive(stations_df, x_candidates)
    ycol = _pick_col_case_insensitive(stations_df, y_candidates)

    if xcol is None or ycol is None:
        raise KeyError(
            "plot_station_layout() 找不到站点坐标列。\n"
            f"当前 stations_df.columns = {list(stations_df.columns)}\n"
            "请确保 stations.csv 至少包含经纬度列（如 lon/lat 或 lng/lat 或 longitude/latitude），"
            "或把你实际的列名加到 x_candidates / y_candidates 里。"
        )

    xs = pd.to_numeric(stations_df[xcol], errors="coerce").to_numpy()
    ys = pd.to_numeric(stations_df[ycol], errors="coerce").to_numpy()

    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]

    fig, ax = plt.subplots(figsize=(4.6, 4.6))

    if y is None:
        ax.scatter(xs, ys, s=9, alpha=0.70, color=COLOR_BLUE, linewidths=0, edgecolors="none")
    else:
        y = np.asarray(y, dtype=float)
        if y.shape[0] != stations_df.shape[0]:
            raise ValueError(f"y 的长度({y.shape[0]})与站点数({stations_df.shape[0]})不一致。")
        y = y[mask]
        active = y > 0.5
        ax.scatter(xs[~active], ys[~active], s=9, alpha=0.65, color=COLOR_BLUE, marker="o",
                   linewidths=0, edgecolors="none", label="inactive")
        ax.scatter(xs[active], ys[active], s=32, alpha=0.85, color=COLOR_GREEN, marker="*",
                   linewidths=0, edgecolors="none", label="active site")
        ax.legend(loc="best")

    _apply_nature_axes(ax, ygrid=False)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title("Station layout")

    plt.show()


def plot_offline_history(history, *, terms_mode="overlay", rolling=None):
    """
    画离线搜索的历史轨迹（random_search / 局部搜索都适用）。

    兼容两种输入：
    - history: list[dict]（每次迭代一条记录）
    - history: dict[str, list/ndarray]（每个指标一条时间序列）
    """
    set_journal_style()

    if isinstance(history, list):
        # list of dicts
        def _get_series(key, default=np.nan):
            return np.array([h.get(key, default) for h in history], dtype=float)

        total = _get_series("total")
        walk = _get_series("walk_cost")
        overflow = _get_series("overflow_cost")
        imb = _get_series("imbalance_cost")
        build = _get_series("build_cost")
    elif isinstance(history, dict):
        total = np.asarray(history.get("total", []), dtype=float)
        walk = np.asarray(history.get("walk_cost", history.get("walk", [])), dtype=float)
        overflow = np.asarray(history.get("overflow_cost", history.get("overflow", [])), dtype=float)
        imb = np.asarray(history.get("imbalance_cost", history.get("imbalance", [])), dtype=float)
        build = np.asarray(history.get("build_cost", history.get("build", [])), dtype=float)
    else:
        raise TypeError("plot_offline_history(history) 需要 list[dict] 或 dict 输入。")

    it = np.arange(len(total))

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(it, total, linewidth=1.2)
    ax.set_title("Offline search trajectory")
    ax.set_xlabel("iteration")
    ax.set_ylabel("objective (normalized)")
    plt.show()


def plot_online_curves(history):
    """
    画在线仿真的几条指标曲线（配色对齐：红/黄/绿/蓝）。
    可选：如果 history 里有 active_stations 或 y_t，则画激活站点随时间变化。
    """
    set_journal_style()

    # 你的四条核心曲线
    walk = np.array(history["walk_cost"])
    overflow = np.array(history["overflow_cost"])
    imb = np.array(history["imbalance_cost"])
    total_r = np.array(history["total_reward"])
    t = np.arange(len(walk))



    # 1) Total reward
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.plot(t, total_r, linewidth=1.2, color=COLOR_BLUE)
    ax.set_title("Total reward per time slice")
    ax.set_xlabel("t")
    ax.set_ylabel("reward")
    plt.show()

    # 2) Cost terms
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.plot(t, walk,     label="walk",     linewidth=1.1, color=COLOR_BLUE)
    ax.plot(t, overflow, label="overflow", linewidth=1.1, color=COLOR_YELLOW)
    ax.plot(t, imb,      label="imbalance",linewidth=1.1, color=COLOR_GREEN)
    ax.set_title("Cost terms per time slice")
    ax.set_xlabel("t")
    ax.set_ylabel("cost")
    ax.legend()
    plt.show()

    # 3) Active stations over time (optional)
    # 你如果在模拟时记录了它，放到 history["active_stations"]（每个时间片一个数）
    active = None
    if "active_stations" in history:
        active = np.array(history["active_stations"])
    elif "used_stations" in history:
        # number of stations that received at least one arrival in this time slice
        active = np.array(history["used_stations"])
    elif "y_t" in history:
        # y_t: list/array of y vectors each time; compute active count
        active = np.array([np.sum(np.array(y) > 0.5) for y in history["y_t"]])
    if active is not None and len(active) == len(t):
            fig, ax = plt.subplots(figsize=(7.6, 2.8))
            ax.plot(t, active, linewidth=1.1, color=COLOR_RED)
            ax.set_title("Active stations over time")
            ax.set_xlabel("t")
            ax.set_ylabel("# active stations")
            plt.show()
    
def plot_objective_terms_separated(history, *, normalized=True, rolling=None):
    """
    将 walk / overflow / imbalance / build 四个目标项拆分成 2×2 子图绘制，避免叠在一起过于混乱。

    Parameters
    ----------
    history : list[dict] or dict[str, array-like]
        random_search / local_search 输出的 history。
    normalized : bool
        True: 使用 *cost*（通常对应 normalize_terms=True 时的归一化项）
        False: 使用 *raw*（受单位影响的原始项）
    rolling : int | None
        可选滑动平均窗口（仅用于视觉平滑，不改变数据本身）。None 表示不平滑。
    """
    set_journal_style()

    # ---- extract series ----
    if isinstance(history, list):
        def _get_series(key, default=np.nan):
            return np.array([h.get(key, default) for h in history], dtype=float)

        if normalized:
            walk = _get_series("walk_cost")
            overflow = _get_series("overflow_cost")
            imb = _get_series("imbalance_cost")
            build = _get_series("build_cost")
        else:
            walk = _get_series("walk_raw")
            overflow = _get_series("overflow_raw")
            imb = _get_series("imbalance_raw")
            build = _get_series("build_raw")

    elif isinstance(history, dict):
        if normalized:
            walk = np.asarray(history.get("walk_cost", history.get("walk", [])), dtype=float)
            overflow = np.asarray(history.get("overflow_cost", history.get("overflow", [])), dtype=float)
            imb = np.asarray(history.get("imbalance_cost", history.get("imbalance", [])), dtype=float)
            build = np.asarray(history.get("build_cost", history.get("build", [])), dtype=float)
        else:
            walk = np.asarray(history.get("walk_raw", []), dtype=float)
            overflow = np.asarray(history.get("overflow_raw", []), dtype=float)
            imb = np.asarray(history.get("imbalance_raw", []), dtype=float)
            build = np.asarray(history.get("build_raw", []), dtype=float)
    else:
        raise TypeError("history must be a list[dict] or dict[str, array-like]")

    n = int(max(map(len, [walk, overflow, imb, build]), default=0))
    it = np.arange(n)

    def _smooth(arr):
        if rolling is None or rolling <= 1:
            return arr
        s = pd.Series(arr)
        return s.rolling(int(rolling), center=True, min_periods=max(1, int(rolling)//3)).mean().to_numpy()

    walk_s = _smooth(walk)
    overflow_s = _smooth(overflow)
    imb_s = _smooth(imb)
    build_s = _smooth(build)

    colors = {
        "walk": COLOR_BLUE,        # 蓝
        "overflow": COLOR_YELLOW,  # 黄
        "imbalance": COLOR_GREEN,  # 绿
        "build": COLOR_RED,        # 红
    }

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.2), sharex=True)
    axes = axes.ravel()

    series = [
        ("walk", walk_s),
        ("overflow", overflow_s),
        ("imbalance", imb_s),
        ("build", build_s),
    ]

    titles = {
        "walk": "Walk",
        "overflow": "Overflow",
        "imbalance": "Imbalance",
        "build": "Build",
    }

    for ax, (name, arr) in zip(axes, series):
        ax.plot(it, arr, linewidth=1.2, color=colors[name])
        ax.set_title(titles[name])
        ax.set_xlabel("iteration")
        ax.set_ylabel("term value")
        _apply_nature_axes(ax, ygrid=True)

    fig.suptitle("Objective terms ({})".format("normalized" if normalized else "raw"))
    plt.show()


def plot_station_distribution_panels(stations_df, cap, L_hist, times=(7, 12, 18, 22),
                                     title="Station load distribution (size=L, color=L/cap)"):
    """
    在同一张大图上对比多个时间片的站点分布。
    - 点大小：站点负载/车辆数 L
    - 点颜色：饱和度 L/cap（蓝→绿→黄→红，对齐调色板）

    Parameters
    ----------
    stations_df : pd.DataFrame
        站点信息表，需包含 lon/lat（或 lng/latitude, x/y 等同义列）
    cap : array-like, shape (J,)
        站点容量
    L_hist : array-like, shape (T, J)
        每个时间片的站点负载/车辆数
    times : tuple[int]
        需要展示的时间片索引
    """
    set_journal_style()

    cap = np.asarray(cap, dtype=float)
    L_hist = np.asarray(L_hist, dtype=float)

    # 自动识别经纬度列
    cols = {c.lower(): c for c in stations_df.columns}
    xcol = cols.get("lon") or cols.get("lng") or cols.get("longitude") or cols.get("x")
    ycol = cols.get("lat") or cols.get("latitude") or cols.get("y")
    if xcol is None or ycol is None:
        raise KeyError(f"Cannot find lon/lat columns in stations_df: {list(stations_df.columns)}")

    xs = stations_df[xcol].to_numpy()
    ys = stations_df[ycol].to_numpy()

    n = len(times)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.1), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, t in zip(axes, times):
        if t < 0 or t >= L_hist.shape[0]:
            raise IndexError(f"time index {t} out of range [0, {L_hist.shape[0]-1}]")
        L = L_hist[t]
        sat = np.clip(L / np.maximum(cap, 1e-9), 0, 1)

        # 点大小：按分位数缩放，避免极端值撑爆尺度
        denom = np.percentile(L[L > 0], 90) if np.any(L > 0) else 1.0
        s = 8 + 60 * np.clip(L / max(denom, 1e-9), 0, 1)

        # 分段颜色（直观、易讲故事）
        colors = np.where(sat < 0.33, COLOR_BLUE,
                  np.where(sat < 0.66, COLOR_GREEN,
                  np.where(sat < 0.90, COLOR_YELLOW, COLOR_RED)))

        ax.scatter(xs, ys, s=s, c=colors, alpha=0.85, edgecolors="none")
        ax.set_title(f"{int(t):02d}:00")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    fig.suptitle(title, y=1.04)
    plt.tight_layout()
    plt.show()


def plot_bike_stock_summary(history):
    """
    展示站内总车量、满站/空站数量随时间的变化。
    要求 history 至少包含：
    - total_bikes_in_stations
    - num_full_stations
    - num_empty_stations
    """
    set_journal_style()

    if "total_bikes_in_stations" not in history:
        raise KeyError("history missing 'total_bikes_in_stations'. Run online simulation with updated simulate.py.")

    total_bikes = np.asarray(history["total_bikes_in_stations"], dtype=float)
    full_cnt = np.asarray(history.get("num_full_stations", []), dtype=float)
    empty_cnt = np.asarray(history.get("num_empty_stations", []), dtype=float)
    t = np.arange(len(total_bikes))

    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.plot(t, total_bikes, color=COLOR_BLUE, linewidth=1.2)
    ax.set_title("Total bikes in stations over time")
    ax.set_xlabel("t")
    ax.set_ylabel("Total bikes (in stations)")
    plt.show()

    if full_cnt.size and empty_cnt.size and len(full_cnt) == len(total_bikes) and len(empty_cnt) == len(total_bikes):
        fig, ax = plt.subplots(figsize=(7.6, 3.0))
        ax.plot(t, full_cnt, color=COLOR_RED, linewidth=1.1, label="full stations")
        ax.plot(t, empty_cnt, color=COLOR_GREEN, linewidth=1.1, label="empty stations")
        ax.set_title("Full/Empty stations over time")
        ax.set_xlabel("t")
        ax.set_ylabel("# stations")
        ax.legend()
        plt.show()