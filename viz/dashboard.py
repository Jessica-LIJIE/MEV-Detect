"""MEV-Detect Streamlit 可视化面板（E4）。

启动: streamlit run viz/dashboard.py
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.log_loader import DEFAULT_LIVE_LOG, load_detection_dataframe, load_jsonl_records

FIG_DIR = ROOT / "data" / "figures"
RESULT_DIR = ROOT / "data" / "results"
E2_DATA_DIR = ROOT / "data" / "E2-data"

DEMO_MOCK_ID = "snap_004"
DEMO_LIVE_ID = "live_0010"

_CJK_FONT_KEYWORDS = (
    "microsoft yahei",
    "simhei",
    "pingfang",
    "noto sans cjk",
    "wenquanyi",
    "source han sans",
)
_CJK_CONFIGURED = False


def _resolve_cjk_font() -> str | None:
    for font in font_manager.fontManager.ttflist:
        name_lower = font.name.lower()
        if any(keyword in name_lower for keyword in _CJK_FONT_KEYWORDS):
            return font.name
    return None


def _setup_matplotlib_cjk() -> None:
    """Pick a CJK-capable font so chart titles/labels render on Windows/Linux."""
    global _CJK_CONFIGURED
    if _CJK_CONFIGURED:
        return

    resolved = _resolve_cjk_font()
    if resolved:
        plt.rcParams["font.sans-serif"] = [resolved, "DejaVu Sans"]
    elif platform.system() == "Windows":
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    elif platform.system() == "Darwin":
        plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]

    plt.rcParams["axes.unicode_minus"] = False
    _CJK_CONFIGURED = True


_setup_matplotlib_cjk()


@st.cache_data(show_spinner=False)
def _load_df(source: str, dedupe: bool) -> pd.DataFrame:
    return load_detection_dataframe(source=None if source == "all" else source, dedupe=dedupe)


@st.cache_data(show_spinner=False)
def _load_adaptive_benchmark() -> dict | None:
    path = RESULT_DIR / "adaptive_pso_benchmark.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _load_e2_benchmarks() -> list[dict]:
    if not E2_DATA_DIR.exists():
        return []
    benchmarks: list[dict] = []
    for path in sorted(E2_DATA_DIR.glob("*/multi_gpu_benchmark.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["_subdir"] = path.parent.name
        benchmarks.append(data)
    return benchmarks


def _e2_results_dataframe(benchmark: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "GPU 数": r["world_size"],
                "墙钟耗时 (ms)": round(r["elapsed_ms_mean"], 2),
                "标准差 (ms)": round(r["elapsed_ms_stdev"], 2),
                "加速比": round(r["speedup"], 3),
                "并行效率 %": round(r["parallel_efficiency_pct"], 2),
                "最优 fitness": round(r["best_fitness_mean"], 2),
                "每卡粒子数": r["num_particles_per_rank"],
            }
            for r in benchmark["results"]
        ]
    )


def _make_e2_time_figure(results: list[dict], particles: int) -> plt.Figure:
    _setup_matplotlib_cjk()
    gpus = [r["world_size"] for r in results]
    times = [r["elapsed_ms_mean"] for r in results]
    colors = ["#4C78A8", "#F58518", "#54A24B"][: len(gpus)]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([str(g) for g in gpus], times, color=colors)
    ax.set_xlabel("GPU 数量")
    ax.set_ylabel("墙钟耗时 (ms)")
    ax.set_title(f"E2 多 GPU 墙钟耗时（{particles} 粒子）")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{t:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    return fig


def _make_e2_efficiency_figure(results: list[dict], particles: int) -> plt.Figure:
    _setup_matplotlib_cjk()
    gpus = [r["world_size"] for r in results]
    base = results[0]["elapsed_ms_mean"]
    speedups = [base / r["elapsed_ms_mean"] for r in results]
    efficiencies = [s / g * 100 for s, g in zip(speedups, gpus)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gpus, speedups, "o-", label="加速比", linewidth=2, color="#4C78A8")
    ax.plot(gpus, [g / gpus[0] for g in gpus], "--", label="理想线性", alpha=0.7, color="#888888")
    ax.set_xlabel("GPU 数量")
    ax.set_ylabel("加速比")
    ax.set_title(f"E2 加速比与并行效率（{particles} 粒子）")
    ax2 = ax.twinx()
    ax2.bar([g - 0.15 for g in gpus], efficiencies, width=0.3, alpha=0.35, color="#72B7B2", label="并行效率 %")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def _render_e2_section() -> None:
    st.subheader("E2：多 GPU")
    st.caption("数据目录：`data/E2-data/`（云主机 `benchmark_multi_gpu.py` 输出）。")

    benchmarks = _load_e2_benchmarks()
    if not benchmarks:
        st.info("未找到 E2 benchmark。请将各配置的 `multi_gpu_benchmark.json` 放入 `data/E2-data/<配置名>/`。")
        return

    labels = [f"{b['_subdir']}（{b['particles']} 粒子，max_iter={b['max_iter']}）" for b in benchmarks]
    choice = st.selectbox("E2 实验配置", labels, key="e2_benchmark_select")
    benchmark = benchmarks[labels.index(choice)]
    results = benchmark["results"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("快照", benchmark["record_id"])
    c2.metric("重复次数", benchmark["repeats"])
    c3.metric("随机种子", benchmark["seed"])
    c4.metric("采集时间", benchmark.get("timestamp", "—")[:19].replace("T", " "))

    st.dataframe(_e2_results_dataframe(benchmark), use_container_width=True, hide_index=True)

    best_1gpu = results[0]["elapsed_ms_mean"]
    worst = max(results, key=lambda r: r["elapsed_ms_mean"])
    if worst["world_size"] > 1 and worst["elapsed_ms_mean"] > best_1gpu:
        slowdown = worst["elapsed_ms_mean"] / best_1gpu
        st.caption(
            f"多卡墙钟耗时高于单卡（{worst['world_size']} 卡约慢 {slowdown:.1f}×），"
            "符合通信同步开销占主导、问题规模偏小的典型现象。"
        )

    col1, col2 = st.columns(2)
    with col1:
        fig_time = _make_e2_time_figure(results, benchmark["particles"])
        st.pyplot(fig_time, use_container_width=True)
        plt.close(fig_time)
    with col2:
        fig_eff = _make_e2_efficiency_figure(results, benchmark["particles"])
        st.pyplot(fig_eff, use_container_width=True)
        plt.close(fig_eff)


def _format_opportunity(value) -> str:
    if pd.isna(value):
        return "—"
    return "有机会" if bool(value) else "无套利"


def _render_kpis(df: pd.DataFrame) -> None:
    total = len(df)
    opp = int(df["has_opportunity"].fillna(False).sum()) if total else 0
    mock_n = int((df["source"] == "mock").sum()) if total else 0
    live_n = int((df["source"] == "live").sum()) if total else 0
    avg_ms = df["search_elapsed_ms"].mean() if total else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总记录", total)
    c2.metric("有机会", opp)
    c3.metric("Mock", mock_n)
    c4.metric("Live", live_n)
    c5.metric("平均搜索耗时", f"{avg_ms:.1f} ms" if total else "—")


def _tab_spread(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("暂无检测记录。请先运行 `python main.py --mock` 或 `--live`。")
        return

    plot_df = df.sort_values("timestamp").copy()
    plot_df["label"] = plot_df["snapshot_id"]

    st.subheader("L1 / L2 ETH-USDC 价格")
    price_df = plot_df.set_index("label")[["l1_price", "l2_price"]]
    st.line_chart(price_df, height=320)

    st.subheader("价差 %")
    spread_df = plot_df.set_index("label")[["spread_pct"]]
    st.line_chart(spread_df, height=240)

    st.caption("Mock 与 Live 价格量级可能不同，建议侧边栏按数据源筛选后查看。")


def _tab_triggers(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("暂无触发记录。")
        return

    display_cols = [
        "snapshot_id",
        "source",
        "timestamp",
        "l1_block",
        "l2_block",
        "l2_lag_ms",
        "spread_pct",
        "swap_amount_usd",
        "has_opportunity",
        "expected_profit_usd",
        "search_elapsed_ms",
    ]
    table = df[display_cols].copy()
    table["timestamp"] = table["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    table["has_opportunity"] = table["has_opportunity"].map(_format_opportunity)
    table = table.rename(
        columns={
            "snapshot_id": "快照 ID",
            "source": "来源",
            "timestamp": "时间",
            "l1_block": "L1 块高",
            "l2_block": "L2 块高",
            "l2_lag_ms": "L2 延迟 (ms)",
            "spread_pct": "价差 %",
            "swap_amount_usd": "Swap USD",
            "has_opportunity": "套利",
            "expected_profit_usd": "净利润 USD",
            "search_elapsed_ms": "搜索 ms",
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)


def _tab_strategy(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("暂无策略详情。")
        return

    ids = df["snapshot_id"].tolist()
    default_idx = 0
    if DEMO_MOCK_ID in ids:
        default_idx = ids.index(DEMO_MOCK_ID)
    elif DEMO_LIVE_ID in ids:
        default_idx = ids.index(DEMO_LIVE_ID)

    selected_id = st.selectbox("选择快照", ids, index=default_idx)
    row = df[df["snapshot_id"] == selected_id].iloc[0]

    st.markdown(f"**场景：** {row['scenario'] or '（Live 无场景描述）'}")
    st.markdown(f"**触发原因：** `{row['trigger']}`")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("L1 价格", f"${row['l1_price']:.2f}")
    m2.metric("L2 价格", f"${row['l2_price']:.2f}")
    m3.metric("价差", f"{row['spread_pct']:.3f}%")
    m4.metric("L2 延迟", f"{int(row['l2_lag_ms'])} ms")

    st.subheader("PSO 搜索结果")
    s1, s2, s3, s4 = st.columns(4)
    profit = row["expected_profit_usd"]
    s1.metric("净利润", f"${profit:.2f}", delta=_format_opportunity(row["has_opportunity"]))
    s2.metric("投入 ETH", f"{row['amount_in_eth']:.2f}")
    s3.metric("Gas 成本", f"${row['gas_cost_usd']:.2f}")
    s4.metric("桥费", f"${row['strategy_bridge_fee_usd']:.2f}")

    st.markdown(
        f"- 路由 L1=`{int(row['route_l1'])}`  L2=`{int(row['route_l2'])}`  桥=`{int(row['bridge_path'])}`  \n"
        f"- 搜索耗时：**{row['search_elapsed_ms']:.1f} ms**"
    )

    if pd.notna(row.get("pso_profile_name")):
        st.subheader("PSO 参数档位（E3c）")
        st.markdown(
            f"- 档位：`{row['pso_profile_name']}`  \n"
            f"- 粒子数：{int(row['pso_particles']) if pd.notna(row['pso_particles']) else '—'}  \n"
            f"- 最大迭代：{int(row['pso_max_iter']) if pd.notna(row['pso_max_iter']) else '—'}  \n"
            f"- 原因：{row['pso_reason'] or '—'}"
        )
    else:
        st.caption("该记录未写入 pso_profile（可能为早期 Live 记录或固定参数 Mock）。")

    with st.expander("原始 JSON 记录"):
        raw = load_jsonl_records()
        match = [r for r in raw if r.get("snapshot_id") == selected_id]
        if match:
            st.json(match[-1])


def _show_figure(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"未找到图表：`{path}`。请先运行对应实验脚本生成。")


def _tab_algorithms() -> None:
    st.subheader("E1：PSO vs GA")
    st.caption("运行 `python tests/test_pso_vs_ga.py` 生成。")
    _show_figure(FIG_DIR / "pso_vs_ga.png", "PSO vs GA 耗时与利润对比")

    st.divider()
    st.subheader("E3c：固定 vs 自适应 PSO")
    st.caption("运行 `python tests/test_adaptive_pso.py` 生成。")

    benchmark = _load_adaptive_benchmark()
    if benchmark:
        c1, c2, c3 = st.columns(3)
        c1.metric("固定总耗时", f"{benchmark['fixed_total_ms']:.1f} ms")
        c2.metric("自适应总耗时", f"{benchmark['adaptive_total_ms']:.1f} ms")
        c3.metric("总耗时节省", f"{benchmark['time_saved_pct']:.2f}%")

        profile_counts = benchmark.get("profile_counts", {})
        if profile_counts:
            st.markdown(
                "**分档统计：** "
                + "，".join(f"`{k}` × {v}" for k, v in profile_counts.items())
            )

        rows = benchmark.get("records", [])
        if rows:
            bench_df = pd.DataFrame(
                [
                    {
                        "快照": r["snapshot_id"],
                        "档位": r["adaptive_profile"],
                        "固定 ms": r["fixed"]["elapsed_ms"],
                        "自适应 ms": r["adaptive"]["elapsed_ms"],
                        "固定利润 $": r["fixed"]["profit_usd"],
                        "自适应利润 $": r["adaptive"]["profit_usd"],
                    }
                    for r in rows
                ]
            )
            st.dataframe(bench_df, use_container_width=True, hide_index=True)
    else:
        st.info("未找到 `data/results/adaptive_pso_benchmark.json`。")

    col_a, col_b = st.columns(2)
    with col_a:
        _show_figure(FIG_DIR / "adaptive_vs_fixed.png", "各快照 / 合计耗时对比")
    with col_b:
        _show_figure(FIG_DIR / "adaptive_vs_fixed_profit.png", "各快照净利润对比")

    st.divider()
    _render_e2_section()


def main() -> None:
    st.set_page_config(
        page_title="MEV-Detect 检测面板",
        page_icon="📊",
        layout="wide",
    )
    st.title("MEV-Detect · L1-L2 跨层 MEV 检测与可视化")
    st.caption("读取检测记录与算法实验结果。")

    with st.sidebar:
        st.header("数据筛选")
        source = st.radio("数据源", ["all", "mock", "live"], format_func=lambda x: {
            "all": "全部",
            "mock": "仅 Mock 回放",
            "live": "仅 Live 监听",
        }[x])
        dedupe = st.checkbox("按 snapshot_id 去重（保留最新）", value=True)
        if st.button("刷新数据"):
            _load_df.clear()
            _load_adaptive_benchmark.clear()
            st.rerun()

        st.divider()
        st.markdown("**数据文件**")
        st.code(str(DEFAULT_LIVE_LOG.relative_to(ROOT)), language=None)
        st.caption(
            f"策略详情默认快照：Mock `{DEMO_MOCK_ID}`，Live `{DEMO_LIVE_ID}`"
        )

    df = _load_df(source, dedupe)
    if not DEFAULT_LIVE_LOG.exists():
        st.error(f"未找到 `{DEFAULT_LIVE_LOG}`。请先运行 Mock 或 Live 检测。")

    _render_kpis(df)

    tab1, tab2, tab3, tab4 = st.tabs(["价差曲线", "触发列表", "策略详情", "算法性能"])
    with tab1:
        _tab_spread(df)
    with tab2:
        _tab_triggers(df)
    with tab3:
        _tab_strategy(df)
    with tab4:
        _tab_algorithms()


if __name__ == "__main__":
    main()
