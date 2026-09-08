"""把混合 jsonl 的结果按 benchmark 拆开，分别套 MMStar / OCRBench 报表。"""

import os

import eval_results_mmstar
import eval_results_ocrbench
import anlz_cont_quality
import write_average_sheet
import dispatch

EVALUATORS = {
    "MMStar": eval_results_mmstar,
    "OCRBench": eval_results_ocrbench,
}

AVG_PREFIXES = {
    "MMStar": ("summary", "breakdown"),
    "OCRBench": ("summary", "category", "breakdown"),
}


def split_df(df):
    if df is None or len(df) == 0:
        return {}
    work = df.copy()
    work["benchmark"] = [
        dispatch.benchmark_of(row) for row in work.to_dict(orient="records")
    ]
    groups = {}
    for name, sub in work.groupby("benchmark", dropna=False):
        key = str(name) if name is not None and str(name) != "nan" else "Unknown"
        groups[key] = sub.reset_index(drop=True)
    return groups


def evaluate_groups(
    groups,
    output_root,
    show_detail,
    sheet_prefix,
    mode,
    writers,
    analysis_name,
):
    for bench, sub in groups.items():
        ev = EVALUATORS.get(bench)
        if ev is None:
            print(f"[WARN] unknown benchmark {bench!r}, skip {len(sub)} rows")
            continue
        bench_dir = os.path.join(output_root, bench)
        os.makedirs(bench_dir, exist_ok=True)
        print(f"\n===== {bench} ({len(sub)} cases) =====")
        ev.evaluate_reference_results(
            sub,
            verbal=show_detail,
            save_summary_path=(
                os.path.join(bench_dir, f"{mode}_Summary.xlsx")
                if show_detail
                else None
            ),
            save_writer=writers.get(bench),
            sheet_prefix=sheet_prefix,
        )
        if show_detail:
            anlz_cont_quality.analyze_entropy_and_repeat(
                sub,
                save_fig_path=os.path.join(bench_dir, analysis_name),
            )


def write_averages(output_root, benches, mode):
    for bench in benches:
        ev = EVALUATORS.get(bench)
        if ev is None:
            continue
        xlsx = os.path.join(output_root, bench, f"{mode}_Summary.xlsx")
        if not os.path.isfile(xlsx):
            continue
        for prefix in AVG_PREFIXES.get(bench, ("summary", "breakdown")):
            write_average_sheet.save_overall_summary(xlsx, prefix, mode=mode)
