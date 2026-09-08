import pandas as pd

QUESTION_TYPE_TO_CATEGORY = {
    "Regular Text Recognition": "Text Recognition",
    "Irregular Text Recognition": "Text Recognition",
    "Artistic Text Recognition": "Text Recognition",
    "Non-Semantic Text Recognition": "Text Recognition",
    "Handwriting Recognition": "Text Recognition",
    "Digit String Recognition": "Text Recognition",
    "Scene Text-centric VQA": "Scene Text-centric VQA",
    "Doc-oriented VQA": "Doc-oriented VQA",
    "Key Information Extraction": "Key Information Extraction",
    "Handwritten Mathematical Expression Recognition": "Handwritten Math Expr",
}


def _overall_row(df, project_name, flavor):
    decode_sum = df["decode_time"].dropna().sum()
    tps = df["response_token_len"].sum() / decode_sum if decode_sum > 0 else 0.0
    correct_cnt = int(df["correct"].sum())
    return {
        "Project": project_name,
        "Flavor": flavor,
        "Test No.": len(df),
        "prompt len": round(df["prompt_token_len"].mean(), 2),
        "response len": round(df["response_token_len"].mean(), 2),
        "get_ans": round(df["get_ans"].mean(), 2),
        "rp>0.1": round((df["repeat"] > 0.1).mean(), 4),
        "rp@99%": round(df["repeat"].quantile(0.99), 5),
        "ent<3.5": round((df["entropy"] < 3.5).mean(), 4),
        "ent@1%": round(df["entropy"].quantile(0.01), 2),
        "TTFT": round(df["first_token_time"].mean(), 2),
        "TPS": round(tps, 2),
        "ACC": round(df["correct"].mean() * 100, 2),
        "Correct": correct_cnt,
        "OCRBench": correct_cnt,
    }


def evaluate_reference_results(
    result_get,
    verbal=False,
    save_summary_path=None,
    save_writer=None,
    sheet_prefix="",
):
    df_results = result_get.copy()
    if "category" not in df_results.columns:
        df_results["category"] = df_results["question_type"].map(
            QUESTION_TYPE_TO_CATEGORY
        ).fillna("Other")

    summary_row = _overall_row(df_results, project_name="OCRBench", flavor="overall")
    df_summary = pd.DataFrame([summary_row])
    print(f"\n[Summary] {sheet_prefix}")
    print(df_summary.to_string(index=False))

    df_category = None
    df_breakdown = None
    if verbal:
        print(f"\n[Category breakdown] {sheet_prefix}")
        cat_rows = []
        for cat, df_c in df_results.groupby("category", dropna=False):
            decode_sum = df_c["decode_time"].dropna().sum()
            total_sum = df_c["total_time"].sum()
            ftt_sum = df_c["first_token_time"].dropna().sum()
            denom = max(1e-5, total_sum - ftt_sum)
            correct_cnt = int(df_c["correct"].sum())
            cat_rows.append({
                "Project": "OCRBench",
                "Flavor": cat,
                "样本数": len(df_c),
                "Correct": correct_cnt,
                "prompt len": round(df_c["prompt_token_len"].mean(), 2),
                "response_token_len": round(df_c["response_token_len"].mean(), 2),
                "TTFT": round(df_c["first_token_time"].mean(), 2),
                "TPS": round(df_c["response_token_len"].sum() / denom, 2),
                "ACC": round(df_c["correct"].mean() * 100, 2),
            })
        df_category = pd.DataFrame(cat_rows)
        print(df_category.to_string(index=False))

        print(f"\n[Breakdown by question_type] {sheet_prefix}")
        details = []
        grouped = df_results.groupby(
            ["category", "question_type"], dropna=False
        )
        for (category, qtype), df_q in grouped:
            decode_sum = df_q["decode_time"].dropna().sum()
            total_sum = df_q["total_time"].sum()
            ftt_sum = df_q["first_token_time"].dropna().sum()
            denom = max(1e-5, total_sum - ftt_sum)
            correct_cnt = int(df_q["correct"].sum())
            details.append({
                "Project": "OCRBench",
                "Flavor": category,
                "Vertical": qtype,
                "样本数": len(df_q),
                "Correct": correct_cnt,
                "prompt len": round(df_q["prompt_token_len"].mean(), 2),
                "response_token_len": round(df_q["response_token_len"].mean(), 2),
                "TTFT": round(df_q["first_token_time"].mean(), 2),
                "TPS": round(df_q["response_token_len"].sum() / denom, 2),
                "ACC": round(df_q["correct"].mean() * 100, 2),
            })
        df_breakdown = pd.DataFrame(details)
        print(df_breakdown.to_string(index=False))

    internal_writer = None
    if save_summary_path and save_writer is None:
        internal_writer = pd.ExcelWriter(save_summary_path)
        save_writer = internal_writer

    if save_writer:
        suffix = f"_{sheet_prefix}" if sheet_prefix else ""
        summary_sheet = ("summary" + suffix)[:31]
        category_sheet = ("category" + suffix)[:31]
        breakdown_sheet = ("breakdown" + suffix)[:31]

        df_summary.to_excel(save_writer, sheet_name=summary_sheet, index=False)
        print(f"[Excel] Summary written to sheet: {summary_sheet}")

        if verbal:
            if df_category is not None:
                df_category.to_excel(
                    save_writer, sheet_name=category_sheet, index=False
                )
                print(f"[Excel] Category written to sheet: {category_sheet}")
            if df_breakdown is not None:
                df_breakdown.to_excel(
                    save_writer, sheet_name=breakdown_sheet, index=False
                )
                print(f"[Excel] Breakdown written to sheet: {breakdown_sheet}")

    if internal_writer:
        internal_writer.close()

    return df_results
