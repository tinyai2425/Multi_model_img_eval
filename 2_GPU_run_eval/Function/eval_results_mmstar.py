import pandas as pd


def _overall_row(df, project_name, flavor):
    decode_sum = df["decode_time"].dropna().sum()
    tps = (
        df["response_token_len"].sum() / decode_sum if decode_sum > 0 else 0.0
    )
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
    }


def evaluate_reference_results(
    result_get,
    verbal=False,
    save_summary_path=None,
    save_writer=None,
    sheet_prefix="",
):
    df_results = result_get.copy()

    summary_row = _overall_row(df_results, project_name="MMStar", flavor="overall")
    df_summary = pd.DataFrame([summary_row])
    print(f"\n[Summary] {sheet_prefix}")
    print(df_summary.to_string(index=False))

    df_breakdown = None
    if verbal:
        print(f"\n[Breakdown] {sheet_prefix}")
        details = []
        grouped = df_results.groupby(["category", "l2_category"], dropna=False)
        for (category, l2_category), df_l2 in grouped:
            decode_sum = df_l2["decode_time"].dropna().sum()
            total_sum = df_l2["total_time"].sum()
            ftt_sum = df_l2["first_token_time"].dropna().sum()
            denom = max(1e-5, total_sum - ftt_sum)
            details.append({
                "Project": "MMStar",
                "Flavor": category,
                "Vertical": l2_category,
                "样本数": len(df_l2),
                "prompt len": round(df_l2["prompt_token_len"].mean(), 2),
                "response_token_len": round(df_l2["response_token_len"].mean(), 2),
                "TTFT": round(df_l2["first_token_time"].mean(), 2),
                "TPS": round(df_l2["response_token_len"].sum() / denom, 2),
                "ACC": round(df_l2["correct"].mean() * 100, 2),
            })

        df_breakdown = pd.DataFrame(details)
        print(df_breakdown.to_string(index=False))

    internal_writer = None
    if save_summary_path and save_writer is None:
        internal_writer = pd.ExcelWriter(save_summary_path)
        save_writer = internal_writer

    if save_writer:
        summary_sheet = f"summary_{sheet_prefix}" if sheet_prefix else "summary"
        breakdown_sheet = f"breakdown_{sheet_prefix}" if sheet_prefix else "breakdown"
        summary_sheet = summary_sheet[:31]
        breakdown_sheet = breakdown_sheet[:31]

        df_summary.to_excel(save_writer, sheet_name=summary_sheet, index=False)
        print(f"[Excel] Summary written to sheet: {summary_sheet}")

        if verbal and df_breakdown is not None:
            df_breakdown.to_excel(save_writer, sheet_name=breakdown_sheet, index=False)
            print(f"[Excel] Breakdown written to sheet: {breakdown_sheet}")

    if internal_writer:
        internal_writer.close()

    return df_results
