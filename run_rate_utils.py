def calculate_risk_scores(df_all, run_interval_hours=8):
    """
    Calculates the Risk Scores for the Risk Tower.
    
    Logic:
    1. Filter for the last 4 weeks of data per tool.
    2. Split data into PRODUCTION RUNS based on the 'run_interval_hours' gap threshold.
       - Any gap > run_interval_hours is treated as a break between runs (not downtime).
    3. Calculate Base Stability Score (0-100) using sum of run durations.
    4. Check for Trend (Split 4-week period into two halves).
       - If stability dropped > 5%, apply penalty (20 points).
    5. Determine Primary Risk Factor:
       - Declining Trend
       - High MTTR (> 30 min)
       - Frequent Stops (MTBF < 60 min)
       - Low Stability
    """
    initial_metrics = []
    
    if df_all.empty or 'tool_id' not in df_all.columns:
        return pd.DataFrame()
    
    RUN_INTERVAL_SEC = run_interval_hours * 3600

    for tool_id, df_tool in df_all.groupby('tool_id'):
        df_tool = df_tool.sort_values('shot_time')
        if df_tool.empty: continue
        
        # Determine analysis period (last 4 weeks of data present)
        max_date = df_tool['shot_time'].max()
        cutoff_date = max_date - timedelta(weeks=4)
        df_period = df_tool[df_tool['shot_time'] >= cutoff_date].copy()
        
        if df_period.empty: continue

        # Use Calculator for metrics
        # tolerance/gap defaults: 0.01 (1%), 2.0s
        calc = RunRateCalculator(df_period, 0.01, 2.0, analysis_mode='aggregate')
        res = calc.results
        
        # Get processed_df which contains time_diff_sec
        df_processed = res.get('processed_df')
        if df_processed is None or df_processed.empty:
            continue
        
        # Identify Runs based on the passed Gap Threshold
        is_new_run = df_processed['time_diff_sec'] > RUN_INTERVAL_SEC
        df_processed['run_id_risk'] = is_new_run.cumsum()
        df_processed['run_label'] = df_processed['run_id_risk'].apply(lambda x: f'Run_{x}')
        
        # Calculate summary metrics by summing up individual runs (excludes large gaps)
        run_summary_df = calculate_run_summaries(df_processed, 0.05, 2.0)
        
        if run_summary_df.empty:
            continue
                
        total_runtime_sec = run_summary_df['total_runtime_sec'].sum()
        production_time_sec = run_summary_df['production_time_sec'].sum()
        downtime_sec = run_summary_df['downtime_sec'].sum()
        stop_events = run_summary_df['stops'].sum()

        res_stability = (production_time_sec / total_runtime_sec * 100) if total_runtime_sec > 0 else 100.0
        res_mttr = (downtime_sec / 60 / stop_events) if stop_events > 0 else 0
        res_mtbf = (production_time_sec / 60 / stop_events) if stop_events > 0 else (production_time_sec / 60)
        
        # --- Weekly Stability Calculation ---
        weekly_stats = []
        df_processed['week'] = df_processed['shot_time'].dt.isocalendar().week
        
        # Group by week but sort groups by time
        weekly_groups = df_processed.groupby('week')
        sorted_weeks = []
        for w_num, g_df in weekly_groups:
            if not g_df.empty:
                sorted_weeks.append((g_df['shot_time'].min(), w_num, g_df))
        
        sorted_weeks.sort(key=lambda x: x[0])
        
        for _, week_num, df_week_full in sorted_weeks:
            df_week = df_week_full.copy()
            # Must re-run interval logic for the weekly slice if gaps exist WITHIN the week slice
            is_new_run_week = df_week['time_diff_sec'] > RUN_INTERVAL_SEC
            df_week['run_id_week'] = is_new_run_week.cumsum()
            df_week['run_label'] = df_week['run_id_week'].apply(lambda x: f'WeekRun_{x}')
            
            weekly_run_summary = calculate_run_summaries(df_week, 0.05, 2.0)
            
            if not weekly_run_summary.empty:
                w_tot_runtime = weekly_run_summary['total_runtime_sec'].sum()
                w_prod_time = weekly_run_summary['production_time_sec'].sum()
                w_stability = (w_prod_time / w_tot_runtime * 100) if w_tot_runtime > 0 else 100.0
                weekly_stats.append({'week': week_num, 'stability': w_stability})
            else:
                pass # Skip empty weeks

        
        weekly_stabilities_df = pd.DataFrame(weekly_stats)
        weekly_stabilities = weekly_stabilities_df['stability'].tolist() if not weekly_stabilities_df.empty else []

        trend = "Stable"
        if len(weekly_stabilities) > 1 and weekly_stabilities[-1] < weekly_stabilities[0] * 0.95:
            trend = "Declining"

        initial_metrics.append({
            'Tool ID': tool_id,
            'Stability': res_stability,
            'MTTR': res_mttr,
            'MTBF': res_mtbf,
            'Weekly Stability': ' → '.join([f'{s:.0f}%' for s in weekly_stabilities]),
            'Trend': trend,
            'Analysis Period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        })

    if not initial_metrics:
        return pd.DataFrame()

    # --- Second pass: Determine risk factors ---
    metrics_df = pd.DataFrame(initial_metrics)
    
    overall_mttr_mean = metrics_df['MTTR'].mean()
    overall_mtbf_mean = metrics_df['MTBF'].mean()

    final_risk_data = []
    for _, row in metrics_df.iterrows():
        risk_score = row['Stability']
        if row['Trend'] == "Declining":
            risk_score -= 20
        
        primary_factor = "Low Stability"
        details = f"Overall stability is {row['Stability']:.1f}%."
        
        if row['Trend'] == "Declining":
            primary_factor = "Declining Trend"
            details = "Declining stability"
        elif row['Stability'] < 70 and overall_mttr_mean > 0 and row['MTTR'] > (overall_mttr_mean * 1.2):
            primary_factor = "High MTTR"
            details = f"Avg stop duration (MTTR) of {row['MTTR']:.1f} min is high."
        elif row['Stability'] < 70 and overall_mtbf_mean > 0 and row['MTBF'] < (overall_mtbf_mean * 0.8):
            primary_factor = "Frequent Stops"
            details = f"Frequent stops (MTBF of {row['MTBF']:.1f} min)."
        elif row['Stability'] < 60:
             risk_factor = "Low Overall Stability"
             details = f"Overall stability is critical ({row['Stability']:.1f}%)."
        
        final_risk_data.append({
            'Tool ID': row['Tool ID'],
            'Analysis Period': row['Analysis Period'],
            'Risk Score': max(0, risk_score),
            'Primary Risk Factor': primary_factor,
            'Weekly Stability': row['Weekly Stability'],
            'Details': details
        })

    if not final_risk_data:
        return pd.DataFrame()
            
    return pd.DataFrame(final_risk_data).sort_values('Risk Score', ascending=True).reset_index(drop=True)