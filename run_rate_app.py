import streamlit as st
import pandas as pd
import numpy as np
import warnings
import streamlit.components.v1 as components
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import importlib

# Import all logic functions from the utils file
import run_rate_utils as rr_utils
importlib.reload(rr_utils) # Forces update of utils on every run

# ==============================================================================
# --- 1. PAGE CONFIG & SETUP ---
# ==============================================================================

warnings.filterwarnings("ignore", category=FutureWarning)
# Check if page config is already set to avoid errors during re-runs
try:
    st.set_page_config(layout="wide", page_title="Run Rate Analysis Dashboard")
except:
    pass

# ==============================================================================
# --- 2. UI RENDERING FUNCTIONS ---
# ==============================================================================

def render_risk_tower(df_all_tools):
    """Renders the Risk Tower tab."""
    st.title("Run Rate Risk Tower")
    st.info("This tower analyzes performance over the last 4 weeks, identifying tools that require attention. Tools with the lowest scores are at the highest risk.")
    
    with st.expander("ℹ️ How the Risk Tower Works"):
        st.markdown("""
        The Risk Tower evaluates each tool based on its performance over its own most recent 4-week period of operation. Here’s how the metrics are calculated:

        - **Analysis Period**: Shows the exact 4-week date range used for each tool's analysis, based on its latest available data.
        - **Risk Score**: A performance indicator from 0-100.
            - It starts with the tool's overall **Stability Index (%)** for the period.
            - A **20-point penalty** is applied if the stability shows a declining trend.
        - **Primary Risk Factor**: Identifies the main issue affecting performance, prioritized as follows:
            1.  **Declining Trend**: If stability is worsening over time.
            2.  **High MTTR**: If the average stop duration is significantly longer than the average of all tools.
            3.  **Frequent Stops**: If the time between stops (MTBF) is significantly shorter than the average of all tools.
            4.  **Low Stability**: If none of the above are true, but overall stability is low.
        - **Color Coding**: Rows are colored based on the Risk Score:
            - <span style='background-color:#ff6961; color: black; padding: 2px 5px; border-radius: 5px;'>Red (0-50)</span>: High Risk
            - <span style='background-color:#ffb347; color: black; padding: 2px 5px; border-radius: 5px;'>Orange (51-70)</span>: Medium Risk
            - <span style='background-color:#77dd77; color: black; padding: 2px 5px; border-radius: 5px;'>Green (>70)</span>: Low Risk
        """, unsafe_allow_html=True)

    # Calculate Risk Scores using the Utils function
    risk_df = rr_utils.calculate_risk_scores(df_all_tools)

    if risk_df.empty:
        st.warning("Not enough data across multiple tools in the last 4 weeks to generate a risk tower.")
        return

    def style_risk(row):
        score = row['Risk Score']
        if score > 70: color = rr_utils.PASTEL_COLORS['green']
        elif score > 50: color = rr_utils.PASTEL_COLORS['orange']
        else: color = rr_utils.PASTEL_COLORS['red']
        return [f'background-color: {color}' for _ in row]
    
    cols_order = ['Tool ID', 'Analysis Period', 'Risk Score', 'Primary Risk Factor', 'Weekly Stability', 'Details']
    display_df = risk_df[[col for col in cols_order if col in risk_df.columns]]

    st.dataframe(display_df.style.apply(style_risk, axis=1).format({'Risk Score': '{:.0f}'}), use_container_width=True, hide_index=True)


def render_trends_tab(df_tool, tolerance, downtime_gap_tolerance, run_interval_hours):
    """Renders the new Trends Analysis tab."""
    st.header("Historical Performance Trends")
    st.info(f"Trends are calculated using 'Run-Based' logic. Gaps larger than {run_interval_hours} hours are excluded from the timeline to provide accurate stability metrics.")
    
    col_ctrl, _ = st.columns([1, 3])
    with col_ctrl:
        trend_freq = st.selectbox("Select Trend Frequency", ["Daily", "Weekly", "Monthly"], key="trend_freq_select")
    
    with st.expander("ℹ️ About Trends Metrics"):
        st.markdown("""
        This table compares performance metrics side-by-side across time periods.
        
        - **Stability Index (%)**: Percentage of run time spent in production.
        - **Efficiency (%)**: Percentage of shots that were normal (non-stops).
        - **MTTR (min)**: Mean Time To Repair (avg stop duration).
        - **MTBF (min)**: Mean Time Between Failures (avg uptime).
        - **Total Shots**: Total output for the period.
        - **Stop Events**: Number of times the machine stopped.
        """)

    # --- Logic to Generate Trend Data ---
    trend_data = []
    
    # 1. Group Data
    if trend_freq == "Daily":
        grouper = df_tool.groupby(df_tool['shot_time'].dt.date)
        period_name = "Date"
    elif trend_freq == "Weekly":
        # Group by Year-Week
        grouper = df_tool.groupby(df_tool['shot_time'].dt.to_period('W'))
        period_name = "Week"
    else: # Monthly
        grouper = df_tool.groupby(df_tool['shot_time'].dt.to_period('M'))
        period_name = "Month"

    # 2. Iterate and Calculate Metrics (Run-Based)
    for period, df_period in grouper:
        if df_period.empty: continue
        
        # A. Pre-process to identify runs within this period
        calc_prep = rr_utils.RunRateCalculator(df_period, tolerance, downtime_gap_tolerance, analysis_mode='aggregate')
        df_p = calc_prep.results.get("processed_df", pd.DataFrame())
        
        if df_p.empty: continue

        # Split into runs
        is_new_run = df_p['time_diff_sec'] > (run_interval_hours * 3600)
        df_p['run_label'] = is_new_run.cumsum()
        
        # B. Calculate summaries per run
        run_summaries = rr_utils.calculate_run_summaries(df_p, tolerance, downtime_gap_tolerance)
        
        if run_summaries.empty: continue

        # C. Aggregate totals for the period
        total_runtime = run_summaries['total_runtime_sec'].sum()
        prod_time = run_summaries['production_time_sec'].sum()
        downtime = run_summaries['downtime_sec'].sum()
        stops = run_summaries['stops'].sum()
        total_shots = run_summaries['total_shots'].sum()
        normal_shots = run_summaries['normal_shots'].sum()

        # D. Calculate derived metrics
        stability = (prod_time / total_runtime * 100) if total_runtime > 0 else 0
        efficiency = (normal_shots / total_shots * 100) if total_shots > 0 else 0
        mttr = (downtime / 60 / stops) if stops > 0 else 0
        mtbf = (prod_time / 60 / stops) if stops > 0 else (prod_time / 60)

        # Format Period Label
        if trend_freq == "Daily":
            label = period.strftime('%Y-%m-%d')
        elif trend_freq == "Weekly":
            label = f"W{period.week} {period.year}" 
        else:
            label = period.strftime('%B %Y')

        trend_data.append({
            period_name: label,
            'SortKey': period if trend_freq == "Daily" else period.start_time,
            'Stability Index (%)': stability,
            'Efficiency (%)': efficiency,
            'MTTR (min)': mttr,
            'MTBF (min)': mtbf,
            'Total Shots': total_shots,
            'Stop Events': stops,
            'Production Time (h)': prod_time / 3600,
            'Downtime (h)': downtime / 3600
        })
    
    if not trend_data:
        st.warning("No data found for the selected tool to generate trends.")
        return

    # 3. Create DataFrame
    df_trends = pd.DataFrame(trend_data).sort_values('SortKey', ascending=False)
    df_trends = df_trends.drop(columns=['SortKey'])
    
    # 4. Styling
    st.dataframe(
        df_trends.style.format({
            'Stability Index (%)': '{:.1f}',
            'Efficiency (%)': '{:.1f}',
            'MTTR (min)': '{:.1f}',
            'MTBF (min)': '{:.1f}',
            'Total Shots': '{:,.0f}',
            'Stop Events': '{:,.0f}',
            'Production Time (h)': '{:.1f}',
            'Downtime (h)': '{:.1f}'
        }).background_gradient(subset=['Stability Index (%)'], cmap='RdYlGn', vmin=0, vmax=100),
        use_container_width=True
    )
    
    # 5. Visual Trend Chart
    st.subheader("Visual Trend")
    metric_to_plot = st.selectbox("Select Metric to Visualize", 
                                  ['Stability Index (%)', 'Efficiency (%)', 'MTTR (min)', 'MTBF (min)', 'Total Shots'],
                                  key="trend_viz_select")
    
    fig = px.line(df_trends.sort_index(ascending=False), x=period_name, y=metric_to_plot, markers=True, 
                  title=f"{metric_to_plot} Trend ({trend_freq})")
    
    # Add coloring for stability/efficiency
    if '%)' in metric_to_plot:
        fig.add_shape(type="rect", xref="paper", x0=0, x1=1, y0=0, y1=50, fillcolor=rr_utils.PASTEL_COLORS['red'], opacity=0.1, layer="below", line_width=0)
        fig.add_shape(type="rect", xref="paper", x0=0, x1=1, y0=50, y1=70, fillcolor=rr_utils.PASTEL_COLORS['orange'], opacity=0.1, layer="below", line_width=0)
        fig.add_shape(type="rect", xref="paper", x0=0, x1=1, y0=70, y1=100, fillcolor=rr_utils.PASTEL_COLORS['green'], opacity=0.1, layer="below", line_width=0)
        fig.update_yaxes(range=[0, 105])

    st.plotly_chart(fig, use_container_width=True)


def render_dashboard(df_tool, tool_id_selection, tolerance, downtime_gap_tolerance, run_interval_hours):
    """Renders the main Run Rate Dashboard tab."""
    
    # Internal Dashboard Controls (Level selection only, others moved to global sidebar)
    analysis_level = st.radio(
        "Select Analysis Level",
        options=["Daily (by Run)", "Weekly (by Run)", "Monthly (by Run)", "Custom Period (by Run)"],
        horizontal=True,
        key="rr_analysis_level"
    )
    
    st.markdown("---")

    @st.cache_data(show_spinner="Performing initial data processing...")
    def get_processed_data(df, interval_hours):
        base_calc = rr_utils.RunRateCalculator(df, 0.01, 2.0)
        df_processed = base_calc.results.get("processed_df", pd.DataFrame())
        if not df_processed.empty:
            df_processed['week'] = df_processed['shot_time'].dt.isocalendar().week
            df_processed['date'] = df_processed['shot_time'].dt.date
            df_processed['month'] = df_processed['shot_time'].dt.to_period('M')
            is_new_run = df_processed['time_diff_sec'] > (interval_hours * 3600)
            df_processed['run_id'] = is_new_run.cumsum()
        return df_processed

    df_processed = get_processed_data(df_tool, run_interval_hours)
    
    # Run Filter logic
    min_shots_filter = 1 
    if 'by Run' in analysis_level:
        enable_run_filter = st.toggle("Filter Small Production Runs", value=False, key="rr_enable_run_filter")
        
        if enable_run_filter and not df_processed.empty:
            run_shot_counts = df_processed.groupby('run_id').size()
            if not run_shot_counts.empty:
                max_shots = int(run_shot_counts.max()) if not run_shot_counts.empty else 1
                default_value = min(10, max_shots) if max_shots > 1 else 1
                min_shots_filter = st.slider(
                    "Remove Runs with Fewer Than X Shots",
                    min_value=1,
                    max_value=max_shots,
                    value=default_value,
                    step=1,
                    key="rr_min_shots_filter",
                    help="Filters out smaller production runs to focus on more significant ones."
                )
    
    detailed_view = st.toggle("Show Detailed Analysis", value=True, key="rr_detailed_view")

    if df_processed.empty:
        st.error(f"Could not process data for {tool_id_selection}. Check file format or data range."); st.stop()

    st.markdown(f"### {tool_id_selection} Overview")

    mode = 'by_run'
    df_view = pd.DataFrame()

    # --- Date/Period Selection ---
    if "Daily" in analysis_level: 
        available_dates = sorted(df_processed["date"].unique())
        if not available_dates:
            st.warning("No data available for any date."); st.stop()
        selected_date = st.selectbox("Select Date", options=available_dates, index=len(available_dates)-1, format_func=lambda d: pd.to_datetime(d).strftime('%d %b %Y'), key="rr_daily_select")
        df_view = df_processed[df_processed["date"] == selected_date]
        sub_header = f"Summary for {pd.to_datetime(selected_date).strftime('%d %b %Y')}"

    elif "Weekly" in analysis_level:
        available_weeks = sorted(df_processed["week"].unique())
        if not available_weeks:
            st.warning("No data available for any week."); st.stop()
        year = df_processed['shot_time'].iloc[0].year
        selected_week = st.selectbox(f"Select Week (Year {year})", options=available_weeks, index=len(available_weeks)-1, key="rr_weekly_select")
        df_view = df_processed[df_processed["week"] == selected_week]
        sub_header = f"Summary for Week {selected_week}"

    elif "Monthly" in analysis_level:
        available_months = sorted(df_processed["month"].unique())
        if not available_months:
            st.warning("No data available for any month."); st.stop()
        selected_month = st.selectbox(f"Select Month", options=available_months, index=len(available_months)-1, format_func=lambda p: p.strftime('%B %Y'), key="rr_monthly_select")
        df_view = df_processed[df_processed["month"] == selected_month]
        sub_header = f"Summary for {selected_month.strftime('%B %Y')}"

    elif "Custom Period" in analysis_level:
        min_date = df_processed['date'].min()
        max_date = df_processed['date'].max()
        c1, c2 = st.columns(2)
        with c1: start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date, key="rr_custom_start")
        with c2: end_date = st.date_input("End date", max_date, min_value=start_date, max_value=max_date, key="rr_custom_end")
        if start_date and end_date:
            mask = (df_processed['date'] >= start_date) & (df_processed['date'] <= end_date)
            df_view = df_processed[mask]
            sub_header = f"Summary for {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"

    # --- Run Filtering and Labeling ---
    if not df_view.empty:
        df_view = df_view.copy()
        if 'run_id' in df_view.columns:
            df_view['run_id_local'] = df_view.groupby('run_id').ngroup()
            unique_run_ids = df_view.sort_values('shot_time')['run_id_local'].unique()
            run_label_map = {run_id: f"Run {i+1:03d}" for i, run_id in enumerate(unique_run_ids)}
            df_view['run_label'] = df_view['run_id_local'].map(run_label_map)

    if 'by Run' in analysis_level and not df_view.empty:
        # runs_before_filter = df_view['run_label'].nunique()
        run_shot_counts_in_view = df_view.groupby('run_label')['run_label'].transform('count')
        df_view = df_view[run_shot_counts_in_view >= min_shots_filter]
        # runs_after_filter = df_view['run_label'].nunique()

    if df_view.empty:
        st.warning(f"No data for the selected period (or all runs were filtered out).")
    else:
        # --- Main Calculation for Selected Period ---
        results = {}
        summary_metrics = {}
        run_summary_df_for_totals = pd.DataFrame()
        run_summary_df = None # Initialize to prevent NameError
        trend_summary_df = None 

        run_summary_df_for_totals = rr_utils.calculate_run_summaries(df_view, tolerance, downtime_gap_tolerance)
        
        if not run_summary_df_for_totals.empty:
            total_runtime_sec = run_summary_df_for_totals['total_runtime_sec'].sum()
            production_time_sec = run_summary_df_for_totals['production_time_sec'].sum()
            downtime_sec = run_summary_df_for_totals['downtime_sec'].sum()
            total_shots = run_summary_df_for_totals['total_shots'].sum()
            normal_shots = run_summary_df_for_totals['normal_shots'].sum()
            stop_events = run_summary_df_for_totals['stops'].sum()

            mttr_min = (downtime_sec / 60 / stop_events) if stop_events > 0 else 0
            mtbf_min = (production_time_sec / 60 / stop_events) if stop_events > 0 else (production_time_sec / 60)
            stability_index = (production_time_sec / total_runtime_sec * 100) if total_runtime_sec > 0 else 100.0
            efficiency = (normal_shots / total_shots) if total_shots > 0 else 0

            summary_metrics = {
                'total_runtime_sec': total_runtime_sec, 'production_time_sec': production_time_sec,
                'downtime_sec': downtime_sec, 'total_shots': total_shots,
                'normal_shots': normal_shots, 'stop_events': stop_events,
                'mttr_min': mttr_min, 'mtbf_min': mtbf_min,
                'stability_index': stability_index, 'efficiency': efficiency,
            }
            sub_header = sub_header.replace("Summary for", "Summary for (Combined Runs)")

        calc = rr_utils.RunRateCalculator(df_view.copy(), tolerance, downtime_gap_tolerance, analysis_mode=mode)
        results = calc.results
        summary_metrics.update({
            'min_lower_limit': results.get('min_lower_limit', 0), 'max_lower_limit': results.get('max_lower_limit', 0),
            'min_mode_ct': results.get('min_mode_ct', 0), 'max_mode_ct': results.get('max_mode_ct', 0),
            'min_upper_limit': results.get('min_upper_limit', 0), 'max_upper_limit': results.get('max_upper_limit', 0),
        })

        # --- Header & Download Button ---
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(sub_header)
        with col2:
            st.download_button(
                label="📥 Export Run-Based Report",
                data=rr_utils.prepare_and_generate_run_based_excel(
                    df_view.copy(), tolerance, downtime_gap_tolerance,
                    run_interval_hours, tool_id_selection
                ),
                file_name=f"Run_Based_Report_{tool_id_selection.replace(' / ', '_').replace(' ', '_')}_{analysis_level.replace(' ', '_')}_{datetime.now():%Y%m%d}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
        # --- Prepare Trend Data ---
        trend_summary_df = None
        if "by Run" in analysis_level: 
            trend_summary_df = rr_utils.calculate_run_summaries(df_view, tolerance, downtime_gap_tolerance)
            if trend_summary_df is not None and not trend_summary_df.empty:
                trend_summary_df.rename(columns={'run_label': 'RUN ID', 'stability_index': 'STABILITY %', 'stops': 'STOPS', 'mttr_min': 'MTTR (min)', 'mtbf_min': 'MTBF (min)', 'total_shots': 'Total Shots'}, inplace=True)
            run_summary_df = trend_summary_df 

        # --- 1. KPI Metrics Display ---
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            total_d = summary_metrics.get('total_runtime_sec', 0); prod_t = summary_metrics.get('production_time_sec', 0); down_t = summary_metrics.get('downtime_sec', 0)
            prod_p = (prod_t / total_d * 100) if total_d > 0 else 0
            down_p = (down_t / total_d * 100) if total_d > 0 else 0
            
            mttr_val_min = summary_metrics.get('mttr_min', 0)
            mtbf_val_min = summary_metrics.get('mtbf_min', 0)
            
            # Use utility for consistent formatting
            mttr_display = f"{mttr_val_min:.1f} min"
            mtbf_display = f"{mtbf_val_min:.1f} min"

            with col1: 
                st.metric("Run Rate MTTR", mttr_display,
                          help="Mean Time To Repair: The average duration of a single stop event.\n\nFormula: Total Downtime / Stop Events")
            with col2: 
                st.metric("Run Rate MTBF", mtbf_display,
                          help="Mean Time Between Failures: The average duration of stable operation *between* stop events.\n\nFormula: Total Production Time / Stop Events")
            
            with col3: 
                st.metric("Total Run Duration", rr_utils.format_duration(total_d),
                          help="The total time the machine was running. This is the SUM of the durations of all individual production runs. Gaps between runs (e.g., weekends) defined by the 'Run Interval Threshold' are excluded.\n\nFormula (per run): (Time of Last Shot - Time of First Shot) + (Actual CT of Last Shot)")
            with col4:
                st.metric("Production Time", f"{rr_utils.format_duration(prod_t)}",
                          help="The total time spent producing parts. This is the sum of the 'Actual CT' for all 'Normal' (non-stop) shots across all runs.\n\nFormula: Sum(Actual CT of all Normal Shots)")
                st.markdown(f'<span style="background-color: {rr_utils.PASTEL_COLORS["green"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{prod_p:.1f}%</span>', unsafe_allow_html=True)
            with col5:
                st.metric("Downtime", f"{rr_utils.format_duration(down_t)}",
                          help="The total time the machine was stopped within the calculated 'Total Run Duration'. This is the 'plug figure' calculated from the script's logic.\n\nFormula: Total Run Duration - Total Production Time")
                st.markdown(f'<span style="background-color: {rr_utils.PASTEL_COLORS["red"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{down_p:.1f}%</span>', unsafe_allow_html=True)
        
        with st.container(border=True):
            c1, c2 = st.columns(2)
            c1.plotly_chart(rr_utils.create_gauge(summary_metrics.get('efficiency', 0) * 100, "Run Rate Efficiency (%)"), use_container_width=True)
            steps = [{'range': [0, 50], 'color': rr_utils.PASTEL_COLORS['red']}, {'range': [50, 70], 'color': rr_utils.PASTEL_COLORS['orange']},{'range': [70, 100], 'color': rr_utils.PASTEL_COLORS['green']}]
            c2.plotly_chart(rr_utils.create_gauge(summary_metrics.get('stability_index', 0), "Run Rate Stability Index (%)", steps=steps), use_container_width=True)

        with st.expander("ℹ️ What do these metrics mean?"):
            st.markdown("""
            **Run Rate Efficiency (%)**
            > The percentage of shots that were 'Normal' (i.e., `stop_flag` = 0).
            > - *Formula: `Normal Shots / Total Shots`*

            **Run Rate Stability Index (%)**
            > The percentage of total run time that was spent in a 'Normal' production state.
            > - *Formula: `Total Production Time / Total Run Duration`*
            
            **Run Rate MTTR (min)**
            > Mean Time To Repair. The average duration of a single stop event.
            > - *Formula: `Total Downtime / Stop Events`*
            
            **Run Rate MTBF (min)**
            > Mean Time Between Failures. The average duration of stable operation *between* stop events.
            > - *Formula: `Total Production Time / Stop Events`*
            
            **Total Run Duration (sec)**
            > The sum of durations of all individual production runs, excluding gaps > 'Run Interval Threshold'.
            
            **Production Time (sec)**
            > The sum of 'Actual CT' for all normal shots.
            
            **Downtime (sec)**
            > `Total Run Duration - Total Production Time`.
            """)
        
        with st.container(border=True):
            c1,c2,c3 = st.columns(3)
            t_s = summary_metrics.get('total_shots', 0); n_s = summary_metrics.get('normal_shots', 0)
            s_s = t_s - n_s
            n_p = (n_s / t_s * 100) if t_s > 0 else 0
            s_p = (s_s / t_s * 100) if t_s > 0 else 0
            with c1: 
                st.metric("Total Shots", f"{t_s:,}", help="The total number of shots (rows) in the selected period.")
            with c2:
                st.metric("Normal Shots", f"{n_s:,}", help="The count of all shots that were *not* flagged as stops.")
                st.markdown(f'<span style="background-color: {rr_utils.PASTEL_COLORS["green"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{n_p:.1f}% of Total</span>', unsafe_allow_html=True)
            with c3:
                st.metric("Stop Events", f"{summary_metrics.get('stop_events', 0)}", help="The count of *new* stoppage incidents.")
                st.markdown(f'<span style="background-color: {rr_utils.PASTEL_COLORS["red"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{s_p:.1f}% Stopped Shots</span>', unsafe_allow_html=True)

        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            min_ll = summary_metrics.get('min_lower_limit', 0); max_ll = summary_metrics.get('max_lower_limit', 0)
            c1.metric("Lower Limit Range (sec)", f"{min_ll:.2f} – {max_ll:.2f}")
            with c2:
                min_mc = summary_metrics.get('min_mode_ct', 0); max_mc = summary_metrics.get('max_mode_ct', 0)
                with st.container(border=True): st.metric("Mode Cycle Time Range (sec)", f"{min_mc:.2f} – {max_mc:.2f}")
            min_ul = summary_metrics.get('min_upper_limit', 0); max_ul = summary_metrics.get('max_upper_limit', 0)
            c3.metric("Upper Limit Range (sec)", f"{min_ul:.2f} – {max_ul:.2f}")
        
        # --- Detailed Analysis Expander ---
        if detailed_view:
            st.markdown("---")
            with st.expander("🤖 View Automated Analysis Summary", expanded=False):
                analysis_df = pd.DataFrame()
                if trend_summary_df is not None and not trend_summary_df.empty:
                    analysis_df = trend_summary_df.copy()
                    rename_map = {}
                    if 'RUN ID' in analysis_df.columns: rename_map = {'RUN ID': 'period', 'STABILITY %': 'stability', 'STOPS': 'stops', 'MTTR (min)': 'mttr'}
                    analysis_df.rename(columns=rename_map, inplace=True)
                
                insights = rr_utils.generate_detailed_analysis(analysis_df, summary_metrics.get('stability_index', 0), summary_metrics.get('mttr_min', 0), summary_metrics.get('mtbf_min', 0), analysis_level)
                
                if "error" in insights: 
                    st.error(insights["error"])
                else:
                    st.components.v1.html(f"""<div style="border:1px solid #333;border-radius:0.5rem;padding:1.5rem;margin-top:1rem;font-family:sans-serif;line-height:1.6;background-color:#0E1117;"><h4 style="margin-top:0;color:#FAFAFA;">Automated Analysis Summary</h4><p style="color:#FAFAFA;"><strong>Overall Assessment:</strong> {insights['overall']}</p><p style="color:#FAFAFA;"><strong>Predictive Trend:</strong> {insights['predictive']}</p><p style="color:#FAFAFA;"><strong>Performance Variance:</strong> {insights['best_worst']}</p> {'<p style="color:#FAFAFA;"><strong>Identified Patterns:</strong> ' + insights['patterns'] + '</p>' if insights['patterns'] else ''}<p style="margin-top:1rem;color:#FAFAFA;background-color:#262730;padding:1rem;border-radius:0.5rem;"><strong>Key Recommendation:</strong> {insights['recommendation']}</p></div>""", height=400, scrolling=True)

        st.markdown("---")

        # --- 3. Main Shot Bar Chart & Data ---
        time_agg = 'hourly' if "Daily" in analysis_level else 'daily' if 'Weekly' in analysis_level else 'weekly'
        rr_utils.plot_shot_bar_chart(results['processed_df'], results.get('lower_limit'), results.get('upper_limit'), results.get('mode_ct'), time_agg=time_agg)
        
        with st.expander("View Shot Data Table", expanded=False):
            cols_to_show = ['shot_time', 'ACTUAL CT', 'adj_ct_sec', 'time_diff_sec', 'stop_flag', 'stop_event']
            rename_map = {
                'shot_time': 'Date / Time',
                'ACTUAL CT': 'Actual CT (sec)',
                'adj_ct_sec': 'Adjusted CT (sec)',
                'time_diff_sec': 'Time Difference (sec)',
                'stop_flag': 'Stop Flag',
                'stop_event': 'Stop Event'
            }
            if 'run_label' in results['processed_df'].columns:
                cols_to_show.append('run_label')
                rename_map['run_label'] = 'Run ID'
                
            df_shot_data = results['processed_df'][cols_to_show].copy()
            df_shot_data.rename(columns=rename_map, inplace=True)
            st.dataframe(df_shot_data)

        st.markdown("---")
        
        # --- 4. Detailed Analysis Section (Run/Hour Switch) ---
        
        # Default view mode
        analysis_view_mode = "Run"
        
        # Only show the toggle if we are in Daily mode
        if analysis_level == "Daily (by Run)":
            c_head, c_view = st.columns([3,1])
            with c_head:
                st.header("Detailed Analysis")
            with c_view:
                analysis_view_mode = st.selectbox("Group By", ["Run", "Hour"], key="rr_view_mode")
        else:
             st.header(f"Run-Based Analysis")

        # --- Prepare Data for Charts (re-used) ---
        run_durations = results.get("run_durations", pd.DataFrame())
        processed_df = results.get('processed_df', pd.DataFrame())
        stop_events_df = processed_df.loc[processed_df['stop_event']].copy()
        complete_runs = pd.DataFrame()
        if not stop_events_df.empty:
            stop_events_df['terminated_run_group'] = stop_events_df['run_group'] - 1
            end_time_map = stop_events_df.set_index('terminated_run_group')['shot_time']
            run_durations['run_end_time'] = run_durations['run_group'].map(end_time_map)
            complete_runs = run_durations.dropna(subset=['run_end_time']).copy()

        # --- Option A: Run-Based View ---
        if analysis_view_mode == "Run":
            
            # --- 4a. Run Breakdown Table ---
            with st.expander("View Run Breakdown Table", expanded=True):
                if run_summary_df is not None and not run_summary_df.empty:
                    d_df = run_summary_df.copy()
                    d_df["Period (date/time from to)"] = d_df.apply(lambda row: f"{row['start_time'].strftime('%Y-%m-%d %H:%M')} to {row['end_time'].strftime('%Y-%m-%d %H:%M')}", axis=1)
                    
                    # Handle total shots
                    if 'Total Shots' in d_df.columns:
                        d_df["Total shots"] = d_df['Total Shots'].apply(lambda x: f"{x:,}")
                    elif 'total_shots' in d_df.columns:
                        d_df["Total shots"] = d_df['total_shots'].apply(lambda x: f"{x:,}")

                    # Handle normal shots %
                    d_df["Normal Shots"] = d_df.apply(lambda r: f"{r['normal_shots']:,} ({r['normal_shots']/r['total_shots']*100:.1f}%)" if 'total_shots' in r and r['total_shots']>0 else (f"{r['normal_shots']:,} ({r['normal_shots']/r['Total Shots']*100:.1f}%)" if 'Total Shots' in r and r['Total Shots']>0 else "0 (0.0%)"), axis=1)
                    
                    # Recalculate stopped_shots if missing
                    if 'stopped_shots' not in d_df.columns:
                        if 'Total Shots' in d_df.columns:
                            d_df['stopped_shots'] = d_df['Total Shots'] - d_df['normal_shots']
                        elif 'total_shots' in d_df.columns:
                            d_df['stopped_shots'] = d_df['total_shots'] - d_df['normal_shots']

                    stops_col = 'STOPS' if 'STOPS' in d_df.columns else 'stops'
                    total_shots_col = 'Total Shots' if 'Total Shots' in d_df.columns else 'total_shots'

                    d_df["Stop Events"] = d_df.apply(lambda r: f"{r[stops_col]} ({r['stopped_shots']/r[total_shots_col]*100:.1f}%)" if r[total_shots_col]>0 else "0 (0.0%)", axis=1)
                    
                    d_df["Total Run duration (d/h/m)"] = d_df['total_runtime_sec'].apply(rr_utils.format_duration)
                    d_df["Production Time (d/h/m)"] = d_df.apply(lambda r: f"{rr_utils.format_duration(r['production_time_sec'])} ({r['production_time_sec']/r['total_runtime_sec']*100:.1f}%)" if r['total_runtime_sec']>0 else "0m (0.0%)", axis=1)
                    d_df["Downtime (d/h/m)"] = d_df.apply(lambda r: f"{rr_utils.format_duration(r['downtime_sec'])} ({r['downtime_sec']/r['total_runtime_sec']*100:.1f}%)" if r['total_runtime_sec']>0 else "0m (0.0%)", axis=1)
                    
                    rename_map = {
                        'run_label':'RUN ID', 'mode_ct':'Mode CT (for the run)',
                        'lower_limit':'Lower limit CT (sec)', 'upper_limit':'Upper Limit CT (sec)',
                        'mttr_min':'MTTR (min)', 'mtbf_min':'MTBF (min)',
                        'stability_index':'Stability (%)', 'stops':'STOPS',
                        'MTTR (min)': 'MTTR (min)', 'MTBF (min)': 'MTBF (min)', 
                        'STABILITY %': 'Stability (%)', 'STOPS': 'STOPS'
                    }
                    d_df.rename(columns=rename_map, inplace=True)
                    
                    final_cols = ['RUN ID','Period (date/time from to)','Total shots','Normal Shots','Stop Events','Mode CT (for the run)','Lower limit CT (sec)','Upper Limit CT (sec)','Total Run duration (d/h/m)','Production Time (d/h/m)','Downtime (d/h/m)','MTTR (min)','MTBF (min)','Stability (%)']
                    final_cols = [col for col in final_cols if col in d_df.columns]
                    
                    format_dict = {}
                    if 'Mode CT (for the run)' in d_df.columns: format_dict['Mode CT (for the run)'] = '{:.2f}'
                    if 'Lower limit CT (sec)' in d_df.columns: format_dict['Lower limit CT (sec)'] = '{:.2f}'
                    if 'Upper Limit CT (sec)' in d_df.columns: format_dict['Upper Limit CT (sec)'] = '{:.2f}'
                    if 'MTTR (min)' in d_df.columns: format_dict['MTTR (min)'] = '{:.1f}'
                    if 'MTBF (min)' in d_df.columns: format_dict['MTBF (min)'] = '{:.1f}'
                    if 'Stability (%)' in d_df.columns: format_dict['Stability (%)'] = '{:.1f}'
                    
                    st.dataframe(d_df[final_cols].style.format(format_dict), use_container_width=True)

            # --- 4b. Total Buckets & Stability (Run) ---
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Total Bucket Analysis")
                if not complete_runs.empty and "time_bucket" in complete_runs.columns:
                    b_counts = complete_runs["time_bucket"].value_counts().reindex(results["bucket_labels"], fill_value=0)
                    fig_b = px.bar(b_counts, title="Total Time Bucket Analysis", labels={"index": "Duration (min)", "value": "Occurrences"}, text_auto=True, color=b_counts.index, color_discrete_map=results["bucket_color_map"]).update_layout(legend_title_text='Duration')
                    fig_b.update_xaxes(title_text="Duration (min)")
                    fig_b.update_yaxes(title_text="Occurrences")
                    st.plotly_chart(fig_b, use_container_width=True)
                    with st.expander("View Bucket Data Table", expanded=False): 
                        df_bucket_data = complete_runs.copy()
                        cols_to_show = ['run_group', 'duration_min', 'time_bucket', 'run_end_time', 'run_label']
                        df_bucket_data = df_bucket_data[[col for col in cols_to_show if col in df_bucket_data.columns]].rename(columns={
                            'run_group': 'Run Group', 'duration_min': 'Duration (min)',
                            'time_bucket': 'Time Bucket', 'run_end_time': 'Run End Date/ Time',
                            'run_label': 'Run ID'
                        })
                        st.dataframe(df_bucket_data)
                else: st.info("No complete runs.")
            
            with c2:
                st.subheader("Stability per Production Run")
                if run_summary_df is not None and not run_summary_df.empty:
                    rr_utils.plot_trend_chart(run_summary_df, 'RUN ID', 'STABILITY %', "Stability per Run", "Run ID", "Stability (%)", is_stability=True)
                    with st.expander("View Stability Data Table", expanded=False): 
                        st.dataframe(rr_utils.get_renamed_summary_df(run_summary_df))
                else: st.info(f"No runs to analyze.")
            
            # --- 4c. Bucket Trend (Run) ---
            st.subheader("Bucket Trend per Production Run")
            if not complete_runs.empty and run_summary_df is not None and not run_summary_df.empty:
                run_group_to_label_map = processed_df.drop_duplicates('run_group')[['run_group', 'run_label']].set_index('run_group')['run_label']
                complete_runs['run_label'] = complete_runs['run_group'].map(run_group_to_label_map)
                pivot_df = pd.crosstab(index=complete_runs['run_label'], columns=complete_runs['time_bucket'].astype('category').cat.set_categories(results["bucket_labels"]))
                all_runs = run_summary_df['RUN ID']
                pivot_df = pivot_df.reindex(all_runs, fill_value=0)
                fig_bucket_trend = make_subplots(specs=[[{"secondary_y": True}]])
                for col in pivot_df.columns:
                    fig_bucket_trend.add_trace(go.Bar(name=col, x=pivot_df.index, y=pivot_df[col], marker_color=results["bucket_color_map"].get(col)), secondary_y=False)
                fig_bucket_trend.add_trace(go.Scatter(name='Total Shots', x=run_summary_df['RUN ID'], y=run_summary_df['Total Shots'], mode='lines+markers+text', text=run_summary_df['Total Shots'], textposition='top center', line=dict(color='blue')), secondary_y=True)
                fig_bucket_trend.update_layout(barmode='stack', title_text='Distribution of Run Durations per Run vs. Shot Count', 
                                                xaxis_title='Run ID', yaxis_title='Number of Runs', 
                                                yaxis2_title='Total Shots', legend_title_text='Run Duration (min)')
                st.plotly_chart(fig_bucket_trend, use_container_width=True)
                with st.expander("View Bucket Trend Data Table & Analysis", expanded=False):
                    st.dataframe(pivot_df)
                    if detailed_view:
                        st.markdown(rr_utils.generate_bucket_analysis(complete_runs, results["bucket_labels"]), unsafe_allow_html=True)
            
            # --- 4d. MTTR & MTBF (Run) ---
            st.subheader("MTTR & MTBF per Production Run")
            if run_summary_df is not None and not run_summary_df.empty and run_summary_df['STOPS'].sum() > 0:
                rr_utils.plot_mttr_mtbf_chart(
                    df=run_summary_df, x_col='RUN ID', mttr_col='MTTR (min)',
                    mtbf_col='MTBF (min)', shots_col='Total Shots',
                    title="MTTR, MTBF & Shot Count per Run"
                )
                with st.expander("View MTTR/MTBF Data Table & Correlation Analysis", expanded=False): 
                    st.dataframe(rr_utils.get_renamed_summary_df(run_summary_df))
                    if detailed_view:
                        st.markdown(rr_utils.generate_mttr_mtbf_analysis(analysis_df, analysis_level), unsafe_allow_html=True)

        # --- Option B: Hourly View (Only for Daily) ---
        elif analysis_view_mode == "Hour":
            
            hourly_summary_df = results.get('hourly_summary', pd.DataFrame())
            
            # --- 4a. Hourly Breakdown Table ---
            with st.expander("View Hourly Breakdown Table", expanded=True):
                if not hourly_summary_df.empty:
                    st.dataframe(rr_utils.get_renamed_summary_df(hourly_summary_df), use_container_width=True)
                else:
                    st.info("No hourly data available.")

            c1,c2 = st.columns(2)
            with c1:
                # Hourly Bucket Analysis = Completed runs that ENDED in that hour
                st.subheader("Hourly Bucket Trend")
                if not complete_runs.empty:
                    complete_runs['hour'] = complete_runs['run_end_time'].dt.hour
                    pivot_df = pd.crosstab(index=complete_runs['hour'], columns=complete_runs['time_bucket'].astype('category').cat.set_categories(results["bucket_labels"]))
                    pivot_df = pivot_df.reindex(pd.Index(range(24), name='hour'), fill_value=0)
                    fig_hourly_bucket = px.bar(pivot_df, x=pivot_df.index, y=pivot_df.columns, 
                                            title='Hourly Distribution of Run Durations', barmode='stack', 
                                            color_discrete_map=results["bucket_color_map"], 
                                            labels={'hour': 'Hour', 'value': 'Number of Buckets', 'variable': 'Run Duration (min)'})
                    st.plotly_chart(fig_hourly_bucket, use_container_width=True)
                    with st.expander("View Bucket Trend Data", expanded=False): st.dataframe(pivot_df)
                else:
                    st.info("No completed runs to chart by hour.")
            
            with c2:
                st.subheader("Hourly Stability Trend")
                if not hourly_summary_df.empty:
                    rr_utils.plot_trend_chart(hourly_summary_df, 'hour', 'stability_index', "Hourly Stability Trend", "Hour of Day", "Stability (%)", is_stability=True)
                    with st.expander("View Stability Data", expanded=False): 
                        st.dataframe(rr_utils.get_renamed_summary_df(hourly_summary_df))
                else:
                    st.info("No hourly stability data.")

            st.subheader("Hourly MTTR & MTBF Trend")
            if not hourly_summary_df.empty and hourly_summary_df['stops'].sum() > 0:
                rr_utils.plot_mttr_mtbf_chart(
                    df=hourly_summary_df, x_col='hour', mttr_col='mttr_min',
                    mtbf_col='mtbf_min', shots_col='total_shots',
                    title="Hourly MTTR & MTBF Trend"
                )
                with st.expander("View MTTR/MTBF Data", expanded=False): 
                    st.dataframe(rr_utils.get_renamed_summary_df(hourly_summary_df))
                if detailed_view:
                    with st.expander("🤖 View MTTR/MTBF Correlation Analysis", expanded=False):
                         # Use the original function logic manually for hourly since helper expects specific cols
                         st.info("Automated correlation analysis is best viewed in 'Run' mode.")
            else:
                 st.info("No hourly stop data for MTTR/MTBF charts.")

# ==============================================================================
# --- 8. MAIN APP ENTRY POINT ---
# ==============================================================================

def run_run_rate_ui():
    
    st.sidebar.title("File Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more Run Rate Excel files", 
        type=["xlsx", "xls"], 
        accept_multiple_files=True,
        key="rr_file_uploader"
    )

    if not uploaded_files:
        st.info("👈 Upload one or more Excel files to see the Run Rate reports.")
        st.stop()

    # Use the centralized load function
    df_all_tools = rr_utils.load_all_data(uploaded_files)

    id_col = "tool_id"
    if id_col not in df_all_tools.columns:
        st.error(f"None of the uploaded files contain a 'TOOLING ID' or 'EQUIPMENT CODE' column.")
        st.stop()

    df_all_tools.dropna(subset=[id_col], inplace=True)
    df_all_tools[id_col] = df_all_tools[id_col].astype(str)

    # --- Sidebar Tool ID Selection ---
    tool_ids = ["All Tools (Risk Tower)"] + sorted(df_all_tools[id_col].unique().tolist())
    dashboard_tool_id_selection = st.sidebar.selectbox(
        "Select Tool ID for Dashboard Analysis", 
        tool_ids,
        key="rr_tool_select"
    )

    # --- GLOBAL ANALYSIS PARAMETERS (For Dashboard & Trends) ---
    st.sidebar.title("Analysis Parameters ⚙️")
    with st.sidebar.expander("Configure Metrics", expanded=True):
        tolerance = st.slider("Tolerance Band (% of Mode CT)", 0.01, 0.50, 0.25, 0.01, key="rr_tolerance", help="Defines the ±% around Mode CT.")
        downtime_gap_tolerance = st.slider("Downtime Gap Tolerance (sec)", 0.0, 5.0, 2.0, 0.5, key="rr_downtime_gap", help="Defines the minimum idle time between shots to be considered a stop.")
        run_interval_hours = st.slider("Run Interval Threshold (hours)", 1, 24, 8, 1, key="rr_run_interval", help="Defines the max hours between shots before a new Production Run is identified.")

    if dashboard_tool_id_selection == "All Tools (Risk Tower)":
        if len(tool_ids) > 1:
            first_tool = tool_ids[1]
            df_for_dashboard = df_all_tools[df_all_tools[id_col] == first_tool]
            tool_id_for_dashboard_display = first_tool
        else:
            df_for_dashboard = pd.DataFrame()
            tool_id_for_dashboard_display = "No Tool Selected"
    else:
        df_for_dashboard = df_all_tools[df_all_tools[id_col] == dashboard_tool_id_selection]
        tool_id_for_dashboard_display = dashboard_tool_id_selection

    # --- Tab Rendering ---
    tab1, tab2, tab3 = st.tabs(["Risk Tower", "Run Rate Dashboard", "Trends"])

    with tab1:
        render_risk_tower(df_all_tools)

    with tab2:
        if not df_for_dashboard.empty:
            render_dashboard(df_for_dashboard, tool_id_for_dashboard_display, tolerance, downtime_gap_tolerance, run_interval_hours)
        else:
            st.info("Select a specific Tool ID from the sidebar to view its dashboard.")
    
    with tab3:
        if not df_for_dashboard.empty:
            render_trends_tab(df_for_dashboard, tolerance, downtime_gap_tolerance, run_interval_hours)
        else:
            st.info("Select a specific Tool ID from the sidebar to view trends.")

if __name__ == "__main__":
    run_run_rate_ui()