import streamlit as st
import pandas as pd
import numpy as np
import warnings
import streamlit.components.v1 as components
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import importlib  # <--- ADDED THIS MISSING IMPORT

# Import all logic functions from the utils file
import run_rate_utils as rr_utils
importlib.reload(rr_utils) # Forces update of utils on every run

# ==============================================================================
# --- 1. PAGE CONFIG & SETUP ---
# ==============================================================================

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(layout="wide", page_title="Run Rate Analysis Dashboard")

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


def render_dashboard(df_tool, tool_id_selection):
    """Renders the main Run Rate Dashboard tab."""
    st.sidebar.title("Dashboard Controls ⚙️")

    with st.sidebar.expander("ℹ️ About This Dashboard", expanded=False):
        st.markdown("""
        ### Run Rate Analysis
        - **Efficiency (%)**: Normal Shots ÷ Total Shots
        - **MTTR (min)**: Average downtime per stop.
        - **MTBF (min)**: Average uptime between stops.
        - **Stability Index (%)**: Uptime ÷ (Uptime + Downtime)
        - **Bucket Analysis**: Groups run durations into 20-min intervals.
        ---
        ### Analysis Levels
        - **Daily (by Run)**: Detailed analysis for a single day, identifying runs based on the 'Run Interval Threshold'.
        - **Weekly / Monthly (by Run)**: Aggregated data, showing trends over time.
        - **Custom Period (by Run)**: Analysis for a user-defined date range.
        ---
        ### Sliders
        - **Tolerance Band**: Defines the acceptable CT range around the Mode CT.
        - **Run Interval Threshold**: Defines the max hours between shots before a new Production Run is identified.
        - **Remove Runs...**: Filters out runs with fewer shots than the selected value.
        """)
        

    analysis_level = st.sidebar.radio(
        "Select Analysis Level",
        options=["Daily (by Run)", "Weekly (by Run)", "Monthly (by Run)", "Custom Period (by Run)"],
        key="rr_analysis_level"
    )

    st.sidebar.markdown("---")
    tolerance = st.sidebar.slider("Tolerance Band (% of Mode CT)", 0.01, 0.50, 0.25, 0.01, key="rr_tolerance", help="Defines the ±% around Mode CT.")
    downtime_gap_tolerance = st.sidebar.slider("Downtime Gap Tolerance (sec)", 0.0, 5.0, 2.0, 0.5, key="rr_downtime_gap", help="Defines the minimum idle time between shots to be considered a stop.")
    run_interval_hours = st.sidebar.slider("Run Interval Threshold (hours)", 1, 24, 8, 1, key="rr_run_interval", help="Defines the max hours between shots before a new Production Run is identified.")
    
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
    
    min_shots_filter = 1 
    # All modes are now effectively "by Run", so we always show the filter option
    if 'by Run' in analysis_level:
        st.sidebar.markdown("---")
        
        # Add the toggle
        enable_run_filter = st.sidebar.toggle("Filter Small Production Runs", value=False, key="rr_enable_run_filter", help="Turn this on to show the slider for filtering out runs with few shots.")
        
        if enable_run_filter and not df_processed.empty:
            run_shot_counts = df_processed.groupby('run_id').size()
            if not run_shot_counts.empty:
                max_shots = int(run_shot_counts.max()) if not run_shot_counts.empty else 1
                default_value = min(10, max_shots) if max_shots > 1 else 1
                min_shots_filter = st.sidebar.slider(
                    "Remove Runs with Fewer Than X Shots",
                    min_value=1,
                    max_value=max_shots,
                    value=default_value,
                    step=1,
                    key="rr_min_shots_filter",
                    help="Filters out smaller production runs to focus on more significant ones."
                )
        elif not enable_run_filter:
            min_shots_filter = 1 # Explicitly set to 1 if filter is off
    
    st.sidebar.markdown("---")
    detailed_view = st.sidebar.toggle("Show Detailed Analysis", value=True, key="rr_detailed_view")


    if df_processed.empty:
        st.error(f"Could not process data for {tool_id_selection}. Check file format or data range."); st.stop()

    st.title(f"Run Rate Dashboard: {tool_id_selection}")

    # Mode is always 'by_run' now
    mode = 'by_run'
    df_view = pd.DataFrame()

    # --- Date/Period Selection ---
    if "Daily" in analysis_level: # Matches "Daily (by Run)"
        st.header(f"Daily Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
        available_dates = sorted(df_processed["date"].unique())
        if not available_dates:
            st.warning("No data available for any date."); st.stop()
        selected_date = st.selectbox("Select Date", options=available_dates, index=len(available_dates)-1, format_func=lambda d: pd.to_datetime(d).strftime('%d %b %Y'), key="rr_daily_select")
        df_view = df_processed[df_processed["date"] == selected_date]
        sub_header = f"Summary for {pd.to_datetime(selected_date).strftime('%d %b %Y')}"

    elif "Weekly" in analysis_level:
        st.header(f"Weekly Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
        available_weeks = sorted(df_processed["week"].unique())
        if not available_weeks:
            st.warning("No data available for any week."); st.stop()
        year = df_processed['shot_time'].iloc[0].year
        selected_week = st.selectbox(f"Select Week (Year {year})", options=available_weeks, index=len(available_weeks)-1, key="rr_weekly_select")
        df_view = df_processed[df_processed["week"] == selected_week]
        sub_header = f"Summary for Week {selected_week}"

    elif "Monthly" in analysis_level:
        st.header(f"Monthly Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
        available_months = sorted(df_processed["month"].unique())
        if not available_months:
            st.warning("No data available for any month."); st.stop()
        selected_month = st.selectbox(f"Select Month", options=available_months, index=len(available_months)-1, format_func=lambda p: p.strftime('%B %Y'), key="rr_monthly_select")
        df_view = df_processed[df_processed["month"] == selected_month]
        sub_header = f"Summary for {selected_month.strftime('%B %Y')}"

    elif "Custom Period" in analysis_level:
        st.header(f"Custom Period Analysis {'(by Production Run)' if mode == 'by_run' else ''}")
        min_date = df_processed['date'].min()
        max_date = df_processed['date'].max()
        start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date, key="rr_custom_start")
        end_date = st.date_input("End date", max_date, min_value=start_date, max_value=max_date, key="rr_custom_end")
        if start_date and end_date:
            mask = (df_processed['date'] >= start_date) & (df_processed['date'] <= end_date)
            df_view = df_processed[mask]
            sub_header = f"Summary for {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"

    # --- Run Filtering and Labeling ---
    if not df_view.empty:
        df_view = df_view.copy()
        # Run identification is always relevant now
        if 'run_id' in df_view.columns:
            # Create a consistent integer-based index for runs within the current view
            df_view['run_id_local'] = df_view.groupby('run_id').ngroup()
            unique_run_ids = df_view.sort_values('shot_time')['run_id_local'].unique()
            run_label_map = {run_id: f"Run {i+1:03d}" for i, run_id in enumerate(unique_run_ids)}
            df_view['run_label'] = df_view['run_id_local'].map(run_label_map)

    if 'by Run' in analysis_level and not df_view.empty:
        runs_before_filter = df_view['run_label'].nunique()
        run_shot_counts_in_view = df_view.groupby('run_label')['run_label'].transform('count')
        df_view = df_view[run_shot_counts_in_view >= min_shots_filter]
        runs_after_filter = df_view['run_label'].nunique()

        if runs_before_filter > 0:
            st.sidebar.metric(
                label="Runs Displayed",
                value=f"{runs_after_filter} / {runs_before_filter}",
                delta=f"-{runs_before_filter - runs_after_filter} filtered",
                delta_color="off"
            )

    if df_view.empty:
        st.warning(f"No data for the selected period (or all runs were filtered out).")
    else:
        # --- Main Calculation for Selected Period ---
        results = {}
        summary_metrics = {}

        # All remaining options are "by Run"
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

        # --- 1. KPI Metrics Display ---
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            total_d = summary_metrics.get('total_runtime_sec', 0); prod_t = summary_metrics.get('production_time_sec', 0); down_t = summary_metrics.get('downtime_sec', 0)
            prod_p = (prod_t / total_d * 100) if total_d > 0 else 0
            down_p = (down_t / total_d * 100) if total_d > 0 else 0
            
            mttr_val_min = summary_metrics.get('mttr_min', 0)
            mtbf_val_min = summary_metrics.get('mtbf_min', 0)
            
            mttr_display = rr_utils.format_minutes_to_dhm(mttr_val_min)
            mtbf_display = rr_utils.format_minutes_to_dhm(mtbf_val_min)

            with col1: st.metric("Run Rate MTTR", f"{mttr_val_min:.1f} min")
            with col2: st.metric("Run Rate MTBF", f"{mtbf_val_min:.1f} min")

            with col3: st.metric("Total Run Duration", rr_utils.format_duration(total_d)) 
            with col4:
                st.metric("Production Time", f"{rr_utils.format_duration(prod_t)}")
                st.markdown(f'<span style="background-color: {rr_utils.PASTEL_COLORS["green"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{prod_p:.1f}%</span>', unsafe_allow_html=True)
            with col5:
                st.metric("Downtime", f"{rr_utils.format_duration(down_t)}")
                st.markdown(f'<span style="background-color: {rr_utils.PASTEL_COLORS["red"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{down_p:.1f}%</span>', unsafe_allow_html=True)
        
        with st.container(border=True):
            c1, c2 = st.columns(2)
            c1.plotly_chart(rr_utils.create_gauge(summary_metrics.get('efficiency', 0) * 100, "Run Rate Efficiency (%)"), use_container_width=True)
            steps = [{'range': [0, 50], 'color': rr_utils.PASTEL_COLORS['red']}, {'range': [50, 70], 'color': rr_utils.PASTEL_COLORS['orange']},{'range': [70, 100], 'color': rr_utils.PASTEL_COLORS['green']}]
            c2.plotly_chart(rr_utils.create_gauge(summary_metrics.get('stability_index', 0), "Run Rate Stability Index (%)", steps=steps), use_container_width=True)
        
        with st.container(border=True):
            c1,c2,c3 = st.columns(3)
            t_s = summary_metrics.get('total_shots', 0); n_s = summary_metrics.get('normal_shots', 0)
            s_s = t_s - n_s
            n_p = (n_s / t_s * 100) if t_s > 0 else 0
            s_p = (s_s / t_s * 100) if t_s > 0 else 0
            with c1: st.metric("Total Shots", f"{t_s:,}")
            with c2:
                st.metric("Normal Shots", f"{n_s:,}")
                st.markdown(f'<span style="background-color: {rr_utils.PASTEL_COLORS["green"]}; color: #0E1117; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: bold;">{n_p:.1f}% of Total</span>', unsafe_allow_html=True)
            with c3:
                st.metric("Stop Events", f"{summary_metrics.get('stop_events', 0)}")
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
        
        # --- 2. Detailed Analysis Expander ---
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
tab1, tab2 = st.tabs(["Risk Tower", "Run Rate Dashboard"])

with tab1:
    render_risk_tower(df_all_tools)

with tab2:
    if not df_for_dashboard.empty:
        render_dashboard(df_for_dashboard, tool_id_for_dashboard_display)
    else:
        st.info("Select a specific Tool ID from the sidebar to view its dashboard.")