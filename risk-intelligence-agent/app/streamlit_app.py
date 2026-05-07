"""Streamlit dashboard for the Scalable Risk Intelligence Agent."""

from pathlib import Path

import pandas as pd
import streamlit as st

from rag.risk_explainer import explain_risk


@st.cache_data
def load_risk_scores() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parents[1]
    scores_dir = base_dir / 'data' / 'processed' / 'risk_scores.csv'
    if scores_dir.is_file():
        df = pd.read_csv(scores_dir)
        return df
    csv_files = sorted(scores_dir.glob('part-*.csv'))
    if not csv_files:
        raise FileNotFoundError('No risk score file found. Run Spark scoring first.')
    return pd.read_csv(csv_files[0])


def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df


def _safe_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors='coerce')
    return float(s.mean()) if not s.dropna().empty else float('nan')


def _zscore(value: float, series: pd.Series) -> float:
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        return 0.0
    std = float(s.std(ddof=0))
    if std == 0.0:
        return 0.0
    return float((value - float(s.mean())) / std)


def _top_drivers(df: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    # Simple, explainable proxy: z-score deviation from population with known directionality.
    # Positive "impact" means higher risk pressure; negative means risk-reducing.
    drivers = [
        ('weather_risk', 'Higher increases risk', +1.0),
        ('delay_rate', 'Higher increases risk', +1.0),
        ('supplier_reliability', 'Higher reduces risk', -1.0),
    ]

    rows = []
    for feature, interpretation, direction in drivers:
        if feature not in df.columns:
            continue
        value = float(selected[feature])
        z = _zscore(value, df[feature])
        impact = direction * z
        arrow = '↑' if impact > 0 else ('↓' if impact < 0 else '→')
        rows.append(
            {
                'feature': feature,
                'value': round(value, 2),
                'direction': arrow,
                'impact_score': round(impact, 2),
                'interpretation': interpretation,
            }
        )

    drivers_df = pd.DataFrame(rows)
    if drivers_df.empty:
        return drivers_df
    return drivers_df.reindex(drivers_df['impact_score'].abs().sort_values(ascending=False).index)


def _histogram(series: pd.Series, bins: int = 18) -> pd.DataFrame:
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        return pd.DataFrame({'count': []})
    cats = pd.cut(s, bins=bins)
    counts = cats.value_counts().sort_index()
    out = counts.to_frame('count')
    out.index = out.index.astype(str)
    return out


def main():
    st.set_page_config(page_title='Scalable Risk Intelligence Agent', layout='wide')
    st.title('Scalable Risk Intelligence Agent')

    try:
        df = _prep_df(load_risk_scores())
    except Exception as exc:
        st.error(f'Unable to load risk scores: {exc}')
        st.stop()

    entity_id = st.selectbox('Select an entity ID', sorted(df['entity_id'].unique()))
    selected = df[df['entity_id'] == entity_id].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric('Risk Score', f"{selected['risk_score']:.2f}")
    col2.metric('Risk Level', selected['risk_level'])
    col3.metric('Route', selected['route'])

    t1, t2, t3, t4 = st.tabs(['Entity details', 'Drivers', 'Trends & breakdowns', 'RAG explanation'])

    with t1:
        st.subheader('Structured Risk Features')
        st.json(
            {
                'delay_rate': float(selected['delay_rate']),
                'weather_risk': float(selected['weather_risk']),
                'supplier_reliability': float(selected['supplier_reliability']),
                'timestamp': str(selected.get('timestamp', '')),
                'region': selected.get('region', 'unknown'),
                'origin': selected['origin'],
                'destination': selected['destination'],
            }
        )

        st.subheader('What changed? (vs baselines)')
        route_avg = _safe_mean(df[df['route'] == selected['route']]['risk_score'])
        region_avg = _safe_mean(df[df.get('region', 'unknown') == selected.get('region', 'unknown')]['risk_score'])
        overall_avg = _safe_mean(df['risk_score'])
        score = float(selected['risk_score'])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('vs Route avg', f'{(score - route_avg):+.2f}', help=f"Route mean risk_score: {route_avg:.2f}")
        c2.metric('vs Region avg', f'{(score - region_avg):+.2f}', help=f"Region mean risk_score: {region_avg:.2f}")
        c3.metric('vs Overall avg', f'{(score - overall_avg):+.2f}', help=f"Overall mean risk_score: {overall_avg:.2f}")
        percentile = float((df['risk_score'] <= score).mean() * 100.0)
        c4.metric('Percentile', f'{percentile:.0f}th', help='Percentile among all entities (higher is riskier).')

    with t2:
        st.subheader('Top drivers (proxy feature contribution)')
        drivers_df = _top_drivers(df, selected)
        if drivers_df.empty:
            st.info('No driver features available.')
        else:
            st.caption('Impact score uses z-score deviation with known directionality (fast, explainable proxy).')
            st.dataframe(drivers_df, use_container_width=True, hide_index=True)

    with t3:
        st.subheader('Risk score distribution')
        hist_df = _histogram(df['risk_score'])
        if hist_df.empty:
            st.info('No risk_score data available for distribution.')
        else:
            st.bar_chart(hist_df, height=220)

        st.subheader('Risk over time (portfolio)')
        if 'timestamp' in df.columns and df['timestamp'].notna().any():
            ts = (
                df.dropna(subset=['timestamp'])
                .set_index('timestamp')
                .sort_index()
                .resample('6H')['risk_score']
                .mean()
                .to_frame('avg_risk_score')
            )
            if not ts.empty:
                st.line_chart(ts, height=220)
        else:
            st.info('No timestamp data available for time trend.')

        st.subheader('High-risk breakdowns')
        high = df[df['risk_level'] == 'High']
        b1, b2, b3 = st.columns(3)

        with b1:
            st.caption('High-risk count by region')
            if 'region' in high.columns and not high.empty:
                region_counts = high['region'].fillna('unknown').value_counts().to_frame('count')
                st.bar_chart(region_counts, height=220)
            else:
                st.info('No region breakdown available.')

        with b2:
            st.caption('High-risk count by route (top 10)')
            if 'route' in high.columns and not high.empty:
                route_counts = high['route'].fillna('unknown').value_counts().head(10).to_frame('count')
                st.bar_chart(route_counts, height=220)
            else:
                st.info('No route breakdown available.')

        with b3:
            st.caption('High-risk by origin→destination (top 10)')
            if {'origin', 'destination'}.issubset(high.columns) and not high.empty:
                od = (high['origin'].fillna('unknown') + ' → ' + high['destination'].fillna('unknown'))
                od_counts = od.value_counts().head(10).to_frame('count')
                st.bar_chart(od_counts, height=220)
            else:
                st.info('No origin/destination breakdown available.')

    with t4:
        try:
            result = explain_risk(entity_id)
            st.subheader('Retrieved Supporting Context')
            for idx, ctx in enumerate(result['retrieved_context'], start=1):
                st.markdown(f"**Context {idx}:** {ctx}")

            st.subheader('Final AI Explanation')
            st.write(result['explanation'])
        except Exception as exc:
            st.warning(f'Could not generate RAG explanation: {exc}')

    st.subheader('All High-Risk Entities')
    high_risk_df = df[df['risk_level'] == 'High'].sort_values(by='risk_score', ascending=False)
    st.dataframe(high_risk_df, use_container_width=True)


if __name__ == '__main__':
    main()
