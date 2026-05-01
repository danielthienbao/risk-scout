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
        return pd.read_csv(scores_dir)
    csv_files = sorted(scores_dir.glob('part-*.csv'))
    if not csv_files:
        raise FileNotFoundError('No risk score file found. Run Spark scoring first.')
    return pd.read_csv(csv_files[0])


def main():
    st.set_page_config(page_title='Scalable Risk Intelligence Agent', layout='wide')
    st.title('Scalable Risk Intelligence Agent')

    try:
        df = load_risk_scores()
    except Exception as exc:
        st.error(f'Unable to load risk scores: {exc}')
        st.stop()

    entity_id = st.selectbox('Select an entity ID', sorted(df['entity_id'].unique()))
    selected = df[df['entity_id'] == entity_id].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric('Risk Score', f"{selected['risk_score']:.2f}")
    col2.metric('Risk Level', selected['risk_level'])
    col3.metric('Route', selected['route'])

    st.subheader('Structured Risk Features')
    st.json(
        {
            'delay_rate': float(selected['delay_rate']),
            'weather_risk': float(selected['weather_risk']),
            'supplier_reliability': float(selected['supplier_reliability']),
            'region': selected.get('region', 'unknown'),
            'origin': selected['origin'],
            'destination': selected['destination'],
        }
    )

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
