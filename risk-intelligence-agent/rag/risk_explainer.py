"""Generate hybrid structured + RAG-based risk explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import ollama
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def _resolve_risk_scores_csv(base_dir: Path) -> Path:
    scores_dir = base_dir / 'data' / 'processed' / 'risk_scores.csv'
    if scores_dir.is_file():
        return scores_dir
    csv_files = sorted(scores_dir.glob('part-*.csv'))
    if not csv_files:
        raise FileNotFoundError('Run spark/compute_risk_scores.py first to generate risk scores.')
    return csv_files[0]


def _build_query(row: pd.Series) -> str:
    return (
        f"Route {row['route']} from {row['origin']} to {row['destination']} "
        f"in region {row.get('region', 'unknown')} with risk level {row['risk_level']} "
        f"and risk score {row['risk_score']:.2f}."
    )


def _get_retriever(base_dir: Path):
    vectorstore_dir = base_dir / 'vectorstore'
    if not vectorstore_dir.exists():
        raise FileNotFoundError('Vectorstore not found. Run rag/build_vector_index.py first.')

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = Chroma(
        persist_directory=str(vectorstore_dir),
        embedding_function=embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={'k': 2})


def _template_explanation(row: pd.Series, contexts: List[str]) -> str:
    return (
        f"Entity {row['entity_id']} is assessed as {row['risk_level']} risk with a score of "
        f"{row['risk_score']:.2f}. Delay rate ({row['delay_rate']:.2f}) and weather risk "
        f"({row['weather_risk']:.2f}) are key drivers, while supplier reliability "
        f"({row['supplier_reliability']:.2f}) offsets some pressure. "
        f"Retrieved context suggests route-specific disruption exposure."
    )


def _generate_ollama_explanation(prompt: str) -> str:
    response = ollama.chat(
        model='llama3.2',
        messages=[
            {'role': 'system', 'content': 'You explain supply-chain risk in concise business language.'},
            {'role': 'user', 'content': prompt},
        ],
    )
    return response['message']['content']


def explain_risk(entity_id: str) -> Dict:
    """Return a structured and natural-language explanation for a selected entity."""
    base_dir = Path(__file__).resolve().parents[1]
    scores_csv = _resolve_risk_scores_csv(base_dir)
    df = pd.read_csv(scores_csv)

    row_df = df[df['entity_id'] == entity_id]
    if row_df.empty:
        raise ValueError(f'Entity {entity_id} not found in risk scores dataset.')
    row = row_df.iloc[0]

    retriever = _get_retriever(base_dir)
    query = _build_query(row)
    docs = retriever.invoke(query)
    retrieved_context = [d.page_content for d in docs]

    structured_signals = (
        f"delay_rate={row['delay_rate']:.2f}, weather_risk={row['weather_risk']:.2f}, "
        f"supplier_reliability={row['supplier_reliability']:.2f}, route={row['route']}, "
        f"region={row.get('region', 'unknown')}"
    )

    prompt = (
        'Explain this risk profile for an operations manager.\n'
        f"Entity: {row['entity_id']}\n"
        f"Risk Score: {row['risk_score']:.2f}\n"
        f"Risk Level: {row['risk_level']}\n"
        f"Structured Signals: {structured_signals}\n"
        'Context snippets:\n- ' + '\n- '.join(retrieved_context)
    )

    try:
        explanation = _generate_ollama_explanation(prompt)
    except Exception:
        explanation = _template_explanation(row, retrieved_context)

    return {
        'entity_id': row['entity_id'],
        'risk_score': float(round(row['risk_score'], 2)),
        'risk_level': row['risk_level'],
        'structured_signals': structured_signals,
        'retrieved_context': retrieved_context,
        'explanation': explanation,
    }
