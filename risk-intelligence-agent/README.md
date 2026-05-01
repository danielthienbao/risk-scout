# Scalable Risk Intelligence Agent

A local, Dockerized project that combines structured risk scoring with Retrieval-Augmented Generation (RAG) to explain financial and supply-chain risk.

## Architecture (text diagram)

```text
Raw CSV + Risk Documents
        |
        v
Spark Ingestion (cleaned CSVs)
        |
        v
Spark Risk Scoring (risk_score + risk_level)
        |
        +--> ML Model Training (RandomForest, models/risk_model.pkl)
        |
        +--> RAG Index Build (Chroma vectorstore)
                    |
                    v
         Risk Explainer (Ollama or fallback template)
                    |
                    v
            Streamlit Dashboard
```

## Tech Stack

- Python 3.11
- PySpark for ingestion and risk scoring
- pandas + scikit-learn for ML training
- LangChain + ChromaDB + sentence-transformers for local RAG
- Ollama (optional) for richer natural-language explanations
- Streamlit for dashboard UI
- Docker + Docker Compose for local deployment

## Run Locally

From project root `risk-intelligence-agent/`:

```bash
pip install -r requirements.txt
python spark/ingest_data.py
python spark/compute_risk_scores.py
python ml/train_risk_model.py
python rag/build_vector_index.py
streamlit run app/streamlit_app.py
```

## Run With Docker

```bash
docker compose up --build
```

Then open: [http://localhost:8501](http://localhost:8501)

## Example Questions

- Why is entity `E0421` marked as High risk today?
- Which risk features are driving route-level risk increases?
- What external disruption signals support this risk classification?

## Resume Bullet

Built a Dockerized Risk Intelligence Agent that uses Spark-based feature pipelines, RandomForest risk prediction, and local RAG (Chroma + sentence-transformers + optional Ollama) to generate explainable supply-chain risk insights in a Streamlit dashboard.

## Future Improvements

- Add time-series forecasting for proactive risk alerts
- Track model and feature drift with scheduled retraining
- Expand document corpus with news and policy advisories
- Add role-based authentication and audit logging
- Introduce batch APIs for enterprise workflow integration
