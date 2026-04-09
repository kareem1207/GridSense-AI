# GridSense AI — Project Context

## What this project is
GridSense AI is a predictive grid fault intelligence platform for Indian DISCOMs.
It ingests smart meter and DTM (Distribution Transformer Meter) data, detects
anomalies using ML, predicts transformer failures 48-72 hours in advance, and
uses a GenAI agent to generate plain-language work orders for field engineers.

## Architecture — 6 layers
1. Simulator   → Python MQTT publisher mimicking 100 transformers + 5000 meters
2. Ingestion   → MQTT → Kafka → TimescaleDB pipeline
3. ML          → Isolation Forest (point anomalies) + LSTM Autoencoder (temporal)
4. Prediction  → EWMA smoothing + Linear regression trend projection
5. GenAI       → LangChain agent + RAG over knowledge base → Gemma4/Claude diagnosis
6. Frontend    → FastAPI REST backend + Streamlit dashboard

## Tech stack (ALWAYS use these — no substitutions)
- Language: Python 3.11
- Package manager: uv (never pip install directly)
- Backend: FastAPI
- ML: scikit-learn (Isolation Forest) + TensorFlow/Keras (LSTM)
- Database: PostgreSQL + TimescaleDB (time-series), use SQLAlchemy ORM
- Message queue: Kafka (kafka-python), MQTT (paho-mqtt)
- GenAI: LangChain + ChromaDB for RAG, Anthropic API as LLM
- Dashboard: Streamlit
- Testing: pytest

## Key domain concepts
- DTM parameters: Va/Vb/Vc (voltage), Ia/Ib/Ic (current), oil_temp,
  power_factor, thd_pct, active_power_kw, reactive_power_kvar, tamper_flag
- Anomaly score: 0-1 float, >0.75 = warning, >0.90 = critical
- EWMA span: 12 (covers ~3 hours of 15-min readings)
- Prediction window: 72 hours forward
- Combined score: 0.6 * isolation_forest_score + 0.4 * lstm_score

## Coding conventions
- All async FastAPI endpoints
- Pydantic v2 models for all schemas
- Type hints on every function
- Docstrings on every class and public method
- Environment variables via python-dotenv, never hardcode credentials
- SQLAlchemy async sessions for all DB operations

## Run commands
- Start simulator: python -m gridsense.simulator.main
- Start API: uvicorn gridsense.api.main:app --reload --port 8000
- Start dashboard: streamlit run gridsense/dashboard/app.py
- Run tests: pytest tests/ -v
- Install package: uv add <package>

## File to check for data schemas
- gridsense/schemas/ for all Pydantic models
- gridsense/ml/utils/data_loader.py for dataset loading logic

## Do not
- Never use pip install — always uv add
- Never hardcode transformer IDs or meter IDs
- Never skip type hints
- Never create files outside the gridsense/ package unless it is a config file
- Clean up all temporary test files after use
