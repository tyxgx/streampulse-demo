# StreamPulse — RAG demo

Live demo of the [StreamPulse](https://github.com/tyxgx/streampulse) chatbot: ask about Spotify streaming by country, artist and label.

**How a question is answered**

1. Small-talk shortcut (greetings / "what can you do")
2. **SQL router** — counts, totals and top-N are computed exactly with DuckDB; the LLM never invents a number
3. Otherwise: embed the question (all-MiniLM-L6-v2, ONNX) → nearest chunks → **confidence gate** (refuses instead of hallucinating when nothing is close enough) → grounded LLM answer with sources
4. LLM chain: Gemini → Groq (both optional keys)

**Why it looks different from the production app:** the original runs Django + Postgres/pgvector on EC2. This version has to fit a free 512 MB host, so:

| | Production app | This demo |
|---|---|---|
| Vectors | pgvector, 215K chunks | int8 numpy matrix (83 MB, memory-mapped), exact L2 scan in blocks |
| Chunk text | Postgres | SQLite, read only for the top-k rows |
| Embedder | PyTorch sentence-transformers | ONNX (fastembed), same model |
| Retrieval | hybrid vector + full-text + cross-encoder rerank | vector only |
| Multi-turn history | yes | no (single question) |

Data: 216,036 yearly-grain chunks (artist / country / label) from a public Kaggle Spotify-charts dataset; per-song chunks were dropped to fit memory. SQL tables are trimmed Parquet.

## Run locally
```bash
pip install -r requirements.txt
python data_build/build_data.py <path-to>/hf_space/demo_data data   # or download the release assets into data/
GROQ_API_KEY=... uvicorn server:app --port 8000
```

## Deploy (Render free)
`render.yaml` builds the Dockerfile (downloads the data release at build time, bakes in the embedder). Set `GROQ_API_KEY` in the dashboard.
