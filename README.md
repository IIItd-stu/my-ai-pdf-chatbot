# My AI — chat + PDF Q&A

Streamlit app: persistent chats (SQLite), upload PDFs and ask questions about them (ChromaDB retrieval, with page-cited passages).
Runs on local **Ollama** or any **OpenAI-compatible API** (OpenAI, Groq, OpenRouter…).

## Run locally
```bash
pip install -r requirements.txt
ollama pull llama3.2 && ollama pull nomic-embed-text
streamlit run app.py
```
## Run this 
https://share.streamlit.io/

## Deploy (pick one)
**Render (easiest, hosted API):** push this folder to GitHub → Render → New → Blueprint → select repo.
Set `OPENAI_API_KEY` and `APP_PASSWORD` when prompted. A 1 GB disk keeps chats and indexed PDFs across restarts.

**Streamlit Community Cloud (free):** push to GitHub → share.streamlit.io → New app → `app.py`.
Under *Advanced settings → Secrets* add:
```
LLM_PROVIDER = "openai"
OPENAI_API_KEY = "sk-..."
APP_PASSWORD = "something-long"
```
Note: its disk is ephemeral, so chats/indexed PDFs reset when the app restarts.

**Your own server / laptop (fully local, free):** `docker compose up -d`, then pull the two models (commands at the top of `docker-compose.yml`).

## Notes
- PDFs indexed with one embedding model aren't searchable after switching models; the app only lists PDFs matching the current one.
- Chat history is shared by everyone who can open the app — keep `APP_PASSWORD` set on public deployments.
- Your original scripts are in `legacy/`.

## What's new vs. the original
Provider switch (Ollama/OpenAI-compatible), optional password gate, delete chat, export chat to Markdown, starter prompts, retrieval slider, remove PDFs, batched embeddings, upload size limit, clean-up of failed indexing, Docker/Render/Compose configs.
