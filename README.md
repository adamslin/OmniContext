# OmniContext: An RAG-Powered Q&A System

A Retrieval-Augmented Generation (RAG) system built with Google Gemini, LangChain, and ChromaDB. Features multi-turn conversation support, streaming responses, and a modern web interface.

## Features

- **PDF Document Processing**: Upload and vectorize PDF documents for knowledge base construction
- **Multi-turn Conversations**: Context-aware dialogue with conversation history
- **Query Rewriting**: Automatically reformulates follow-up questions for better retrieval
- **Streaming Responses**: Real-time AI response streaming for better UX
- **Source Attribution**: View which document chunks were used to generate answers
- **Web Interface**: ChatGPT-style UI built with Streamlit
- **RESTful API**: FastAPI backend with comprehensive endpoints

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Google Gemini (gemini-3-flash-preview) |
| Embeddings | Google text-embedding-004 |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| PDF Processing | PyPDF |

## Project Structure

```
IntelligentBot/
├── core/                    # Core logic module
├── data/                    # PDF documents directory
├── db/                      # ChromaDB vector store
├── .env                     # API key configuration
├── .gitignore
├── requirements.txt         # Python dependencies
│
├── app_test.py              # API connection test
├── ingest.py                # PDF vectorization script
├── rag_chain.py             # Basic RAG Q&A
├── rag_with_history.py      # Multi-turn conversation RAG
│
├── main.py                  # FastAPI backend
├── ui.py                    # Streamlit frontend
└── run.sh                   # One-click startup script
```

## Prerequisites

- Python 3.10+
- Google API Key with Gemini API access

## Installation

### 1. Clone the Repository

```bash
cd IntelligentBot
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Edit the `.env` file and add your Google API Key:

```env
GOOGLE_API_KEY=your_actual_api_key_here
```

Get your API key at: https://aistudio.google.com/app/apikey

## Usage

### Quick Start (Web Application)

**Option 1: One-click startup**

```bash
./run.sh
```

**Option 2: Manual startup**

Terminal 1 - Start backend:
```bash
python main.py
```

Terminal 2 - Start frontend:
```bash
streamlit run ui.py
```

Access the application:
- **Frontend UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

### Command Line Tools

#### Test API Connection

```bash
python app_test.py
```

#### Vectorize PDF Documents

Place PDF files in the `data/` directory, then run:

```bash
python ingest.py
```

#### Interactive Q&A (Terminal)

Basic RAG:
```bash
python rag_chain.py
```

With conversation history:
```bash
python rag_with_history.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/upload` | POST | Upload PDF and update vector store |
| `/chat/stream` | POST | Streaming chat response |
| `/chat` | POST | Non-streaming chat response |
| `/session/create` | POST | Create new session |
| `/session/{id}/clear` | POST | Clear session history |
| `/documents` | GET | List uploaded documents |
| `/chat/{id}/sources` | GET | Get reference sources |

## Configuration

### Text Splitting Parameters

In `ingest.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `chunk_size` | 1000 | Maximum characters per chunk |
| `chunk_overlap` | 200 | Overlap between adjacent chunks |

### Retrieval Parameters

In `rag_chain.py` and `rag_with_history.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `k` | 3 | Number of documents to retrieve |
| `search_type` | similarity | Vector similarity search |

### Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model` | gemini-3-flash-preview | Gemini model for generation |
| `embedding` | text-embedding-004 | Embedding model |
| `temperature` | 0.3 | Lower for more accurate responses |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                          │
│                    (Streamlit - ui.py)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP
┌─────────────────────────▼───────────────────────────────────┐
│                      FastAPI Backend                         │
│                       (main.py)                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    RAG Pipeline                              │
│              (rag_with_history.py)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Query     │  │  History-   │  │   Stuff Documents   │  │
│  │  Rewriter   │→ │   Aware     │→ │       Chain         │  │
│  │             │  │  Retriever  │  │                     │  │
│  └─────────────┘  └──────┬──────┘  └──────────┬──────────┘  │
│                          │                     │             │
│                   ┌──────▼──────┐       ┌──────▼──────┐     │
│                   │  ChromaDB   │       │   Gemini    │     │
│                   │  (db/)      │       │    LLM      │     │
│                   └─────────────┘       └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### API Quota Exceeded (429 Error)

```
429 You exceeded your current quota
```

**Solutions:**
- Wait for quota reset (free tier has per-minute/daily limits)
- Enable billing in Google Cloud Console
- Request quota increase

### Model Not Found (404 Error)

```
404 models/gemini-xxx is not found
```

**Solution:** Update the model name in the Python files to a valid model.

### Vector Store Not Found

```
Vector database does not exist
```

**Solution:** Run `python ingest.py` after placing PDF files in `data/`.

### Backend Connection Failed

```
Backend service not running
```

**Solution:** Ensure `python main.py` is running before accessing the frontend.

## License

MIT License

## Acknowledgments

- [LangChain](https://langchain.com/) - RAG framework
- [Google Gemini](https://ai.google.dev/) - LLM and embeddings
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework
- [Streamlit](https://streamlit.io/) - Frontend framework
