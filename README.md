# FloatChat AI 🌊

An intelligent oceanographic data analysis platform powered by AI, combining RAG (Retrieval-Augmented Generation) with natural language querying for ARGO float data.

## 🚀 Features

- **AI-Powered Chat Interface**: Ask questions about ocean data in natural language
- **Dual Query System**: 
  - Semantic search using ChromaDB for descriptive queries
  - NL-to-SQL translation for analytical queries
- **Interactive Dashboard**: Visualize ocean temperature, salinity, and depth data
- **Multiple LLM Support**: Works with both Ollama (local) and HuggingFace models
- **Real-time Data Processing**: Process and analyze ARGO float measurements
- **Export Capabilities**: Export data in CSV, NetCDF, and ASCII formats

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.13+
- **Frontend**: Streamlit
- **Database**: PostgreSQL
- **Vector Store**: ChromaDB
- **LLM**: Ollama (gemma2:2b) / HuggingFace
- **Embeddings**: nomic-embed-text / sentence-transformers

## 📋 Prerequisites

- Python 3.13+
- PostgreSQL
- Ollama (for local LLM)
- Git

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/NematSachdeva/FloatChat-AI_107.git
cd FloatChat-AI_107/floatchat-ai
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

### 4. Setup PostgreSQL Database

```bash
# Create database
createdb argo

# Or using psql
psql -U postgres
CREATE DATABASE argo;
\q
```

### 5. Install and Setup Ollama

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull gemma2:2b
ollama pull nomic-embed-text:latest
```

### 6. Configure Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# Database Configuration
DB_PASSWORD=
DATABASE_URL=postgresql+psycopg://your_username@localhost:5432/argo

# LLM Configuration
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=gemma2:2b
EMBEDDING_MODEL=nomic-embed-text:latest

# ChromaDB Configuration
CHROMA_PATH=./chroma_db
VECTOR_STORE=persistent

# Backend URL
BACKEND_URL=http://127.0.0.1:8000
```

## 🚀 Running the Application

### Start Ollama Service

```bash
ollama serve
```

### Start Backend (FastAPI)

```bash
cd floatchat-ai
source venv/bin/activate
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Start Frontend (Streamlit)

In a new terminal:

```bash
cd floatchat-ai
source venv/bin/activate
streamlit run streamlit_app.py
```

### Access the Application

- **Frontend**: http://localhost:8501
- **Backend API**: http://127.0.0.1:8000
- **API Docs**: http://127.0.0.1:8000/docs

## 📊 Usage

### Chat Interface

1. Open the frontend at http://localhost:8501
2. Type your question in the chat interface
3. Examples:
   - "What is ARGO?"
   - "Show me temperature data by depth"
   - "What are the average salinity measurements?"

### API Endpoints

#### Health Check
```bash
curl http://127.0.0.1:8000/health
```

#### Query Endpoint
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query_text":"What is ARGO?"}'
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Run specific tests:

```bash
pytest tests/test_api_client.py
pytest tests/test_chat_interface.py
```

## 📁 Project Structure

```
floatchat-ai/
├── main.py                 # FastAPI backend
├── streamlit_app.py        # Streamlit frontend
├── config.py               # Configuration management
├── components/             # UI components
│   ├── api_client.py
│   ├── chat_interface.py
│   ├── data_manager.py
│   └── ...
├── tests/                  # Test suite
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
└── README.md              # This file
```

## 🔐 Security

- Never commit `.env` files with sensitive data
- Use environment variables for all credentials
- The `.gitignore` file excludes sensitive files automatically

## 🐛 Troubleshooting

### Backend Returns 500 Error

- Ensure Ollama is running: `ollama serve`
- Check `.env` file has correct `LLM_PROVIDER=ollama`
- Verify models are installed: `ollama list`

### ChromaDB Collection Error

- Delete and recreate: `rm -rf chroma_db/`
- Restart backend to reinitialize

### Database Connection Error

- Verify PostgreSQL is running
- Check database exists: `psql -l`
- Verify credentials in `.env`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👥 Authors

- **Nemat Sachdeva** - [GitHub](https://github.com/NematSachdeva)

## 🙏 Acknowledgments

- ARGO float data program
- Ollama for local LLM support
- ChromaDB for vector storage
- FastAPI and Streamlit communities

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

Made with ❤️ for oceanographic research
