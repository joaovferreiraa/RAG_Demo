# RAG Demo - Study Project

This is a study project developed to learn and explore **Retrieval-Augmented Generation (RAG)** techniques using LangChain, OpenAI, and FAISS.

**Live Demo:** https://ragdemo-artieu7yevdrr9hwuaqmfj.streamlit.app/

## About

This project demonstrates how to build a question-answering system that can read PDF documents and answer questions about their content using AI. It combines document retrieval with large language models to provide accurate, context-aware responses.

## Technologies Used

- **LangChain**: Framework for building LLM applications
- **OpenAI API**: GPT-4o-mini for answer generation and text-embedding-3-small for embeddings
- **FAISS**: Facebook AI Similarity Search for efficient vector storage and retrieval
- **Streamlit**: Modern web interface for the application
- **Python-dotenv**: Environment variable management
- **PyPDF**: PDF document processing

## Project Structure

```
RAG_DEMO/
├── app.py               # Streamlit web application (main interface)
├── main.py              # Alternative CLI version for quick testing
├── create_db.py         # Script to create the vector database from PDFs
├── app_config.py        # Centralized configuration
├── requirements.txt     # Python dependencies
├── data/                # PDF documents to be processed
├── db/                  # FAISS vector database storage
└── .env                 # OpenAI API key (not included in repo)
```

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone or download this repository

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

5. Place your PDF documents in the `data/` folder

6. Create the vector database:
```bash
python create_db.py
```

## Usage

### Streamlit Web Interface

Run the application:
```bash
streamlit run app.py
```

The web interface provides:
- Interactive question input
- Formatted answers with source citations
- Expandable source references
- Example questions

### CLI Version (Alternative)

For quick testing without the web interface:
```bash
python main.py
```

This runs a simpler command-line version that processes one question at a time.

## How It Works

1. **Document Processing**: PDFs are split into chunks and converted to vector embeddings
2. **Vector Storage**: Embeddings are stored in a FAISS database for fast similarity search
3. **Question Processing**: User questions are converted to embeddings
4. **Retrieval**: The system finds the most relevant document chunks using MMR
5. **Generation**: An LLM generates an answer based on the retrieved context

## Learning Goals

This project was created to understand:
- Retrieval-Augmented Generation (RAG) architecture
- Vector databases and semantic search
- LangChain framework for LLM applications
- Embedding models and similarity search
- Prompt engineering for accurate responses
- Building web interfaces with Streamlit

## Notes

This is an educational project for study purposes. The code prioritizes clarity and learning over production optimization.
