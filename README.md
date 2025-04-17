# Document RAG System

A production-grade Retrieval Augmented Generation (RAG) system built with Streamlit, LangChain, and Google's Gemini API. The system supports various document types (PDF, TXT, and images) and uses FAISS for efficient vector search.

## Features

- **Multi-document support**: Process PDFs, text files, and images
- **Intelligent document processing**: Different handling based on document type
- **Efficient retrieval**: FAISS vector store with Google's embeddings
- **Conversational interface**: Chat with your documents using Gemini API
- **Source attribution**: See which parts of documents were used to answer your questions
- **Production-ready**: Error handling, logging, and modular architecture

## System Architecture

The RAG system consists of several components:

1. **DocumentProcessor**: Handles different document types and splits them into chunks
2. **VectorStoreManager**: Manages embeddings and vector retrieval using FAISS
3. **RAGSystem**: Coordinates document processing, vector stores, and LLM interactions
4. **StreamlitApp**: Provides the user interface

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

## Usage

1. Enter your Google API key in the sidebar
2. Upload documents (PDF, TXT) or images
3. Click "Process Documents" to analyze and index the content
4. Ask questions about your documents in the chat interface

## Requirements

- Python 3.8+
- Google API key with access to Gemini Pro and Gemini Pro Vision
- Dependencies listed in requirements.txt

## Configuration

The system can be configured through the application's UI:
- Set your Google API key
- Upload and process documents
- Ask questions about your documents

## Security Considerations

- API keys are not stored persistently but kept in session state
- Temporary files are used for document processing
- Safety settings are enabled for the Gemini model

## Deployment

To deploy to production:

1. Set up environment variables for API keys
2. Deploy to Streamlit Cloud or your preferred hosting platform
3. Configure logging for production use

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.