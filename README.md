# MultiWorkflow AI Application

![Python](https://img.shields.io/badge/python-v3.11.0+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-latest-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A sophisticated AI application that combines **Retrieval-Augmented Generation (RAG)** and **Conversational Flow** capabilities, built with FastAPI and LangChain. The application provides dual chat modes for different use cases: document-based Q&A and interactive LinkedIn post generation.

## 🚀 Features

### RAG Mode
- **PDF Document Processing**: Upload and process PDF documents with intelligent text extraction
- **Vector-based Retrieval**: Advanced semantic search using FAISS vector database
- **Context-aware Responses**: Generate accurate answers based on document content
- **Persistent Chat Memory**: Maintains conversation context across sessions

### Flow Mode
- **Interactive Data Collection**: Step-by-step user information gathering
- **LinkedIn Post Generation**: Automated professional post creation
- **Validation Framework**: Built-in input validation for email and technology fields
- **Conversational AI**: Natural language interaction flow

## 🛠️ Tech Stack

### Backend Framework
- **FastAPI**: Modern, fast web framework for building APIs
- **Python 3.11.0**: Latest stable Python version

### AI/ML Libraries
- **LangChain**: Framework for developing applications with language models
- **LangGraph**: State management for conversational AI workflows
- **ChatGroq**: High-performance language model integration
- **HuggingFace Transformers**: Embedding models for semantic search

### Vector Database & Search
- **FAISS**: Efficient similarity search and clustering
- **Sentence Transformers**: State-of-the-art text embeddings

### Frontend
- **HTML5/CSS3**: Modern responsive web interface
- **Vanilla JavaScript**: Dynamic user interactions
- **Jinja2 Templates**: Server-side template rendering

## 📁 Project Structure

```
gbiplabg-multiworkflow-ai/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── app/
    ├── main.py              # FastAPI application entry point
    ├── services/
    │   ├── rag.py           # RAG implementation with vector store
    │   └── flow_graph.py    # Conversational flow logic
    ├── templates/
    │   └── home.html        # Frontend user interface
    ├── uploaded_pdfs/       # PDF storage directory
    └── __pycache__/         # Python cache files
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.11.0 or higher
- pip package manager
- Git

### Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/gbiplabg/multiworkflow-ai.git
cd multiworkflow-ai
```

2. **Create Virtual Environment**
```bash
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Application**
```bash
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

5. **Access the Application**
Open your browser and navigate to: `http://localhost:8000`

## 💡 Usage Guide

### RAG Mode
1. Select "RAG" mode from the interface
2. Upload a PDF document using the file input
3. Wait for processing confirmation
4. Ask questions about the document content
5. Receive contextually accurate responses

### Flow Mode
1. Select "Flow" mode from the interface
2. Engage in conversation to provide:
   - Your name
   - Valid email address
   - Technology you're working with
3. Receive a professionally formatted LinkedIn post

## 🔍 API Endpoints

### Core Endpoints
- `GET /` - Serve the main web interface
- `POST /upload_pdf/` - Upload and process PDF documents
- `POST /chat/` - RAG-based chat with uploaded documents
- `POST /chat/flow` - Conversational flow for LinkedIn post generation

### Request/Response Examples

**PDF Upload:**
```bash
curl -X POST "http://localhost:8000/upload_pdf/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@document.pdf"
```

**RAG Chat:**
```bash
curl -X POST "http://localhost:8000/chat/" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "query=What is the main topic of the document?"
```

**Flow Chat:**
```bash
curl -X POST "http://localhost:8000/chat/flow" \
     -H "Content-Type: application/json" \
     -d '{"user_message": "Hello, I want to create a LinkedIn post"}'
```

## 🎨 Design Decisions

### Architecture Patterns
- **Modular Design**: Separated RAG and Flow services for maintainability
- **State Management**: LangGraph for complex conversational flows
- **Memory Persistence**: InMemorySaver for session continuity
- **Async Processing**: FastAPI's async capabilities for better performance

### UI/UX Considerations
- **Mode Switching**: Clear visual distinction between RAG and Flow modes
- **Progressive Enhancement**: Works without JavaScript, enhanced with JS
- **Responsive Design**: Mobile-first approach with adaptive layouts
- **Real-time Feedback**: Live status updates during file processing

### Performance Optimizations
- **Lazy Loading**: Components loaded only when needed
- **Efficient Embeddings**: Optimized sentence-transformers model
- **Memory Management**: Proper cleanup of uploaded files
- **Caching Strategy**: Vector store caching for repeated queries

## ⚠️ Limitations & Known Issues

### Technical Limitations
1. **Memory Storage**: Uses in-memory vector store (data lost on restart)
2. **File Size**: Large PDFs may cause processing delays
3. **Concurrent Users**: Limited by single-instance deployment
4. **Language Support**: Optimized for English text processing

### Scalability Considerations
- **Database**: Requires persistent vector database for production
- **File Storage**: Local storage not suitable for distributed systems
- **API Rate Limits**: Groq API has usage limitations
- **Session Management**: In-memory sessions don't scale horizontally

### Security Notes
- **File Validation**: Basic PDF validation (enhance for production)
- **Input Sanitization**: Limited input validation implemented
- **API Security**: No authentication mechanism currently implemented
- **CORS Policy**: Currently allows all origins (adjust for production)

## 🚀 Future Enhancements

### Planned Features
- [ ] Persistent vector database integration (PostgreSQL + pgvector)
- [ ] User authentication and authorization system
- [ ] Multiple document support per user
- [ ] Advanced file format support (Word, PowerPoint, etc.)
- [ ] Real-time collaboration features
- [ ] API rate limiting and throttling
- [ ] Comprehensive logging and monitoring

### Performance Improvements
- [ ] Database connection pooling
- [ ] Redis caching layer
- [ ] Async file processing with queues
- [ ] Load balancing for multi-instance deployment


**Built with ❤️ by [gbiplabg](https://github.com/gbiplabg)**