# Investment Banking AI Assistant API

A FastAPI-based conversational AI assistant specialized in investment banking, powered by Google's Gemini AI and enhanced with real-time web search capabilities.

## Features

- **Investment Banking Expertise**: M&A, IPOs, Capital Markets, Corporate Finance, Risk Management
- **Real-time Search**: Current market data, news, and financial information via Serper API
- **Conversational AI**: Natural language processing with context awareness
- **Session Management**: Multi-user support with isolated conversation sessions
- **Context Updates**: Dynamic context adjustment for personalized responses
- **FastAPI**: High-performance API with automatic documentation

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd investment-banking-ai

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
SERPER_API_KEY=your_serper_api_key_here
```

### 3. Run the API

```bash
# Start the FastAPI server
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the API

- **API Server**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

## API Usage

### Using the Client Script

```bash
# Interactive chat mode
python app.py


### Direct API Calls

```bash
# Start a conversation
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the key factors in M&A valuation?"}'

# Update context
curl -X POST "http://localhost:8000/update_context" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your_session_id",
    "context_data": {
      "client_sector": "Technology",
      "deal_type": "Acquisition"
    }
  }'
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status and info |
| POST | `/chat` | Send message to AI assistant |
| POST | `/update_context` | Update session context |
| GET | `/session/{session_id}/info` | Get session information |
| DELETE | `/session/{session_id}` | Delete session |
| GET | `/sessions` | List all active sessions |

## Example Queries

- "What are the current IPO market trends?"
- "Explain the differences between DCF and comparable company analysis"
- "What regulatory changes affect investment banking this year?"
- "How do you structure a leveraged buyout?"
- "What are the latest M&A multiples in the tech sector?"

## Configuration

The assistant comes pre-configured with:
- **Specialization**: Investment Banking
- **Focus Areas**: M&A, Capital Markets, Corporate Finance, Risk Management
- **Market Focus**: Global Financial Markets
- **Regulatory Knowledge**: SEC, FINRA, Basel III, MiFID II

## Requirements

- Python 3.8+
- Google Gemini API key
- Serper API key (for web search)

## Dependencies

See `requirements.txt` for the complete list. Key dependencies:
- `fastapi` - Web framework
- `google-generativeai` - Google Gemini AI
- `requests` - HTTP client for search API
- `uvicorn` - ASGI server

## License

MIT License - Feel free to use and modify as needed.
