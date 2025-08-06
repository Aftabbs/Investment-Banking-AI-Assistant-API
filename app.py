import google.generativeai as genai
import json
import time
from typing import Dict, Any, Optional, List
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel 
from dotenv import load_dotenv
import atexit
import signal
import sys
import uvicorn
from contextlib import asynccontextmanager

load_dotenv()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context_data: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: float

class ContextUpdateRequest(BaseModel):
    session_id: str
    context_data: Dict[str, Any]

bot_instances = {}

class SerperSearchTool:
    """Tool for searching the internet using Serper API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://google.serper.dev/search"
    
    def search(self, query: str, location: Optional[str] = None, num_results: int = 5) -> Dict[str, Any]:
        """Search the internet for information"""
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'num': num_results
        }
        
        if location:
            payload['location'] = location
            
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def search_news(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Search for recent news articles"""
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'type': 'news',
            'num': num_results,
            'tbs': 'qdr:m'
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

def get_investment_banking_system_prompt() -> str:
    """Get the system prompt for Investment Banking Assistant"""
    return """You are Agent Alpha, an expert Investment Banking AI Assistant with comprehensive knowledge in:

CORE EXPERTISE:
- Investment Banking Services (M&A, IPOs, Debt/Equity Capital Markets)
- Financial Analysis & Valuation (DCF, Trading Comps, Transaction Comps)
- Capital Markets & Securities
- Corporate Finance & Restructuring
- Risk Management & Compliance
- Market Analysis & Economic Trends
- Financial Modeling & Analytics
- Regulatory Environment (SEC, FINRA, Basel III)

COMMUNICATION STYLE:
- Professional yet approachable
- Data-driven and analytical
- Precise financial terminology
- Confident and authoritative
- Provide specific insights and actionable information

CAPABILITIES:
- Market trend analysis
- Deal structure recommendations
- Financial metric calculations
- Industry sector insights
- Regulatory guidance
- Investment strategy recommendations
- Risk assessment
- Economic impact analysis

Always provide detailed, accurate, and current information relevant to investment banking and finance.
Be direct and specific in your responses. Present information as part of your expertise.
"""

def get_validator_system_prompt() -> str:
    """Get the system prompt for input validation"""
    return """You are a content validator for an Investment Banking AI Assistant.

VALIDATION CRITERIA:
Your job is to determine if user input is appropriate for an investment banking assistant.

VALID topics include:
- Investment banking services and processes
- Financial markets and analysis
- Corporate finance questions
- M&A and capital markets
- Economic trends and market conditions
- Financial regulations and compliance
- Investment strategies and recommendations
- Risk management
- Financial modeling and valuation
- Industry analysis

INVALID topics include:
- Personal financial advice for individuals
- Tax preparation or personal tax advice
- Basic arithmetic or general math
- Non-financial topics
- Inappropriate or offensive content
- Requests for illegal financial activities

RESPONSE FORMAT:
- If valid: "VALID"
- If invalid: "INVALID: [brief redirect message focusing on investment banking topics]"

Example redirects:
"INVALID: I specialize in investment banking and corporate finance. I can help with M&A analysis, capital markets, financial modeling, or market insights."
"""

class ConversationalInvestmentBankingSystem:
    def __init__(self, gemini_api_key: str, serper_api_key: str, session_id: str):
        """Initialize the Conversational Investment Banking System with search capabilities"""
        genai.configure(api_key=gemini_api_key)
        
        self.session_id = session_id
        self.investment_model = genai.GenerativeModel("gemini-2.0-flash")
        self.validator_model = genai.GenerativeModel("gemini-2.0-flash")
        
        self.search_tool = SerperSearchTool(serper_api_key)
        
        self.investment_chat = None
        self.validator_chat = None
        
        self.json_context = {}
        self.conversation_memory = []
        self.search_history = []
        self.created_at = time.time()
        self.last_activity = time.time()
        
        self._setup_conversational_agents()
    
    def _setup_conversational_agents(self):
        """Set up both agents as conversational chatbots"""
        
        investment_system_prompt = get_investment_banking_system_prompt() + """

You have comprehensive knowledge about investment banking, financial markets, corporate finance, and economic trends.
When providing information about market conditions, deal flows, regulatory changes, or financial analysis, 
present it naturally as part of your expertise.

Be confident and specific in your responses. Provide detailed analysis and insights.
Always aim to provide specific, actionable information naturally integrated into the conversation.

Focus on:
- Market analysis and trends
- Deal structures and valuations
- Regulatory impacts
- Risk assessments
- Investment recommendations
- Financial modeling insights
"""

        validator_system_prompt = get_validator_system_prompt()
        
        self.investment_chat = self.investment_model.start_chat(history=[])
        self.validator_chat = self.validator_model.start_chat(history=[])
        
        self.investment_chat.send_message(investment_system_prompt)
        self.validator_chat.send_message(validator_system_prompt)

    def _should_search(self, user_input: str) -> tuple[bool, str]:
        """Determine if the query requires internet search"""
        search_keywords = [
            'current market', 'latest', 'recent', 'today', 'this week', 'this month',
            'stock price', 'market cap', 'news about', 'earnings report', 'financial results',
            'merger announcement', 'acquisition', 'IPO', 'market trends', 'economic data',
            'interest rates', 'fed announcement', 'gdp', 'inflation', 'unemployment',
            'sector analysis', 'industry report', 'competitor analysis', 'market volatility',
            'trading volume', 'market performance', 'regulatory changes', 'sec filing',
            'analyst report', 'credit rating', 'bond yields', 'currency exchange',
            'commodity prices', 'market outlook', 'financial forecast'
        ]
        
        input_lower = user_input.lower()
        
        # Check for search indicators
        for keyword in search_keywords:
            if keyword in input_lower:
                return True, self._generate_search_query(user_input)
        
        # Ask the model if search is needed
        try:
            prompt = f"""Analyze this investment banking query: "{user_input}"
            
Does this require current market data or recent financial information from the internet? 
If yes, provide a good search query. If no, respond with "NO_SEARCH".
Format: SEARCH: <query> or NO_SEARCH"""

            response = self.investment_chat.send_message(prompt)
            result = response.text.strip()
            
            if result.startswith("SEARCH:"):
                query = result.replace("SEARCH:", "").strip()
                return True, query
            
            return False, ""
            
        except Exception:
            return False, ""

    def _generate_search_query(self, user_input: str) -> str:
        """Generate an optimized search query from user input"""
        # Add financial/investment banking context to queries
        financial_terms = ['financial', 'investment banking', 'market', 'stock', 'trading']
        
        if not any(term in user_input.lower() for term in financial_terms):
            return f"{user_input} financial market investment banking"
        return user_input

    def _perform_search(self, query: str) -> Dict[str, Any]:
        """Perform search and return results"""
        
        if any(word in query.lower() for word in ['news', 'latest', 'recent', 'today']):
            results = self.search_tool.search_news(query)
        else:
            results = self.search_tool.search(query)
        
        self.search_history.append({
            'query': query,
            'timestamp': time.time(),
            'results_count': len(results.get('organic', []))
        })
        
        return results

    def _format_search_results(self, search_results: Dict[str, Any]) -> str:
        """Format search results for the model to use"""
        if 'error' in search_results:
            return ""
        
        formatted_results = []
        
        if 'organic' in search_results:
            for i, result in enumerate(search_results['organic'][:6], 1):
                formatted_results.append(f"""
Market Information {i}:
{result.get('title', 'N/A')}
{result.get('snippet', 'N/A')}
Source: {result.get('link', 'N/A')}
""")
        
        if 'news' in search_results:
            for i, news in enumerate(search_results['news'][:5], 1):
                formatted_results.append(f"""
Financial News {i}:
{news.get('title', 'N/A')}
{news.get('snippet', 'N/A')}
Date: {news.get('date', 'N/A')}
""")
        
        return "\n".join(formatted_results) if formatted_results else ""

    def _is_valid_input(self, user_input: str) -> tuple[bool, str]:
        """Validation check using the validator agent"""
        try:
            validation_prompt = f"""Validate this user input: "{user_input}"
            
Previous conversation context: {self._get_recent_context()}

Is this appropriate for an investment banking assistant?"""

            response = self.validator_chat.send_message(validation_prompt)
            result = response.text.strip()
            
            if result.startswith("VALID"):
                return True, ""
            elif result.startswith("INVALID"):
                redirect_msg = result.replace("INVALID:", "").strip()
                if not redirect_msg:
                    redirect_msg = "I specialize in investment banking and corporate finance. I can help with M&A analysis, capital markets, financial modeling, or market insights. What would you like to explore?"
                return False, redirect_msg
            else:
                return True, ""
                
        except Exception:
            return True, ""

    def _get_recent_context(self) -> str:
        """Get recent conversation context"""
        if len(self.conversation_memory) <= 3:
            return json.dumps(self.conversation_memory)
        else:
            return json.dumps(self.conversation_memory[-3:])

    def _add_to_memory(self, user_msg: str, agent_response: str):
        """Add exchange to conversation memory"""
        self.conversation_memory.append({
            "user": user_msg,
            "agent": agent_response,
            "timestamp": time.time()
        })
        
        # Keep last 25 exchanges for context
        if len(self.conversation_memory) > 25:
            self.conversation_memory = self.conversation_memory[-25:]

    def update_context(self, context_data: Dict[str, Any]):
        """Update context information"""
        self.json_context.update(context_data)
        self.last_activity = time.time()

    def chat(self, user_input: str) -> str:
        """Main chat method - handles user input and returns response"""
        
        self.last_activity = time.time()
        user_input = user_input.strip()
        
        if not user_input:
            return "I'm here to help with investment banking and financial market insights! What would you like to discuss?"
        
        is_valid, validation_msg = self._is_valid_input(user_input)
        
        if not is_valid:
            return validation_msg
        
        try:
            should_search, search_query = self._should_search(user_input)
            search_results_text = ""
            
            if should_search:
                search_results = self._perform_search(search_query)
                search_results_text = self._format_search_results(search_results)
            
            # Construct context prompt
            context_prompt = f"""User query: "{user_input}"

Recent conversation: {self._get_recent_context()}

Current context: {json.dumps(self.json_context) if self.json_context else "None"}

{f"Current Market Data Available: {search_results_text}" if search_results_text else ""}

Instructions:
- Respond as Agent Alpha, the investment banking expert
- Provide detailed, professional analysis
- Be confident and specific in your responses
- Integrate market data seamlessly into your response
- Focus on actionable insights and recommendations
- Use appropriate financial terminology"""

            response = self.investment_chat.send_message(context_prompt)
            agent_response = response.text
            
            self._add_to_memory(user_input, agent_response)
            
            return agent_response
            
        except Exception as e:
            error_msg = "I'm experiencing a technical issue. Could you please try your question again?"
            return error_msg


# FastAPI setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting Investment Banking API...")
    yield
    # Shutdown
    print("Shutting down Investment Banking API...")

app = FastAPI(
    title="Investment Banking AI Assistant API",
    description="API for Agent Alpha - Investment Banking AI Assistant with search capabilities",
    version="1.0.0",
    lifespan=lifespan
)

def get_or_create_bot_instance(session_id: str) -> ConversationalInvestmentBankingSystem:
    """Get existing bot instance or create new one"""
    global bot_instances
    
    if session_id not in bot_instances:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        SERPER_API_KEY = os.getenv("SERPER_API_KEY")
        
        if not GEMINI_API_KEY or not SERPER_API_KEY:
            raise HTTPException(status_code=500, detail="API keys not configured")
        
        bot_instances[session_id] = ConversationalInvestmentBankingSystem(
            GEMINI_API_KEY, SERPER_API_KEY, session_id
        )
        
        # Set default context
        default_context = {
            "specialization": "Investment Banking",
            "focus_areas": ["M&A", "Capital Markets", "Corporate Finance", "Risk Management"],
            "market_focus": "Global Financial Markets",
            "regulatory_knowledge": ["SEC", "FINRA", "Basel III", "MiFID II"]
        }
        bot_instances[session_id].update_context(default_context)
    
    return bot_instances[session_id]

@app.get("/")
async def root():
    return {
        "message": "Investment Banking AI Assistant API", 
        "version": "1.0.0",
        "status": "active"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """Chat with the Investment Banking AI Assistant"""
    try:
        session_id = request.session_id or f"session_{int(time.time())}_{hash(request.message) % 10000}"
        
        bot = get_or_create_bot_instance(session_id)
        
        # Update context if provided
        if request.context_data:
            bot.update_context(request.context_data)
        
        response_text = bot.chat(request.message)
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            timestamp=time.time()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/update_context")
async def update_context(request: ContextUpdateRequest):
    """Update context for a specific session"""
    try:
        if request.session_id not in bot_instances:
            raise HTTPException(status_code=404, detail="Session not found")
        
        bot = bot_instances[request.session_id]
        bot.update_context(request.context_data)
        
        return {"message": "Context updated successfully", "session_id": request.session_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating context: {str(e)}")

@app.get("/session/{session_id}/info")
async def get_session_info(session_id: str):
    """Get session information"""
    if session_id not in bot_instances:
        raise HTTPException(status_code=404, detail="Session not found")
    
    bot = bot_instances[session_id]
    
    return {
        "session_id": session_id,
        "created_at": bot.created_at,
        "last_activity": bot.last_activity,
        "conversation_length": len(bot.conversation_memory),
        "search_count": len(bot.search_history),
        "current_context": bot.json_context
    }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    global bot_instances
    
    if session_id not in bot_instances:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del bot_instances[session_id]
    
    return {"message": "Session deleted successfully", "session_id": session_id}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    global bot_instances
    
    sessions = []
    for session_id, bot in bot_instances.items():
        sessions.append({
            "session_id": session_id,
            "created_at": bot.created_at,
            "last_activity": bot.last_activity,
            "conversation_length": len(bot.conversation_memory)
        })
    
    return {"active_sessions": len(sessions), "sessions": sessions}

if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY") or not os.getenv("SERPER_API_KEY"):
        print("Error: Please set both GEMINI_API_KEY and SERPER_API_KEY in your .env file")
        sys.exit(1)
    
    print("Starting Investment Banking AI Assistant API...")
    print("API Documentation available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",  
        host="0.0.0.0",
        port=8000,
        reload=True

    )
