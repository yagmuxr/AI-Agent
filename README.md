# AI-Agent

A simple **research assistant agent** built with [LangChain].  
It can connect to different LLM providers (OpenAI, Anthropic, Google Gemini) and return **structured outputs** using Pydantic.  

---

## Quick Start

1. **Clone & setup environment**
   ```bash
   git clone https://github.com/yagmuxr/AI-Agent.git
   cd AI-Agent
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt

   Add your API key(s)
2.Create a .env file in the project root and add at least one provider key:

OPENAI_API_KEY=sk-xxxx
ANTHROPIC_API_KEY=sk-ant-xxxx
GOOGLE_API_KEY=xxxx



3.Run the app on the terminal by:

python main.py

