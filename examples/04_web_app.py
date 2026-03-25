"""
04 — Complete Web Application

A full FastAPI app with an agent that streams responses to the browser.
Run it and open http://localhost:8000 in your browser.

Usage:
    pip install "corza-agents[openai]"   # or [anthropic]
    export OPENAI_API_KEY="sk-..."
    python examples/04_web_app.py

Or with Ollama (free, local):
    ollama pull qwen3:8b && ollama serve
    python examples/04_web_app.py
"""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from corza_agents import AgentDefinition, ToolRegistry, create_app, tool


# --- Define your tools ----------------------------------------------------

@tool(description="Search the knowledge base for information")
async def search(query: str) -> str:
    # In production, query your database or API here
    return f"Found 3 results for '{query}': AI agents are autonomous systems that use LLMs to reason and act."


@tool(description="Get the current date")
def today() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# --- Define your agent ----------------------------------------------------

assistant = AgentDefinition(
    name="assistant",
    model="ollama:qwen3:8b",  # Change to "openai:gpt-5.4" etc.
    system_prompt="You are a helpful assistant. Use your tools to answer questions.",
    tools=["search", "today"],
)

# --- Create the app -------------------------------------------------------

tools = ToolRegistry()
tools.register_many([search, today])

app = create_app(
    agents={"assistant": assistant},
    tool_registry=tools,
    # db_url="postgresql+asyncpg://user:pass@localhost:5432/mydb",  # production
)


# --- Add a simple HTML frontend ------------------------------------------

CHAT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Corza Agent</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, system-ui, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 20px; }
        h1 { margin-bottom: 20px; }
        #messages { min-height: 200px; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 12px; white-space: pre-wrap; font-size: 14px; line-height: 1.6; }
        #input-row { display: flex; gap: 8px; }
        #message { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; }
        button { padding: 10px 20px; background: #2563eb; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; }
        button:disabled { opacity: 0.5; }
        .tool { color: #666; font-style: italic; }
        .error { color: red; }
    </style>
</head>
<body>
    <h1>Corza Agent</h1>
    <div id="messages"></div>
    <div id="input-row">
        <input id="message" placeholder="Ask me anything..." autofocus />
        <button onclick="send()">Send</button>
    </div>

    <script>
        let sessionId = null;
        const msgDiv = document.getElementById('messages');
        const input = document.getElementById('message');

        input.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });

        async function send() {
            const content = input.value.trim();
            if (!content) return;
            input.value = '';
            msgDiv.textContent += '\\nYou: ' + content + '\\n\\n';

            // Create session if first message
            if (!sessionId) {
                const res = await fetch('/api/agent/sessions', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({agent_id: 'assistant'}),
                });
                const data = await res.json();
                sessionId = data.session_id;
            }

            // Stream response via SSE
            msgDiv.textContent += 'Assistant: ';
            const res = await fetch(`/api/agent/sessions/${sessionId}/messages`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({content: content, stream: true}),
            });

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const {done, value} = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, {stream: true});

                const lines = buffer.split('\\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const event = JSON.parse(line.slice(6));
                            if (event.data?.text) {
                                msgDiv.textContent += event.data.text;
                            } else if (event.type === 'tool.executing') {
                                msgDiv.textContent += `\\n  [using ${event.data.tool_name}...]\\n`;
                            }
                        } catch(e) {}
                    }
                }
                msgDiv.scrollTop = msgDiv.scrollHeight;
            }
            msgDiv.textContent += '\\n';
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return CHAT_HTML


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
