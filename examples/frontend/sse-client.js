/**
 * Corza Agent — Vanilla JS SSE Client
 *
 * Streams agent responses from your FastAPI backend to the browser.
 * Works with any frontend framework or vanilla HTML.
 *
 * Usage:
 *   const agent = new CorzaAgent("http://localhost:8000/api/agent");
 *   const session = await agent.createSession("assistant");
 *   agent.sendMessage(session.session_id, "Hello!", {
 *     onText: (text) => console.log(text),
 *     onToolCall: (name) => console.log(`Using ${name}...`),
 *     onComplete: (data) => console.log("Done!", data),
 *     onError: (err) => console.error(err),
 *   });
 */

class CorzaAgent {
  constructor(baseUrl = "/api/agent", headers = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.headers = { "Content-Type": "application/json", ...headers };
  }

  /** Create a new session for an agent. */
  async createSession(agentId, metadata = {}) {
    const res = await fetch(`${this.baseUrl}/sessions`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify({ agent_id: agentId, metadata }),
    });
    if (!res.ok) throw new Error(`Failed to create session: ${res.status}`);
    return res.json();
  }

  /** Cancel a running session and all its sub-agents (nuclear stop). */
  async cancelSession(sessionId) {
    const res = await fetch(`${this.baseUrl}/sessions/${sessionId}/cancel`, {
      method: "POST",
      headers: this.headers,
    });
    return res.json();
  }

  /** Send a message and stream the response via SSE. */
  async sendMessage(sessionId, content, callbacks = {}) {
    const { onText, onToolCall, onToolResult, onComplete, onError } = callbacks;

    const res = await fetch(`${this.baseUrl}/sessions/${sessionId}/messages`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify({ content, stream: true }),
    });

    if (!res.ok) {
      const err = await res.text();
      onError?.(new Error(err));
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        try {
          const event = JSON.parse(line.slice(6));
          const type = event.type;
          const data = event.data || {};

          if (type === "llm.text_delta" && data.text) {
            onText?.(data.text);
          } else if (type === "tool.executing") {
            onToolCall?.(data.tool_name, data.tool_call_id);
          } else if (type === "tool.result") {
            onToolResult?.(data.tool_name, data.status, data.output_preview);
          } else if (type === "session.completed") {
            onComplete?.(data);
          } else if (type === "error") {
            onError?.(new Error(data.message || "Agent error"));
          }
        } catch (e) {
          // Skip malformed events
        }
      }
    }
  }

  /** Get message history for a session. */
  async getMessages(sessionId) {
    const res = await fetch(`${this.baseUrl}/sessions/${sessionId}/messages`, {
      headers: this.headers,
    });
    return res.json();
  }

  /** Delete a session. */
  async deleteSession(sessionId) {
    await fetch(`${this.baseUrl}/sessions/${sessionId}`, {
      method: "DELETE",
      headers: this.headers,
    });
  }
}

// Export for ES modules
if (typeof module !== "undefined") module.exports = { CorzaAgent };
