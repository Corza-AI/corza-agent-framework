/**
 * Corza Agent — React Hook
 *
 * Provides streaming agent chat in React with one hook.
 *
 * Usage:
 *   import { useAgent } from "./use-agent";
 *
 *   function Chat() {
 *     const { messages, isStreaming, sendMessage } = useAgent("/api/agent", "assistant");
 *     return (
 *       <div>
 *         {messages.map((m, i) => <p key={i}>{m.role}: {m.content}</p>)}
 *         <input onKeyDown={e => e.key === "Enter" && sendMessage(e.target.value)} />
 *       </div>
 *     );
 *   }
 */

import { useCallback, useRef, useState } from "react";

interface Message {
  role: "user" | "assistant" | "tool";
  content: string;
}

interface UseAgentReturn {
  messages: Message[];
  isStreaming: boolean;
  error: string | null;
  sessionId: string | null;
  sendMessage: (content: string) => Promise<void>;
  reset: () => void;
}

export function useAgent(
  baseUrl: string = "/api/agent",
  agentId: string = "assistant",
  headers: Record<string, string> = {},
): UseAgentReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionIdRef = useRef<string | null>(null);

  const sendMessage = useCallback(
    async (content: string) => {
      if (!content.trim() || isStreaming) return;
      setError(null);

      // Add user message
      setMessages((prev) => [...prev, { role: "user", content }]);

      try {
        // Create session on first message
        if (!sessionIdRef.current) {
          const res = await fetch(`${baseUrl}/sessions`, {
            method: "POST",
            headers: { "Content-Type": "application/json", ...headers },
            body: JSON.stringify({ agent_id: agentId }),
          });
          if (!res.ok) throw new Error("Failed to create session");
          const data = await res.json();
          sessionIdRef.current = data.session_id;
        }

        setIsStreaming(true);

        // Send message and stream response
        const res = await fetch(
          `${baseUrl}/sessions/${sessionIdRef.current}/messages`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json", ...headers },
            body: JSON.stringify({ content, stream: true }),
          },
        );

        if (!res.ok) throw new Error(`Request failed: ${res.status}`);
        if (!res.body) throw new Error("No response body");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let assistantText = "";

        // Add empty assistant message that we'll update
        setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

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
              if (event.type === "llm.text_delta" && event.data?.text) {
                assistantText += event.data.text;
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    role: "assistant",
                    content: assistantText,
                  };
                  return updated;
                });
              } else if (event.type === "tool.executing") {
                setMessages((prev) => [
                  ...prev.slice(0, -1),
                  {
                    role: "tool",
                    content: `Using ${event.data.tool_name}...`,
                  },
                  { role: "assistant", content: assistantText },
                ]);
              } else if (event.type === "error") {
                setError(event.data.message || "Agent error");
              }
            } catch {
              // Skip parse errors
            }
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setIsStreaming(false);
      }
    },
    [baseUrl, agentId, headers, isStreaming],
  );

  const reset = useCallback(() => {
    setMessages([]);
    setError(null);
    sessionIdRef.current = null;
  }, []);

  return {
    messages,
    isStreaming,
    error,
    sessionId: sessionIdRef.current,
    sendMessage,
    reset,
  };
}
