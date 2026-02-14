"""Full-featured chatbot with persistent memory.

Uses py-cml. Set AUTH__API_KEY, CML_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL in .env.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from openai import OpenAI
from cml import CognitiveMemoryLayer


@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: datetime


class MemoryPoweredChatbot:
    """Chatbot with memory retrieval, auto-extraction, and commands (!remember, !forget, etc.)."""

    def __init__(
        self,
        session_id: str,
        llm_api_key: Optional[str] = None,
        memory_base_url: Optional[str] = None,
        memory_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        auto_remember: bool = True,
        memory_context_tokens: int = 1500,
    ):
        self.session_id = session_id
        self.llm_model = (
            llm_model or os.environ.get("OPENAI_MODEL") or os.environ.get("LLM__MODEL") or ""
        ).strip()
        self.auto_remember = auto_remember
        self.memory_context_tokens = memory_context_tokens
        base_url = (
            memory_base_url
            or os.environ.get("CML_BASE_URL")
            or os.environ.get("MEMORY_API_URL")
            or "http://localhost:8000"
        ).strip()
        self.llm = OpenAI(api_key=llm_api_key or os.getenv("OPENAI_API_KEY"))
        self.memory = CognitiveMemoryLayer(
            api_key=memory_api_key
            or os.environ.get("CML_API_KEY")
            or os.environ.get("AUTH__API_KEY"),
            base_url=base_url,
        )
        self.history: List[ConversationTurn] = []
        self.base_system_prompt = """You are a helpful assistant with persistent memory.

Use the provided memory context to personalize responses. Be conversational."""

    def _handle_command(self, message: str) -> Optional[str]:
        msg = message.strip()
        if msg.startswith("!remember "):
            info = msg[10:].strip()
            if info:
                r = self.memory.write(
                    info,
                    session_id=self.session_id,
                    context_tags=["conversation"],
                    memory_type="semantic_fact",
                )
                return f"✓ Stored: {info}" if r.success else f"✗ Failed"
            return "Usage: !remember <info>"
        elif msg.startswith("!forget "):
            q = msg[8:].strip()
            if q:
                r = self.memory.forget(query=q, action="delete")
                return f"✓ Forgot {r.affected_count} memories"
            return "Usage: !forget <query>"
        elif msg == "!stats":
            s = self.memory.stats()
            return f"Total: {s.total_memories}, Active: {s.active_memories}, By type: {s.by_type}"
        elif msg.startswith("!search "):
            q = msg[8:].strip()
            if q:
                r = self.memory.read(q, max_results=5)
                if r.memories:
                    lines = [f"Found {r.total_count}:"]
                    for m in r.memories[:5]:
                        lines.append(f"  [{m.type}] {m.text[:80]}...")
                    return "\n".join(lines)
                return "No memories found."
            return "Usage: !search <query>"
        elif msg == "!clear":
            self.history.clear()
            return "✓ Session history cleared"
        elif msg == "!help":
            return "!remember <info> | !forget <query> | !stats | !search <query> | !clear | !help"
        return None

    def _get_memory_context(self, message: str) -> str:
        try:
            turn = self.memory.turn(
                user_message=message,
                session_id=self.session_id,
                max_context_tokens=self.memory_context_tokens,
            )
            return turn.memory_context or ""
        except Exception:
            try:
                r = self.memory.read(message, max_results=10, response_format="llm_context")
                return r.context or ""
            except Exception:
                return ""

    def _extract_memorable_info(self, user_msg: str, assistant_msg: str) -> List[Tuple[str, str]]:
        prompt = f"""Extract important information worth remembering from this turn.

User: {user_msg}
Assistant: {assistant_msg}

Respond with JSON array: [{{"content": "...", "type": "semantic_fact|preference|constraint|hypothesis"}}]
If nothing worth remembering: []"""
        try:
            resp = self.llm.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=500,
            )
            data = json.loads(resp.choices[0].message.content)
            if isinstance(data, list):
                return [(i["content"], i.get("type", "semantic_fact")) for i in data]
            if isinstance(data, dict) and "items" in data:
                return [(i["content"], i.get("type", "semantic_fact")) for i in data["items"]]
        except Exception:
            pass
        return []

    def chat(self, message: str) -> str:
        cmd = self._handle_command(message)
        if cmd is not None:
            return cmd
        ctx = self._get_memory_context(message)
        system = self.base_system_prompt
        if ctx:
            system += f"\n\n--- MEMORY ---\n{ctx}"
        messages = [{"role": "system", "content": system}]
        for t in self.history[-10:]:
            messages.append({"role": t.role, "content": t.content})
        messages.append({"role": "user", "content": message})
        resp = self.llm.chat.completions.create(
            model=self.llm_model, messages=messages, max_tokens=1000
        )
        out = resp.choices[0].message.content
        self.history.append(ConversationTurn("user", message, datetime.now()))
        self.history.append(ConversationTurn("assistant", out, datetime.now()))
        if self.auto_remember:
            for content, mtype in self._extract_memorable_info(message, out):
                try:
                    self.memory.write(
                        content,
                        session_id=self.session_id,
                        context_tags=["conversation"],
                        memory_type=mtype,
                    )
                    print(f"  [Stored: {mtype}] {content[:50]}...")
                except Exception:
                    pass
        return out

    def close(self):
        self.memory.close()


def main():
    print("=" * 60)
    print("Memory-Powered Chatbot (py-cml)")
    print("=" * 60)
    print("Commands: !remember, !forget, !stats, !search, !clear, !help. Type 'quit' to exit.\n")
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in .env")
        return
    if not (os.environ.get("OPENAI_MODEL") or os.environ.get("LLM__MODEL")):
        print("Set OPENAI_MODEL or LLM__MODEL in .env")
        return
    chatbot = MemoryPoweredChatbot(session_id="chatbot-demo", auto_remember=True)
    try:
        print("Bot: Hello! I remember things across conversations. How can I help?\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "bye"):
                print("\nBot: Goodbye!")
                break
            if not user_input:
                continue
            response = chatbot.chat(user_input)
            print(f"\nBot: {response}\n")
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        chatbot.close()


if __name__ == "__main__":
    main()
