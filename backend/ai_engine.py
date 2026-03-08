"""
AI Engine - All Groq LLM calls run in thread executor so they never block the async event loop.
"""
import asyncio
import logging
import json
import base64
from typing import Dict, Any, Optional, List
from functools import partial

from groq import Groq

from config import settings
from embeddings import generate_embeddings, generate_embeddings_async
from vector_index import search_index

logger = logging.getLogger(__name__)
_client: Optional[Groq] = None


def get_client() -> Optional[Groq]:
    global _client
    if _client is None and settings.groq_api_key:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        text = inner.strip()
    return json.loads(text)


def _chat_sync(system: str, user: str, temperature: float = 0.2, max_tokens: int = 4000) -> str:
    """Synchronous Groq call — always call via _chat_async in production."""
    client = get_client()
    if not client:
        raise RuntimeError("GROQ_API_KEY not configured")
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


async def _chat_async(system: str, user: str, temperature: float = 0.2, max_tokens: int = 4000) -> str:
    """Run Groq call in thread pool so the event loop stays free."""
    loop = asyncio.get_event_loop()
    fn = partial(_chat_sync, system, user, temperature, max_tokens)
    return await loop.run_in_executor(None, fn)


# ── Prompts ───────────────────────────────────────────────────────────────────

EXPLAIN_SYSTEM = "You are an expert software engineer. Always respond with valid JSON only — no markdown fences, no extra text."
EXPLAIN_PROMPT = """Analyze the following code and return a JSON object:
{{
  "overview": "Clear plain-English summary (2-3 sentences)",
  "functions": [
    {{
      "name": "function name",
      "purpose": "What it does",
      "parameters": "Parameters description",
      "returns": "What it returns",
      "logic": "Step-by-step explanation",
      "docstring": "Complete Python docstring"
    }}
  ],
  "flowchart": "Valid Mermaid flowchart TD code",
  "key_concepts": ["concepts used"],
  "potential_issues": ["bugs or code smells"],
  "complexity": "Time and space complexity"
}}

Code:
```{language}
{code}
```"""

DEBUG_SYSTEM = "You are an expert debugger. Always respond with valid JSON only — no markdown fences."
DEBUG_PROMPT = """Analyze this error and return JSON:
{{
  "error_type": "Category (TypeError, LogicError, etc.)",
  "root_cause": "Most likely root cause",
  "explanation": "Detailed explanation",
  "fix": "Specific fix to apply",
  "fixed_code": "Corrected code snippet or empty string",
  "prevention": "How to prevent this in future",
  "related_issues": ["other potential issues"],
  "confidence": "high or medium or low"
}}

Stacktrace: {stacktrace}
Error message: {error_message}
Code context: {code_context}
Console logs: {console_logs}
Screenshot: {screenshot_desc}"""

MULTIMODAL_SYSTEM = "You are an expert full-stack debugger. Always respond with valid JSON only."
MULTIMODAL_PROMPT = """Correlate these signals and return JSON:
{{
  "most_likely_file": "File most likely containing the bug",
  "most_likely_line": "Approximate line or function",
  "root_cause": "Plain English root cause",
  "signal_correlation": "How screenshot, logs, and stacktrace relate",
  "fix_steps": ["step 1", "step 2", "step 3"],
  "backend_vs_frontend": "Backend or frontend issue, and why"
}}

Screenshot: {screenshot_desc}
Console logs: {console_logs}
Stacktrace: {stacktrace}"""

DECODE_SYSTEM = "You are a JS/TS debugging expert. Always respond with valid JSON only."
DECODE_PROMPT = """Analyze this stacktrace and return JSON:
{{
  "decoded_frames": [
    {{"original": "minified frame", "decoded": "readable reference", "confidence": "high/medium/low"}}
  ],
  "likely_source_files": ["source files"],
  "entry_point": "Starting point of the error",
  "summary": "Plain English summary"
}}

Stacktrace: {stacktrace}
{source_map_section}"""

RAG_SYSTEM = (
    "You are an expert coding assistant. Your task is to help users with their "
    "question. Use the retrieved code context to inform your responses, but feel "
    "free to suggest better solutions if appropriate."
)

VISION_PROMPT = """Describe this UI screenshot for debugging. Focus on:
- Visible error messages or alerts
- Current UI state
- Any broken UI elements
- Network/console errors visible
Be specific and technical."""


# ── Public API ────────────────────────────────────────────────────────────────

async def explain_code(code: str, language: str = "python") -> Dict[str, Any]:
    try:
        prompt = EXPLAIN_PROMPT.format(code=code, language=language)
        text = await _chat_async(EXPLAIN_SYSTEM, prompt, temperature=0.2, max_tokens=4000)
        return _parse_json(text)
    except json.JSONDecodeError:
        return {"error": "AI returned malformed JSON. Try again."}
    except Exception as e:
        logger.error(f"explain_code error: {e}")
        return {"error": str(e)}


async def debug_analyze(
    stacktrace: str = "",
    error_message: str = "",
    code_context: str = "",
    console_logs: str = "",
    screenshot_desc: str = "",
) -> Dict[str, Any]:
    try:
        prompt = DEBUG_PROMPT.format(
            stacktrace=stacktrace or "none",
            error_message=error_message or "none",
            code_context=code_context or "none",
            console_logs=console_logs or "none",
            screenshot_desc=screenshot_desc or "none",
        )
        text = await _chat_async(DEBUG_SYSTEM, prompt, temperature=0.1, max_tokens=3000)
        return _parse_json(text)
    except json.JSONDecodeError:
        return {"error": "AI returned malformed JSON. Try again."}
    except Exception as e:
        logger.error(f"debug_analyze error: {e}")
        return {"error": str(e)}


async def multimodal_debug(
    stacktrace: str = "",
    console_logs: str = "",
    screenshot_desc: str = "",
) -> Dict[str, Any]:
    try:
        prompt = MULTIMODAL_PROMPT.format(
            stacktrace=stacktrace or "none",
            console_logs=console_logs or "none",
            screenshot_desc=screenshot_desc or "No screenshot provided",
        )
        text = await _chat_async(MULTIMODAL_SYSTEM, prompt, temperature=0.1, max_tokens=2000)
        return _parse_json(text)
    except json.JSONDecodeError:
        return {"error": "AI returned malformed JSON. Try again."}
    except Exception as e:
        logger.error(f"multimodal_debug error: {e}")
        return {"error": str(e)}


async def decode_minified_stacktrace(stacktrace: str, source_map_content: Optional[str] = None) -> Dict[str, Any]:
    try:
        source_map_section = (
            f"Source map (partial):\n{source_map_content[:2000]}"
            if source_map_content else "No source map provided."
        )
        prompt = DECODE_PROMPT.format(
            stacktrace=stacktrace,
            source_map_section=source_map_section,
        )
        text = await _chat_async(DECODE_SYSTEM, prompt, temperature=0.1, max_tokens=2000)
        return _parse_json(text)
    except json.JSONDecodeError:
        return {"error": "AI returned malformed JSON. Try again."}
    except Exception as e:
        logger.error(f"decode_minified_stacktrace error: {e}")
        return {"error": str(e)}


async def rag_chat(
    session_id: str,
    query: str,
    conversation_history: Optional[List[Dict]] = None,
) -> str:
    client = get_client()
    if not client:
        return "Error: GROQ_API_KEY not configured."

    # Async embedding for query — same as CodeRAG's search_code()
    query_embedding = await generate_embeddings_async(query)
    context_chunks = []
    if query_embedding is not None:
        results = search_index(session_id, query_embedding, k=5)
        # Format exactly like CodeRAG's prompt_flow.py PRE_PROMPT:
        # File, Path, Similarity score, Content
        for r in results[:3]:
            context_chunks.append(
                f"File: {r['filename']}\n"
                f"Path: {r['filepath']}\n"
                f"Similarity: {r['distance']:.3f}\n"
                f"{r['content']}"
            )

    # Mirror CodeRAG's PRE_PROMPT format exactly
    if context_chunks:
        code_context = "\n\n".join(context_chunks)
        user_content = (
            f"Based on the user's query and the following code context, provide a helpful response. "
            f"If improvements can be made, suggest them with explanations.\n\n"
            f"User Query: {query}\n\n"
            f"Retrieved Code Context:\n{code_context}\n\nYour response:"
        )
    else:
        user_content = (
            f"No relevant code found for your query. The codebase might not be "
            f"indexed yet or your query might be too specific.\n\nUser Query: {query}"
        )

    messages = [{"role": "system", "content": RAG_SYSTEM}]
    if conversation_history:
        messages.extend(conversation_history[-6:])
    messages.append({"role": "user", "content": user_content})

    def _groq_call():
        return client.chat.completions.create(
            model=settings.chat_model,
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
        ).choices[0].message.content.strip()

    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _groq_call)
    except Exception as e:
        logger.error(f"rag_chat error: {e}")
        return f"Error calling Groq API: {str(e)}"


async def analyze_image_for_debug(image_base64: str) -> str:
    client = get_client()
    if not client:
        return "No Groq API key configured."

    def _vision_call():
        return client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }],
            max_tokens=500,
        ).choices[0].message.content.strip()

    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _vision_call)
    except Exception as e:
        logger.warning(f"Vision model error: {e}")
        return f"Vision analysis unavailable: {str(e)}"
