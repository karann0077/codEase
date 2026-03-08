"""
AI Engine - All LLM calls go through Groq's free API.
Groq is OpenAI-compatible so the SDK interface is nearly identical.
Adapted from ChatDBG's structured prompt building approach.
"""
import logging
import json
import base64
from typing import Dict, Any, Optional, List

from groq import Groq

from config import settings
from embeddings import generate_embeddings
from vector_index import search_index

logger = logging.getLogger(__name__)


def get_client() -> Optional[Groq]:
    if not settings.groq_api_key:
        return None
    return Groq(api_key=settings.groq_api_key)


def _parse_json(text: str) -> dict:
    """Strip markdown fences and parse JSON from LLM response."""
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        inner = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        text = inner.strip()
    return json.loads(text)


def _chat(system: str, user: str, temperature: float = 0.2, max_tokens: int = 4000) -> str:
    """Single Groq chat completion — reusable helper."""
    client = get_client()
    if not client:
        raise RuntimeError("GROQ_API_KEY not configured")
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ── Prompts ──────────────────────────────────────────────────────────────────

EXPLAIN_SYSTEM = "You are an expert software engineer and technical writer. Always respond with valid JSON only — no markdown fences, no extra text before or after the JSON."

EXPLAIN_PROMPT = """Analyze the following code and return a JSON object with this exact structure:
{{
  "overview": "Clear plain-English summary of what this code does (2-3 sentences)",
  "functions": [
    {{
      "name": "function name",
      "purpose": "What this function does in plain English",
      "parameters": "Description of each parameter",
      "returns": "What it returns",
      "logic": "Step-by-step explanation of the logic",
      "docstring": "A complete docstring for this function (Python style)"
    }}
  ],
  "flowchart": "Valid Mermaid flowchart TD code showing the high-level control flow",
  "key_concepts": ["programming concepts used"],
  "potential_issues": ["bugs or code smells spotted"],
  "complexity": "Time and space complexity if applicable"
}}

Code to analyze:
```{language}
{code}
```"""

DEBUG_SYSTEM = "You are an expert debugger. Always respond with valid JSON only — no markdown fences, no extra text."

DEBUG_PROMPT = """Analyze this error and return JSON:
{{
  "error_type": "Category (TypeError, LogicError, NetworkError, etc.)",
  "root_cause": "Most likely root cause in plain English",
  "explanation": "Detailed explanation of why this error occurs",
  "fix": "The specific fix to apply",
  "fixed_code": "Corrected code snippet (if applicable, else empty string)",
  "prevention": "How to prevent this class of error in future",
  "related_issues": ["other potential issues to watch for"],
  "confidence": "high or medium or low"
}}

Stacktrace: {stacktrace}
Error message: {error_message}
Code context: {code_context}
Console logs: {console_logs}
Screenshot description: {screenshot_desc}"""

MULTIMODAL_SYSTEM = "You are an expert full-stack debugger. Always respond with valid JSON only."

MULTIMODAL_PROMPT = """Correlate these signals to identify the root cause. Return JSON:
{{
  "most_likely_file": "File most likely containing the bug",
  "most_likely_line": "Approximate line number or function name",
  "root_cause": "Plain English explanation of the root cause",
  "signal_correlation": "How the screenshot, logs, and stacktrace relate to each other",
  "fix_steps": ["step 1", "step 2", "step 3"],
  "backend_vs_frontend": "Is this a backend or frontend issue, and why?"
}}

Screenshot description: {screenshot_desc}
Console logs:
{console_logs}
Stacktrace:
{stacktrace}"""

DECODE_SYSTEM = "You are a JavaScript/TypeScript debugging expert. Always respond with valid JSON only."

DECODE_PROMPT = """Analyze this stacktrace (may be minified/obfuscated) and return JSON:
{{
  "decoded_frames": [
    {{"original": "minified frame text", "decoded": "readable file:line reference", "confidence": "high or medium or low"}}
  ],
  "likely_source_files": ["list of likely original source files"],
  "entry_point": "Most likely starting point of the error chain",
  "summary": "Plain English summary of what went wrong"
}}

Stacktrace:
{stacktrace}

{source_map_section}"""

RAG_SYSTEM = """You are an expert software engineer with deep knowledge of this codebase.
Answer questions clearly and concisely. Reference specific files and functions when relevant.
If the context doesn't contain the answer, say so — don't make things up."""

RAG_PROMPT = """Retrieved code context from the codebase:
{code_context}

---
User question: {query}"""

VISION_PROMPT = """Describe this UI screenshot for debugging purposes. Focus on:
- Visible error messages or alerts
- Current UI state (what page/component is shown)
- Any unusual or broken UI elements
- What the user was likely trying to do
- Network/console errors visible on screen
Be specific and technical."""


# ── Public API ────────────────────────────────────────────────────────────────

async def explain_code(code: str, language: str = "python") -> Dict[str, Any]:
    """Explain code: overview, functions with docstrings, Mermaid flowchart."""
    try:
        prompt = EXPLAIN_PROMPT.format(code=code, language=language)
        text = _chat(EXPLAIN_SYSTEM, prompt, temperature=0.2, max_tokens=4000)
        return _parse_json(text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error in explain_code: {e}")
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
    """Analyze error: root cause, fix, prevention."""
    try:
        prompt = DEBUG_PROMPT.format(
            stacktrace=stacktrace or "none",
            error_message=error_message or "none",
            code_context=code_context or "none",
            console_logs=console_logs or "none",
            screenshot_desc=screenshot_desc or "none",
        )
        text = _chat(DEBUG_SYSTEM, prompt, temperature=0.1, max_tokens=3000)
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
    """Correlate screenshot + logs + stacktrace → pinpoint root cause."""
    try:
        prompt = MULTIMODAL_PROMPT.format(
            stacktrace=stacktrace or "none",
            console_logs=console_logs or "none",
            screenshot_desc=screenshot_desc or "No screenshot provided",
        )
        text = _chat(MULTIMODAL_SYSTEM, prompt, temperature=0.1, max_tokens=2000)
        return _parse_json(text)
    except json.JSONDecodeError:
        return {"error": "AI returned malformed JSON. Try again."}
    except Exception as e:
        logger.error(f"multimodal_debug error: {e}")
        return {"error": str(e)}


async def decode_minified_stacktrace(stacktrace: str, source_map_content: Optional[str] = None) -> Dict[str, Any]:
    """Decode minified/obfuscated JS/TS stacktraces."""
    try:
        source_map_section = (
            f"Source map content (partial):\n{source_map_content[:2000]}"
            if source_map_content
            else "No source map provided."
        )
        prompt = DECODE_PROMPT.format(
            stacktrace=stacktrace,
            source_map_section=source_map_section,
        )
        text = _chat(DECODE_SYSTEM, prompt, temperature=0.1, max_tokens=2000)
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
    """RAG-powered chat: search indexed codebase, inject context, answer with Groq."""
    client = get_client()
    if not client:
        return "Error: GROQ_API_KEY not configured. Add it to your .env file."

    # Vector search for relevant code chunks
    query_embedding = generate_embeddings(query)
    context_chunks = []
    if query_embedding is not None:
        results = search_index(session_id, query_embedding, k=5)
        for r in results[:3]:
            context_chunks.append(
                f"File: {r['filepath']} ({r['language']})\n```\n{r['content'][:1500]}\n```"
            )

    code_context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No code indexed yet for this session."

    # Build messages with conversation history (last 6 messages = 3 exchanges)
    messages = [{"role": "system", "content": RAG_SYSTEM}]
    if conversation_history:
        messages.extend(conversation_history[-6:])
    messages.append({
        "role": "user",
        "content": RAG_PROMPT.format(code_context=code_context, query=query),
    })

    try:
        response = client.chat.completions.create(
            model=settings.chat_model,
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"rag_chat error: {e}")
        return f"Error calling Groq API: {str(e)}"


async def analyze_image_for_debug(image_base64: str) -> str:
    """
    Describe a screenshot for debugging context.
    NOTE: Groq's vision support depends on the model. We use llama-3.2-11b-vision-preview
    which supports images on Groq's free tier.
    Falls back to a text description if vision unavailable.
    """
    client = get_client()
    if not client:
        return "No Groq API key configured."

    try:
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",  # Groq's free vision model
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }],
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Vision model error (falling back): {e}")
        return f"Screenshot uploaded but vision analysis unavailable: {str(e)}. Describe the screenshot manually in the console logs field."
