"""Settings read from the environment. Locally, .env.local is loaded if python-dotenv
is installed (dev group); on Vercel the variables come from the project settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env.local")
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# Open-weight models served by Groq. Groq no longer lists Llama 3.x chat models, so the
# agents use gpt-oss; Meta's Llama Prompt Guard 2 scores prompt injection.
AGENT_MODEL = os.getenv("AGENT_MODEL", "openai/gpt-oss-120b")
GUARD_MODEL = os.getenv("GUARD_MODEL", "openai/gpt-oss-20b")
INJECTION_MODEL = os.getenv("INJECTION_MODEL", "meta-llama/llama-prompt-guard-2-86m")
EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 768


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return url


def keys_present() -> bool:
    return all(os.getenv(k) for k in ("GROQ_API_KEY", "GOOGLE_API_KEY", "DATABASE_URL"))
