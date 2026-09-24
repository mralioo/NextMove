"""
ai_connection.py
================
Central AI connection module for the InnoTrans Hackathon.

Usage
-----
Quick test (run directly):
    python ai_connection.py

Import in your code:
    from ai_connection import ask, AzureOpenAIClient

The endpoint uses the Azure OpenAI **Responses API** (not the Chat Completions
API), so the request/response shape differs slightly from the standard openai SDK.
This module uses only stdlib (json, urllib) so no extra dependencies are needed,
but it also supports python-dotenv if installed.
"""

from __future__ import annotations

import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Load .env (optional – works without python-dotenv too)
# ---------------------------------------------------------------------------
def _load_dotenv(dotenv_path: str = ".env") -> None:
    """Minimal .env loader (no dependency required)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dotenv_path)
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:   # don't override real env vars
                os.environ[key] = value


_load_dotenv()


# ---------------------------------------------------------------------------
# Configuration (read from environment / .env)
# ---------------------------------------------------------------------------
class Config:
    ENDPOINT: str = os.environ.get(
        "AZURE_OPENAI_ENDPOINT",
        "https://alstom-innotrans-pub.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview",
    )
    API_KEY: str = os.environ.get("AZURE_OPENAI_API_KEY", "")
    MODEL: str = os.environ.get("AZURE_OPENAI_MODEL", "gpt-5.6-luna")
    TIMEOUT: int = int(os.environ.get("AI_REQUEST_TIMEOUT", "30"))


# ---------------------------------------------------------------------------
# Low-level client
# ---------------------------------------------------------------------------
class AzureOpenAIClient:
    """
    Thin wrapper around the Azure OpenAI Responses API.

    Attributes
    ----------
    endpoint : str   – Full URL including api-version query param
    api_key  : str   – Azure api-key header value
    model    : str   – Deployment / model name
    timeout  : int   – HTTP timeout in seconds
    """

    def __init__(
        self,
        endpoint: str = Config.ENDPOINT,
        api_key: str = Config.API_KEY,
        model: str = Config.MODEL,
        timeout: int = Config.TIMEOUT,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Core request
    # ------------------------------------------------------------------
    def raw_request(self, payload: dict) -> dict:
        """
        POST *payload* to the Responses endpoint and return the parsed JSON.

        Raises
        ------
        ValueError  – Missing api_key
        HTTPError   – Non-2xx HTTP response (re-raised with readable message)
        URLError    – Network-level error
        """
        if not self.api_key:
            raise ValueError(
                "AZURE_OPENAI_API_KEY is not set. "
                "Add it to your .env file or set the environment variable."
            )

        data = json.dumps(payload).encode("utf-8")
        req = Request(self.endpoint, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("api-key", self.api_key)

        try:
            with urlopen(req, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as err:
            raw = err.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(raw).get("error", {}).get("message", raw)
            except json.JSONDecodeError:
                detail = raw
            raise HTTPError(
                err.url, err.code, f"HTTP {err.code}: {detail}", err.headers, None
            ) from err

    # ------------------------------------------------------------------
    # Convenience: extract text from response
    # ------------------------------------------------------------------
    @staticmethod
    def extract_text(result: dict) -> str:
        """
        Pull the assistant's reply text from the Responses API shape:
            output -> content (output_text) -> text
        Returns an empty string if the path is not found.
        """
        for item in result.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text" and "text" in content:
                    return content["text"]
        return ""

    # ------------------------------------------------------------------
    # High-level: send a single prompt, get text back
    # ------------------------------------------------------------------
    def ask(self, prompt: str, **extra_payload) -> str:
        """
        Send *prompt* to the model and return the text response.

        Parameters
        ----------
        prompt        : The user message / instruction.
        **extra_payload : Any additional keys to merge into the JSON body
                          (e.g. temperature, max_output_tokens, …).
        """
        payload = {"model": self.model, "input": prompt, **extra_payload}
        result = self.raw_request(payload)
        return self.extract_text(result)

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------
    def test_connection(self) -> bool:
        """
        Send a minimal ping and print the result.
        Returns True on success, False on failure.
        """
        print(f"  Endpoint : {self.endpoint}")
        print(f"  Model    : {self.model}")
        print(f"  API key  : {'*' * max(0, len(self.api_key) - 8)}{self.api_key[-8:] if self.api_key else '(not set)'}")
        print()
        try:
            reply = self.ask("Say 'connection successful' in exactly those words.")
            print(f"✅  Response : {reply!r}")
            return True
        except ValueError as exc:
            print(f"❌  Config error: {exc}")
        except HTTPError as exc:
            print(f"❌  HTTP error: {exc}")
        except URLError as exc:
            print(f"❌  Network error: {exc.reason}")
        except Exception as exc:  # noqa: BLE001
            print(f"❌  Unexpected error: {exc}")
        return False


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------
_default_client: AzureOpenAIClient | None = None


def _get_client() -> AzureOpenAIClient:
    global _default_client
    if _default_client is None:
        _default_client = AzureOpenAIClient()
    return _default_client


def ask(prompt: str, **kwargs) -> str:
    """Shortcut: send a prompt using the default client."""
    return _get_client().ask(prompt, **kwargs)


# ---------------------------------------------------------------------------
# CLI entry-point: python ai_connection.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  InnoTrans – Azure OpenAI Connection Test")
    print("=" * 60)
    client = AzureOpenAIClient()
    success = client.test_connection()
    print("=" * 60)
    sys.exit(0 if success else 1)
