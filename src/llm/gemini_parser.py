"""Optional Gemini parser for free-form shopping conversations."""

from __future__ import annotations

import json
import os
from copy import deepcopy


EMPTY_STATE = {
    "intent": "browsing",
    "category": None,
    "brand": None,
    "color": None,
    "material": None,
    "size": None,
    "style": None,
    "use_case": None,
    "budget_max": None,
    "features": [],
    "clarification_question": "What kind of product are you looking for?",
    "confidence": 0.0,
}


STATE_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["buying", "browsing"],
            "description": "Buying when requirements are specific; browsing when exploratory.",
        },
        "category": {"type": ["string", "null"]},
        "brand": {"type": ["string", "null"]},
        "color": {"type": ["string", "null"]},
        "material": {"type": ["string", "null"]},
        "size": {"type": ["string", "null"]},
        "style": {"type": ["string", "null"]},
        "use_case": {"type": ["string", "null"]},
        "budget_max": {"type": ["number", "null"]},
        "features": {"type": "array", "items": {"type": "string"}},
        "clarification_question": {
            "type": "string",
            "description": "One useful, short follow-up question, or an empty string.",
        },
        "confidence": {"type": "number"},
    },
    "required": list(EMPTY_STATE),
    "additionalProperties": False,
}


class GeminiParseError(RuntimeError):
    """Raised when the optional Gemini parse cannot be used."""


class GeminiShoppingParser:
    def __init__(self, model: str | None = None) -> None:
        if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            raise GeminiParseError("GEMINI_API_KEY is not set")
        try:
            from google import genai
        except ImportError as exc:
            raise GeminiParseError(
                "Optional package missing; run: python3 -m pip install -r requirements-optional.txt"
            ) from exc
        self.client = genai.Client()
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.7-flash")

    def parse(self, message: str, current_state: dict | None = None) -> dict:
        state = normalise_state(current_state)
        prompt = f"""
You are the understanding layer of a shopping assistant.

Update the COMPLETE shopping state from the customer's newest message.
- Preserve requirements that the customer did not change.
- Replace a slot when the customer changes their mind.
- Set a slot to null when the customer explicitly removes that preference.
- Never invent a brand, budget, product category, or preference.
- Put requirements that do not fit a named slot in features.
- Ask at most one concise clarification question that would most improve retrieval.
- Return only data matching the supplied JSON schema.

Current state:
{json.dumps(state, ensure_ascii=False)}

Newest customer message:
{message}
""".strip()
        try:
            interaction = self.client.interactions.create(
                model=self.model,
                input=prompt,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": STATE_SCHEMA,
                },
            )
            return normalise_state(json.loads(interaction.output_text))
        except Exception as exc:
            raise GeminiParseError(f"Gemini parsing failed: {exc}") from exc


def normalise_state(value: dict | None) -> dict:
    """Return a validated, JSON-safe state with no unexpected keys."""
    result = deepcopy(EMPTY_STATE)
    if not isinstance(value, dict):
        return result
    for key in result:
        if key in value:
            result[key] = value[key]
    result["intent"] = result["intent"] if result["intent"] in {"buying", "browsing"} else "browsing"
    result["features"] = [str(item) for item in (result["features"] or []) if str(item).strip()]
    try:
        result["budget_max"] = None if result["budget_max"] is None else float(result["budget_max"])
    except (TypeError, ValueError):
        result["budget_max"] = None
    try:
        result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))
    except (TypeError, ValueError):
        result["confidence"] = 0.0
    return result


def state_constraints(state: dict) -> list[str]:
    """Translate typed slots into the existing lexical ranker's input."""
    constraints: list[str] = []
    for key in ("brand", "color", "material", "size", "style", "use_case"):
        value = state.get(key)
        if value:
            constraints.append(f"{key.replace('_', ' ')}: {value}")
    constraints.extend(str(item) for item in state.get("features", []) if item)
    if state.get("budget_max") is not None:
        constraints.append(f"budget around ${state['budget_max']}")
    return constraints
