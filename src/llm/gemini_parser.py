"""Optional Gemini parser for free-form shopping conversations."""

from __future__ import annotations

import json
import os
from copy import deepcopy


EMPTY_STATE = {
    "recipient": "self",
    "context_action": "continue",
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
    "profile_updates": [],
    "profile_removals": [],
}


STATE_SCHEMA = {
    "type": "object",
    "properties": {
        "recipient": {
            "type": "string",
            "description": "Who the product is for. Use self for the customer.",
        },
        "context_action": {
            "type": "string",
            "enum": ["continue", "new", "switch", "resume"],
            "description": "How this turn relates to the stored shopping contexts.",
        },
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
        "profile_updates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Stable self preferences stated with words such as always or usually.",
        },
        "profile_removals": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Stable self preferences the customer explicitly retracts.",
        },
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
        self.client = genai.Client(
            http_options={
                "timeout": 15_000,
                "retry_options": {"attempts": 1},
            }
        )
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    def parse(
        self,
        message: str,
        current_state: dict | None = None,
        saved_contexts: list[dict] | None = None,
        user_profile: dict | None = None,
    ) -> dict:
        state = normalise_state(current_state)
        contexts = saved_contexts or []
        profile = user_profile or {"stable_preferences": []}
        prompt = f"""
You are the understanding layer of a shopping assistant.

Select the relevant shopping context and return its COMPLETE updated state.
- A context is identified by recipient plus product category.
- Use recipient "self" for the customer; use a short relationship such as "brother" otherwise.
- Use context_action="continue" for the active goal, "new" for a new goal,
  "switch" for another recipient/category, and "resume" when returning to a saved goal.
- When switching to a new recipient, do not copy the previous recipient's preferences.
- When resuming, restore the matching saved context and then apply the newest message.
- Preserve requirements that the customer did not change.
- Replace a slot when the customer changes their mind.
- Set a slot to null when the customer explicitly removes that preference.
- Never invent a brand, budget, product category, or preference.
- Put requirements that do not fit a named slot in features.
- Use the long-term profile only for the self recipient and only when relevant.
- Add profile_updates only for explicit stable statements such as "I always" or "I usually".
- Never add a one-off request or another recipient's preference to the long-term profile.
- Ask at most one concise clarification question that would most improve retrieval.
- Return only data matching the supplied JSON schema.

Current active state:
{json.dumps(state, ensure_ascii=False)}

Saved shopping contexts:
{json.dumps(contexts, ensure_ascii=False)}

Long-term user profile:
{json.dumps(profile, ensure_ascii=False)}

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
    recipient = str(result.get("recipient") or "self").strip().lower()
    result["recipient"] = "self" if recipient in {"me", "myself", "user", "customer"} else recipient
    if result["context_action"] not in {"continue", "new", "switch", "resume"}:
        result["context_action"] = "continue"
    result["intent"] = result["intent"] if result["intent"] in {"buying", "browsing"} else "browsing"
    result["features"] = [str(item) for item in (result["features"] or []) if str(item).strip()]
    result["profile_updates"] = [
        str(item).strip() for item in (result["profile_updates"] or []) if str(item).strip()
    ]
    result["profile_removals"] = [
        str(item).strip() for item in (result["profile_removals"] or []) if str(item).strip()
    ]
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
