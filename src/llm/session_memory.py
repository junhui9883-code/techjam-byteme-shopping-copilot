"""Recipient-aware short-term contexts plus a compact long-term profile."""

from __future__ import annotations

from copy import deepcopy

from .gemini_parser import EMPTY_STATE, normalise_state


def context_key(recipient: object, category: object) -> str:
    who = str(recipient or "self").strip().lower()
    product = str(category or "unknown").strip().lower()
    return f"{who}|{product}"


class SessionMemory:
    """Keep multiple shopping goals without mixing recipients or categories."""

    def __init__(self, user_profile: dict | None = None) -> None:
        self.contexts: dict[str, dict] = {}
        self.context_messages: dict[str, list[str]] = {}
        self.active_key: str | None = None
        self.user_profile = deepcopy(user_profile or {})
        self.user_profile.setdefault("stable_preferences", [])

    def active_state(self) -> dict:
        if self.active_key is None:
            return deepcopy(EMPTY_STATE)
        return deepcopy(self.contexts[self.active_key])

    def active_transcript(self) -> list[str]:
        if self.active_key is None:
            return []
        return list(self.context_messages.get(self.active_key, []))

    def summaries(self) -> list[dict]:
        return [deepcopy(state) for state in self.contexts.values()]

    def apply(self, parsed_state: dict, message: str) -> dict:
        state = normalise_state(parsed_state)
        key = context_key(state["recipient"], state["category"])

        preferences = self.user_profile["stable_preferences"]
        removals = {item.casefold() for item in state["profile_removals"]}
        preferences[:] = [item for item in preferences if str(item).casefold() not in removals]
        known = {str(item).casefold() for item in preferences}
        for item in state["profile_updates"]:
            if item.casefold() not in known:
                preferences.append(item)
                known.add(item.casefold())

        # Profile mutations belong to memory, not to a shopping context.
        state["profile_updates"] = []
        state["profile_removals"] = []
        self.contexts[key] = deepcopy(state)
        self.context_messages.setdefault(key, []).append(message)
        self.active_key = key
        return deepcopy(state)

    def new_request(self) -> None:
        """Clear short-term goals while retaining stable user preferences."""
        self.contexts.clear()
        self.context_messages.clear()
        self.active_key = None

    def inspect(self) -> dict:
        return {
            "active_context": self.active_key,
            "contexts": deepcopy(self.contexts),
            "long_term_profile": deepcopy(self.user_profile),
        }
