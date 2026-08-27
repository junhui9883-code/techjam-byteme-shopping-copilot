"""Compatibility shim.

evaluator/local_evaluator.py hardcodes `from starter.agent import Agent` and we
are forbidden from modifying anything under evaluator/, so this path has to
keep resolving. The real implementation is agent.py at the repository root.

Do not add logic here.
"""

from agent import Agent

__all__ = ["Agent"]
