"""Prompt templates for VLM tag extraction.

This module re-exports all prompts from app.domain.prompts for backward compatibility.
"""

from app.domain.prompts import get_optimized_prompt, get_safe_prompt

__all__ = ["get_safe_prompt", "get_optimized_prompt"]
