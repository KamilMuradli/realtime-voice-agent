# -*- coding: utf-8 -*-
"""
================================================================================
prompt_config.py - Centralized Prompt Configuration
================================================================================
"""

# =============================================================================
# DEFAULT (General Assistant)
# =============================================================================
DEFAULT_PROMPT = """Sən Azərbaycan dilində danışan səsli köməkçisən.
Qısa, konkret və faydalı cavablar ver.
Cavabların danışıq üçün uyğun olmalıdır - çox uzun cümlələrdən qaç.
Hər cavab 2-3 cümlədən çox olmamalıdır."""

# =============================================================================
# SCENARIO REGISTRY
# =============================================================================

SCENARIOS = {
    "default": {"prompt": DEFAULT_PROMPT, "temp": 0.7, "first_turn": None},
}

def get_scenario_config(scenario_name: str) -> dict:
    return SCENARIOS.get(scenario_name, SCENARIOS["default"])
