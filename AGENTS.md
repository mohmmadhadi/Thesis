# Instructions for Codex

This is a notebook-based data science project being refactored into an OOP Python package.

Rules:
- Do not refactor the whole project at once.
- Preserve the original notebook behavior.
- Make small, reviewable changes.
- Before editing, explain which files will be modified.
- Do not change formulas, thresholds, model names, or column names unless explicitly asked.
- Do not hardcode absolute local paths.
- Put reusable logic in src/social_analysis/.
- Keep notebooks as report/exploration wrappers.
- Add type hints and short docstrings.
- Add lightweight tests for deterministic logic.
- Do not test heavy transformer models unless explicitly asked.
- Prefer config values from configs/default.yaml.