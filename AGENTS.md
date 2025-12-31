# Repository Guidelines

## Project Structure & Module Organization

- `main.py`: primary entrypoint (CLI flags control wake word, tools, LLM, STT/TTS engines).
- `audio/`: audio pipeline pieces (VAD, STT, TTS, device helpers).
- `llm/`: LLM backends (`ollama`, `litellm`).
- `tools/`: optional internet-connected tools (web search/scrape, weather); see `tools/README.md`.
- `tests/`: automated tests (`tests/unit/` and `tests/integration/`).
- `personas/` + `voices/`: persona prompts and voice samples used by some TTS modes.
- `start_bot.sh`, `stop_bot.sh`, `status_bot.sh`: run the assistant as a background process and tail logs.

## Build, Test, and Development Commands

- Create env: `python3 -m venv .venv && source .venv/bin/activate` (Python version pinned in `.python-version`).
- Install deps (CPU): `pip install -r requirements.txt`
- Install deps (GPU extras): `pip install -r requirements_gpu.txt` (read notes inside for CUDA/PyTorch order).
- Configure: `cp env.example .env` and edit keys (e.g., `OPENWEATHERMAP_API_KEY`, `FIRECRAWL_*`, `LITELLM_API_KEY`).
- Run locally: `python main.py` (examples: `python main.py --wakeword`, `python main.py --tools`, `python main.py --wakeword --llm litellm --tools`).
- Background run: `./start_bot.sh wakeword_tools_online --tts-engine piper --tts-gpu` (logs in `bot.log`).
- Audio helpers: `./set_audio_defaults.sh --auto` then `./check_audio_defaults.sh`
- Tests: `python -m pytest -q` (unit only: `python -m pytest -q tests/unit`).

## Coding Style & Naming Conventions

- Python: 4-space indentation, PEP 8 naming (`snake_case` for functions/files, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants).
- Keep CLI flags backward-compatible when possible (see the “Equivalent to old scripts” section in `main.py`).

## Testing Guidelines

- Prefer adding fast, deterministic unit tests under `tests/unit/`.
- Put hardware- or network-dependent coverage in `tests/integration/` and document prerequisites in the test docstring.
- Test file naming: `test_*.py`; keep tests narrowly scoped to one behavior.

## Commit & Pull Request Guidelines

- Commits follow an imperative, verb-led summary style (e.g., “Add…”, “Update…”, “Refactor…”, “Enhance…”). Keep subjects short; use the body for context.
- PRs should include: what changed, how to run/verify (commands + expected output), and hardware/service assumptions (Jetson vs desktop, audio devices, API keys). Add screenshots/log snippets when changing UX or audio behavior.

## Security & Configuration Tips

- Never commit secrets in `.env`; use `env.example` for new configuration keys.
- Avoid adding large binary assets (models/audio) to Git unless necessary; prefer documenting download steps or using Git LFS.
