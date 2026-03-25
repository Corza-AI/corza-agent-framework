# Contributing to Corza Agent Framework

## Getting Started

```bash
git clone https://github.com/corza-ai/corza-agent-framework.git
cd corza-agent-framework
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,sqlite]"
```

## Running Tests

```bash
pytest                    # all tests
pytest tests/ -v          # verbose
pytest tests/ -k "memory" # filter by name
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check src/
ruff format src/
```

## Making Changes

1. Create a branch from `main`
2. Make your changes
3. Add or update tests in `tests/`
4. Run `pytest` and `ruff check` — both must pass
5. Open a pull request

## Project Layout

```
src/corza_agents/
    core/          Engine, LLM, types, errors
    tools/         @tool decorator, registry
    orchestrator/  Multi-agent orchestration
    skills/        Skill loading and injection
    memory/        Working memory, context management
    middleware/     Audit, token tracking, permissions
    persistence/   Multi-backend: memory, sqlite, postgres
    streaming/     Event definitions
    prompts/       System prompt construction
    api/           Service layer + FastAPI router
```

## Adding a New Persistence Backend

1. Create `persistence/mybackend.py`
2. Implement all methods from `BaseRepository` (see `persistence/base.py`)
3. Add it to `persistence/factory.py`
4. Add tests in `tests/test_persistence.py`
5. Update `persistence/__init__.py` exports

## Adding a New LLM Provider

1. Add the streaming and completion methods in `core/llm.py`
2. Add message/tool format converters
3. Add the provider to `_get_client()` client initialization
4. Add optional dependency in `pyproject.toml`

## Reporting Issues

Open an issue at https://github.com/corza-ai/corza-agent-framework/issues.
