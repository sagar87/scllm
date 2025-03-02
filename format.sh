uv run black src
uv run black tests
uv run isort src --profile black
uv run isort tests --profile black
uv run flake8 src
uv run pytest tests
# uv run bandit src
# uv run safety check