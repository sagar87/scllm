uv run black src
uv run black tests
uv run black tools/annotate_notebook.py 
uv run isort src --profile black
uv run isort tests --profile black
uv run flake8 src
uv run flake8 tools
uv run pytest tests
# uv run bandit src
# uv run safety check