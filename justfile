run:
    docker build -t bayes-xai .
    docker run -p 8000:8000 bayes-xai

down:
    docker stop $(docker ps -q --filter ancestor=bayes-xai) 2>/dev/null || true
    docker rm $(docker ps -aq --filter ancestor=bayes-xai) 2>/dev/null || true

lint:
    uv run ruff check .

format:
    uv run ruff format .

check:
    uv run ruff check .
    uv run ruff format --check .
