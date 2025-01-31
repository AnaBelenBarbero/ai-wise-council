FROM python:3.12-slim

ARG POETRY_VERSION=1.8.2

# Set working directory
WORKDIR /code

COPY pyproject.toml poetry.lock README.md ./

COPY ai_wise_council/ ./ai_wise_council

RUN pip3 install --no-cache-dir poetry==${POETRY_VERSION} \ 
    && poetry env use 3.12

# main and finetune
RUN poetry install --with finetune --no-interaction --no-ansi

COPY data/ ./data

CMD ["poetry", "run", "python", "ai_wise_council/train.py"]