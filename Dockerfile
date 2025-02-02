FROM python:3.12-slim

ARG POETRY_VERSION=1.8.2

# Set working directory
WORKDIR /code

RUN apt update \
    && apt install wget -y \
    && apt install openssh-server -y \
    && mkdir -p ~/.ssh \
    && chmod 700 ~/.ssh \
    && echo "$PUBLIC_KEY" >> authorized_keys \
    && chmod 700 authorized_keys \
    && service ssh start

COPY pyproject.toml poetry.lock README.md ./

COPY ai_wise_council/ ./ai_wise_council

RUN pip3 install --no-cache-dir poetry==${POETRY_VERSION} \ 
    && poetry env use 3.12 \
    && poetry config virtualenvs.create false

# main and finetune
RUN poetry install --with finetune --no-interaction --no-ansi

COPY data/ ./data

CMD ["python", "ai_wise_council/finetuning.py"]
