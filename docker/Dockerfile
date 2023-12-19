FROM python:3.11-buster

WORKDIR /app

COPY pyproject.toml .
COPY poetry.lock .

RUN pip install poetry
RUN poetry install

RUN poetry run python -m spacy download en_core_web_sm

# The code to run when container is started:
COPY src/ga_prompt_llm .
ENTRYPOINT [ "poetry", "run", "docker/entrypoint.sh" ]