FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /srv/partfidelity

RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn>=0.30" "pillow>=10"

COPY app/ app/
COPY config/film.toml config/retention.toml config/exports.toml config/display_images.toml config/

RUN useradd --system --no-create-home appuser \
    && mkdir -p /srv/partfidelity/var \
    && chown -R appuser /srv/partfidelity

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "--factory", "app.main:create_app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
