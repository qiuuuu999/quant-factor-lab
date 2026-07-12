FROM python:3.12-slim

# yfinance/pandas need certificate authorities for HTTPS downloads.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (before copying source) so `docker build` only
# re-installs packages when pyproject.toml actually changes.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

RUN chmod +x docker/reproduce.sh

ENTRYPOINT ["docker/reproduce.sh"]
