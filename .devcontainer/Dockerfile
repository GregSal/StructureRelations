FROM mcr.microsoft.com/vscode/devcontainers/anaconda:3

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libasound2 \
        libatk-bridge2.0-0 \
        libatk1.0-0 \
        libatspi2.0-0 \
        libcairo2 \
        libcups2 \
        libgbm1 \
        libnspr4 \
        libnss3 \
        libdbus-1-3 \
        libpango-1.0-0 \
        libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

USER vscode
