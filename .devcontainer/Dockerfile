# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# ---------------------------------------------------------------------------
# System tools: Fortran compiler, f2c transpiler, f2c runtime library
#
# The base image ships a Yarn apt source whose GPG key is no longer trusted.
# Remove it before running apt-get update to prevent the build from failing.
# ---------------------------------------------------------------------------
RUN find /etc/apt/sources.list.d/ -name "*yarn*" -delete 2>/dev/null; \
    apt-get update && apt-get install -y --no-install-recommends \
        gfortran \
        f2c \
        libf2c2 \
        libf2c2-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# GitHub CLI (gh) — used for `gh auth login` and token discovery
# ---------------------------------------------------------------------------
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Rust toolchain (stable) — installed as the non-root "vscode" user so
# cargo/rustup live in /home/vscode/.cargo and are on PATH automatically.
# ---------------------------------------------------------------------------
USER vscode
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --component clippy,rustfmt
ENV PATH="/home/vscode/.cargo/bin:${PATH}"

# ---------------------------------------------------------------------------
# Python virtual environment and dependencies
# ---------------------------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /home/vscode/.venv \
    && /home/vscode/.venv/bin/pip install --upgrade pip --no-cache-dir \
    && /home/vscode/.venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

ENV VIRTUAL_ENV="/home/vscode/.venv"
ENV PATH="/home/vscode/.venv/bin:${PATH}"

USER root
