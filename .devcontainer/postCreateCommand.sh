#! /usr/bin/env bash

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

sudo chown -v -R vscode:vscode ~/.cache

# Install Dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install --install-hooks
