#!/usr/bin/env python3
"""
Configuration file for VortexBench evaluation
"""

import os

# Azure OpenAI Configuration
# Users should set these environment variables or modify this file
AZURE_API_KEY = os.getenv(
    "AZURE_OPENAI_API_KEY", 
    ""
)

AZURE_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://dil-nu-openai-eastus.openai.azure.com"
)

AZURE_DEPLOYMENT_NAME = os.getenv(
    "AZURE_OPENAI_DEPLOYMENT",
    "gpt-4o"
)

AZURE_API_VERSION = os.getenv(
    "AZURE_OPENAI_API_VERSION", 
    "2024-08-01-preview"
)

# Instructions for users
def print_config_help():
    """Print help for configuring Azure OpenAI"""
    print("""
=== Azure OpenAI Configuration ===

Option 1 - Environment Variables (Recommended):
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"

Option 2 - Edit config.py:
Modify the default values in /code/qwen/eval/gen/vortex/config.py

Current Configuration:
- API Key: {'Set' if AZURE_API_KEY else 'Not Set'}
- Endpoint: {AZURE_ENDPOINT}
- Deployment: {AZURE_DEPLOYMENT}
    """)

if __name__ == "__main__":
    print_config_help()
