# Use the official uv base image with Python 3.11
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Set working directory inside the container
WORKDIR /app

# Set environment variables for uv behavior
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy project files into the container
ADD . /app

# Create and activate a virtual environment using uv
RUN uv venv \
 && uv pip install -e .

# Expose the port your app listens on
EXPOSE 8080

# Run the application via uv script execution
CMD ["uv", "run", "nexonco", "--transport", "sse"]