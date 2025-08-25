# LangChain-MCP

LangChain-MCP (Multi-Context Processor) is a versatile framework built on LangChain, designed to facilitate advanced language model interactions by building and leveraging context from diverse sources. It integrates with Ollama for local LLM inference and provides capabilities for Retrieval Augmented Generation (RAG), summarization, and agent-based workflows.

## Features

*   **Multi-Source Context Building**: Seamlessly build context from:
    *   **PDF Documents**: Extract and process content from PDF files.
    *   **Web Pages**: Scrape and parse information from URLs.
    *   **Local Directories**: Read and index content from local file systems.
*   **Retrieval Augmented Generation (RAG)**: Enhance LLM responses by retrieving relevant information from built contexts.
*   **Summarization**: Generate concise summaries of large documents or web content.
*   **Agent Framework**: Develop and deploy intelligent agents capable of performing complex tasks.
*   **Ollama Integration**: Utilize local Large Language Models (LLMs) via Ollama for privacy-preserving and efficient inference.
*   **Command-Line Interface (CLI)**: Interact with the framework's functionalities through a user-friendly command-line interface.
*   **API Server**: Expose core functionalities via a FastAPI-based server for programmatic access and integration.

## Installation

To get started with LangChain-MCP, follow these steps:

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/KiranSilwal/langchain-mcp.git
    cd langchain-mcp
    ```

2.  **Install dependencies**:

    It is recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -e .
    ```

    Alternatively, you can install directly from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Command-Line Interface (CLI)

The primary way to interact with LangChain-MCP is through its CLI. You can explore available commands and their options using the `--help` flag:

```bash
python -m langchain_mcp.cli.main --help
```

Example usage for building context (specific commands will vary based on implementation):

```bash
# Example: Build context from a directory
python -m langchain_mcp.cli.main build-context directory --path ./data/my_documents

# Example: Summarize a web page
python -m langchain_mcp.cli.main summarize web --url https://example.com/article
```

### API Server

To run the FastAPI server and access the functionalities via an API:

```bash
uvicorn langchain_mcp.server.server:app --reload
```

The API documentation (Swagger UI) will typically be available at `http://127.0.0.1:8000/docs` after the server starts.

## License

This project is developed by Kiran Silwal. All users have full license to use, modify, and distribute this software.
