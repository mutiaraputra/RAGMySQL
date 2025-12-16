import logging
import sys
import os
from typing import Optional

import typer

from config.settings import settings, Settings, get_initialized_settings
from scripts.run_chatbot import run_chatbot
from scripts.run_scraper import run_scrape

app = typer.Typer(
    name="ragjson",
    help="RAG system with JSON endpoint, Pinecone vector database, and Google Gemini.",
)


@app.callback()
def main_callback(
    ctx: typer.Context,
    config: Optional[str] = typer.Option(None, "--config", help="Path to .env file"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging"),
):
    """Global options for CLI."""
    if config:
        try:
            new_settings = Settings(_env_file=config)
            from config.settings import settings_proxy
            settings_proxy.bind_real(new_settings)
        except Exception as e:
            typer.echo(f"Failed to load config from {config}: {e}", err=True)
            raise typer.Exit(1)

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        handlers=[logging.StreamHandler(sys.stderr)],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@app.command()
def scrape(
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without executing"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging")
):
    """Scrape data from JSON endpoint and ingest to Pinecone."""
    run_scrape(dry_run=dry_run, verbose=verbose)


@app.command()
def chat(
    top_k: int = typer.Option(5, help="Number of results to retrieve"),
    model: Optional[str] = typer.Option(None, help="Override Gemini model"),
    no_history: bool = typer.Option(False, help="Disable conversation memory"),
):
    """Run interactive RAG chatbot with Google Gemini."""
    run_chatbot(top_k=top_k, model=model, no_history=no_history)


@app.command()
def serve():
    """Run FastAPI server for Cloud Run deployment."""
    import uvicorn
    from api import app as fastapi_app
    
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


@app.command()
def version():
    """Show application version."""
    typer.echo("RAG JSON-Pinecone-Gemini System v1.0.0")


if __name__ == "__main__":
    app()