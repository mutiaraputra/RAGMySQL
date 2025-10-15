import logging
import sys
from typing import Optional
from pathlib import Path

import typer

import importlib
import config.settings as settings_module
from config.settings import settings, Settings, get_initialized_settings
from scripts.run_chatbot import run_chatbot
from scripts.run_scraper import run_scrape
from scripts.setup_database import perform_setup

app = typer.Typer(
    name="ragmysql",
    help="RAGMySQL: Retrieval Augmented Generation system using MySQL, TiDB Vector Database, and OpenRouter LLM API.",
    epilog="For more information, visit the project README."
)


@app.callback()
def main_callback(
    ctx: typer.Context,
    config: Optional[str] = typer.Option(None, "--config", help="Path to .env configuration file"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging"),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="Path to log file")
):
    """Global options for RAGMySQL CLI."""
    # Set custom config file if provided
    if config:
        # When a config file path is provided, instantiate Settings with that env file
        # to ensure nested Pydantic Settings pick up the provided .env values.
        try:
            # Create a new Settings instance reading the provided env file path
            new_settings = Settings(_env_file=config)

            # Replace the module-level settings in the config.settings module so future
            # attribute lookups reference the newly created Settings instance.
            # This also rebinds the local `settings` name used in this module.
            from config.settings import settings_proxy
            settings_proxy.bind_real(new_settings)
        except Exception as e:
            typer.echo(f"Failed to load configuration from {config}: {e}", err=True)
            raise typer.Exit(1)
 

    # Setup logging
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Basic validation
    try:
        actual_settings = get_initialized_settings()
        _ = actual_settings.mysql.host
        _ = actual_settings.tidb.host
        _ = actual_settings.openrouter.api_key
    except Exception as e:
        typer.echo(f"Configuration error: {e}", err=True)
        typer.echo("Please check your .env file or use --config to specify a configuration file.", err=True)
        raise typer.Exit(1)


@app.command()
def scrape(
    tables: Optional[str] = typer.Option(None, "--tables", help="Comma-separated list of table names to scrape"),
    all_tables: bool = typer.Option(False, "--all", help="Scrape all tables"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to table config JSON file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate configuration without executing"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging")
):
    """Run the MySQL scraping pipeline."""
    run_scrape(tables=tables, all_tables=all_tables, config=Path(config) if config else None, dry_run=dry_run, verbose=verbose)


@app.command()
def chat(
    top_k: int = typer.Option(5, help="Number of top documents to retrieve"),
    model: Optional[str] = typer.Option(None, help="Override OpenRouter model from config"),
    no_history: bool = typer.Option(False, help="Disable conversation memory"),
    stream: bool = typer.Option(False, help="Enable streaming responses (not yet implemented)"),
):
    """Run the interactive RAG chatbot with OpenRouter LLM support."""
    run_chatbot(top_k=top_k, model=model, no_history=no_history, stream=stream)


@app.command()
def setup(
    drop: bool = typer.Option(False, "--drop", help="Drop existing table before creating (requires confirmation)"),
    verify_only: bool = typer.Option(False, "--verify-only", help="Only verify existing setup without creating anything")
):
    """Setup TiDB database: create database, table, and vector index."""
    # Use shared perform_setup function from scripts.setup_database
    actual_settings = get_initialized_settings()
    perform_setup(drop, verify_only)


@app.command()
def version():
    """Show application version."""
    typer.echo("RAGMySQL v1.0.0")


@app.command()
def models():
    """Show current OpenRouter model and link to available models."""
    try:
        actual_settings = get_initialized_settings()
        current_model = actual_settings.openrouter.model
        typer.echo(f"Current OpenRouter model: {current_model}")
        typer.echo("For a full list of available models and pricing, visit: https://openrouter.ai/models")
        typer.echo("You can change the model by setting OPENROUTER_MODEL in your .env file.")
    except Exception as e:
        typer.echo(f"Error accessing model configuration: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()