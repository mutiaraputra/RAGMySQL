import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import typer

from config.settings import get_initialized_settings
from src.pipeline.ingestion import IngestionPipeline

app = typer.Typer()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_table_config_from_file(config_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load table configuration from JSON file.

    Returns a mapping of table_name -> {"text_columns": [...], "id_column": "..."}
    """
    with open(config_path, 'r') as f:
        data = json.load(f)
    table_config: Dict[str, Dict[str, Any]] = {}
    for table in data.get('tables', []):
        table_name = table['name']
        text_columns = table.get('text_columns', [])
        id_column = table.get('id_column', 'id')
        table_config[table_name] = {"text_columns": text_columns, "id_column": id_column}
    return table_config


def build_table_config(tables: Optional[str], all_tables: bool, config_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Build table configuration dict.

    Returns a mapping of table_name -> {"text_columns": [...], "id_column": "..."}
    """
    pipeline = IngestionPipeline()
    table_config = {}

    if config_path:
        table_config = load_table_config_from_file(config_path)
        if tables:
            # Filter to specified tables
            specified = set(tables.split(','))
            table_config = {k: v for k, v in table_config.items() if k in specified}
        elif not all_tables:
            # If config provided but no --tables or --all, use all from config
            pass
    else:
        # No config file, determine tables and text_columns from schema
            all_table_names = pipeline.scraper.get_tables()
            if tables:
                table_names = [t.strip() for t in tables.split(',')]
            elif all_tables:
                table_names = all_table_names
            else:
                typer.echo("Error: Must specify --tables, --all, or --config")
                raise typer.Exit(1)

            for table_name in table_names:
                if table_name not in all_table_names:
                    typer.echo(f"Warning: Table '{table_name}' not found in database")
                    continue
                schema = pipeline.scraper.get_table_schema(table_name)
                text_columns = [col['Field'] for col in schema if col['Type'].lower().startswith(('varchar', 'text', 'mediumtext', 'longtext'))]
                if not text_columns:
                    typer.echo(f"Warning: No text columns found for table '{table_name}'")
                    continue
                # Default id_column to 'id' when not specified in config
                table_config[table_name] = {"text_columns": text_columns, "id_column": 'id'}

    return table_config


@app.command()
def scrape(
    tables: Optional[str] = typer.Option(None, "--tables", help="Comma-separated list of table names to scrape"),
    all_tables: bool = typer.Option(False, "--all", help="Scrape all tables"),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to table config JSON file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate configuration without executing"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging")
):
    """Run the MySQL scraping pipeline."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_scrape(tables=tables, all_tables=all_tables, config=config, dry_run=dry_run, verbose=verbose)


def run_scrape(tables: Optional[str] = None, all_tables: bool = False, config: Optional[Path] = None, dry_run: bool = False, verbose: bool = False):
    """Callable function to run the scraping pipeline; can be imported and registered by `main.py`."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load settings
        typer.echo("Loading settings...")
        settings = get_initialized_settings()

        # Build table config
        typer.echo("Building table configuration...")
        table_config = build_table_config(tables, all_tables, config)
        if not table_config:
            typer.echo("No tables to process")
            return

        typer.echo(f"Tables to process: {list(table_config.keys())}")

        # Initialize pipeline
        typer.echo("Initializing ingestion pipeline...")
        pipeline = IngestionPipeline()

        if dry_run:
            typer.echo("Dry run: Validating pipeline...")
            typer.echo(f"JSON Endpoint: {settings.json_endpoint.url}")
            typer.echo("✓ Validation successful")
            return

        # Run pipeline
        typer.echo("Starting ingestion from JSON endpoint to Pinecone...")
        stats = pipeline.run_full_pipeline()

        # Print summary
        typer.echo("\n✓ Pipeline completed successfully!")
        typer.echo(f"Total documents: {stats['total_documents']}")
        typer.echo(f"Total chunks: {stats['total_chunks']}")
        typer.echo(f"Total embeddings: {stats['total_embeddings']}")
        typer.echo(f"Processing time: {stats['processing_time']:.2f}s")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        typer.echo(f"✗ Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == '__main__':
    app()