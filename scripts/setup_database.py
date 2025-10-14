import logging
import sys
from typing import Optional

import click
import mysql.connector
from mysql.connector import Error

from config.settings import settings
from src.vectorstore.tidb_store import TiDBVectorStore

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def perform_setup(drop: bool, verify_only: bool):
    """Perform TiDB database setup logic.

    This function contains the core setup logic and can be called from the Click command
    or from other CLI entry points (e.g., `main.py`).
    """
    try:
        tidb_config = settings.tidb
        app_config = settings.app

        # Step 1: Connect to TiDB server (without specifying database) to create database if needed
        if not verify_only:
            try:
                connection = mysql.connector.connect(
                    host=tidb_config.host,
                    port=tidb_config.port,
                    user=tidb_config.user,
                    password=tidb_config.password,
                    ssl_disabled=not tidb_config.use_tls
                )
                cursor = connection.cursor()
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {tidb_config.database}")
                cursor.close()
                connection.close()
                logger.info(f"Database '{tidb_config.database}' created or already exists")
            except Error as e:
                logger.error(f"Failed to create database: {e}")
                sys.exit(1)

        # Step 2: Initialize TiDBVectorStore and verify connection
        vector_store = TiDBVectorStore(
            tidb_config=tidb_config,
            table_name=app_config.vector_table_name,
            embedding_dim=app_config.embedding_dimension,
            distance_metric=app_config.distance_metric
        )

        # Verify connectivity to TiDB server with a lightweight probe (do not require table to exist)
        try:
            with vector_store.client.cursor() as cursor:
                cursor.execute("SELECT 1")
                _ = cursor.fetchone()
            logger.info("Connected to TiDB server successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to TiDB server: {e}")
            sys.exit(1)

        if verify_only:
            # Just verify connectivity and exit
            logger.info("Verification complete. TiDB server is reachable.")
            return

        # Step 3: Handle drop if requested
        if drop:
            if click.confirm(f"Are you sure you want to drop the existing table '{app_config.vector_table_name}'?"):
                try:
                    # Drop table manually since TiDBVectorStore doesn't have a drop method
                    with vector_store.client.cursor() as cursor:
                        cursor.execute(f"DROP TABLE IF EXISTS {app_config.vector_table_name}")
                    logger.info(f"Table '{app_config.vector_table_name}' dropped successfully")
                except Exception as e:
                    logger.error(f"Failed to drop table: {e}")
                    sys.exit(1)
            else:
                logger.info("Drop cancelled by user")
                return

        # Step 4: Create table
        try:
            vector_store.create_table()
            logger.info(f"Table '{app_config.vector_table_name}' created successfully")
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            sys.exit(1)

        # Step 5: Create vector index (handled in create_table for pytidb)
        try:
            vector_store.create_vector_index()
            logger.info("Vector index created successfully")
        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
            sys.exit(1)

        # Step 6: Verify table schema and index
        try:
            with vector_store.client.cursor() as cursor:
                # Check table schema
                cursor.execute(f"DESCRIBE {app_config.vector_table_name}")
                columns = cursor.fetchall()
                logger.info(f"Table schema verified. Columns: {[col[0] for col in columns]}")
                
                # Check for vector index (pytidb creates it automatically)
                cursor.execute(f"SHOW INDEX FROM {app_config.vector_table_name}")
                indexes = cursor.fetchall()
                vector_indexes = [idx for idx in indexes if 'embedding' in idx[4]]  # idx[4] is Column_name
                if vector_indexes:
                    logger.info(f"Vector index verified: {len(vector_indexes)} index(es) on embedding column")
                else:
                    logger.warning("No vector index found on embedding column")
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {app_config.vector_table_name}")
                count = cursor.fetchone()[0]
                logger.info(f"Table information: {count} rows")
        except Exception as e:
            logger.error(f"Failed to verify table schema/index: {e}")
            sys.exit(1)

        logger.info("TiDB database setup completed successfully!")

    except Exception as e:
        logger.error(f"Unexpected error during setup: {e}")
        sys.exit(1)
    finally:
        # Ensure connection is closed
        if 'vector_store' in locals():
            vector_store.client.close()


@click.command()
@click.option('--drop', is_flag=True, help='Drop existing table before creating (requires confirmation)')
@click.option('--verify-only', is_flag=True, help='Only verify existing setup without creating anything')
def setup_database(drop: bool, verify_only: bool):
    """Click command wrapper that calls perform_setup.

    This keeps the script executable as a standalone CLI while exposing `perform_setup`
    for reuse by `main.py`.
    """
    perform_setup(drop, verify_only)


if __name__ == '__main__':
    setup_database()