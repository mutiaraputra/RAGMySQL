import logging
from typing import Dict, List, Generator, Any, Optional

import mysql.connector
from mysql.connector import Error as MySQLError

from config.settings import MySQLConfig

logger = logging.getLogger(__name__)


class MySQLScraper:
    """MySQL scraper for extracting data from MySQL databases with batching and memory efficiency."""

    def __init__(self, mysql_config: MySQLConfig):
        """Initialize the scraper with MySQL configuration and establish connection."""
        self.config = mysql_config
        self.connection = None
        try:
            self.connection = mysql.connector.connect(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                database=self.config.database
            )
            logger.info("Successfully connected to MySQL database.")
        except MySQLError as e:
            logger.error(f"Failed to connect to MySQL database: {e}")
            raise

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: close the connection."""
        if self.connection:
            self.connection.close()
            logger.info("MySQL connection closed.")

    def get_tables(self) -> List[str]:
        """List all tables in the database."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
            cursor.close()
            logger.info(f"Retrieved {len(tables)} tables from database.")
            return tables
        except MySQLError as e:
            logger.error(f"Error retrieving tables: {e}")
            raise

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Inspect columns and types for a given table."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"DESCRIBE {table_name}")
            schema = [{"Field": row[0], "Type": row[1], "Null": row[2], "Key": row[3], "Default": row[4], "Extra": row[5]} for row in cursor.fetchall()]
            cursor.close()
            logger.info(f"Retrieved schema for table '{table_name}' with {len(schema)} columns.")
            return schema
        except MySQLError as e:
            logger.error(f"Error retrieving schema for table '{table_name}': {e}")
            raise

    def get_row_count(self, table_name: str) -> int:
        """Get the total row count for a table for progress estimation."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            cursor.close()
            logger.info(f"Table '{table_name}' has {count} rows.")
            return count
        except MySQLError as e:
            logger.error(f"Error getting row count for table '{table_name}': {e}")
            raise

    def scrape_table(self, table_name: str, text_columns: List[str], id_column: str = "id", batch_size: int = 100) -> Generator[List[Dict[str, Any]], None, None]:
        """Scrape data from a table in batches, yielding lists of records.

        Each record is a dict with keys: id, source_table, content (concatenated text), metadata (dict of all columns).
        """
        try:
            # Fetch schema once per table for efficiency
            all_columns = [col["Field"] for col in self.get_table_schema(table_name)]
            select_columns = ", ".join(all_columns)

            last_id = None
            while True:
                cursor = self.connection.cursor(dictionary=True)
                if last_id is None:
                    # First batch: no WHERE clause
                    query = f"SELECT {select_columns} FROM {table_name} ORDER BY {id_column} LIMIT %s"
                    cursor.execute(query, (batch_size,))
                else:
                    # Keyset pagination: fetch rows with id > last_id
                    query = f"SELECT {select_columns} FROM {table_name} WHERE {id_column} > %s ORDER BY {id_column} LIMIT %s"
                    cursor.execute(query, (last_id, batch_size))

                rows = cursor.fetchall()
                cursor.close()

                if not rows:
                    break

                batch = []
                for row in rows:
                    # Concatenate text columns
                    content_parts = [str(row[col]) for col in text_columns if row.get(col) is not None]
                    content = " ".join(content_parts)
                    # Metadata as dict of all columns
                    metadata = {col: row[col] for col in all_columns}
                    record = {
                        "id": row[id_column],
                        "source_table": table_name,
                        "content": content,
                        "metadata": metadata
                    }
                    batch.append(record)

                logger.info(f"Scraped batch of {len(batch)} records from '{table_name}' (last_id {last_id}).")
                yield batch

                # Advance last_id to the last row's id to continue keyset pagination
                last_row = rows[-1]
                last_id = last_row[id_column]

        except MySQLError as e:
            logger.error(f"Error scraping table '{table_name}': {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing column in table '{table_name}': {e}")
            raise

    def scrape_all_tables(self, table_config: Dict[str, Dict[str, Any]], batch_size: int = 100) -> Generator[List[Dict[str, Any]], None, None]:
        """Scrape all tables specified in table_config.

        table_config: dict mapping table_name -> {"text_columns": [...], "id_column": "..."}.
        Yields batches from each table sequentially.
        """
        for table_name, cfg in table_config.items():
            text_columns = cfg.get('text_columns', [])
            id_column = cfg.get('id_column', 'id')
            logger.info(f"Starting scrape for table '{table_name}' with text columns: {text_columns} and id_column: {id_column}")
            yield from self.scrape_table(table_name, text_columns, id_column, batch_size)