import pytest
from unittest.mock import patch, MagicMock

from src.scraper.mysql_scraper import MySQLScraper


class TestMySQLScraper:
    """Unit tests for MySQLScraper class."""

    @patch('mysql.connector.connect')
    def test_scraper_initialization(self, mock_connect, test_settings):
        """Test scraper initialization and connection setup."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        scraper = MySQLScraper(test_settings.mysql)

        mock_connect.assert_called_once_with(
            host=test_settings.mysql.host,
            port=test_settings.mysql.port,
            user=test_settings.mysql.user,
            password=test_settings.mysql.password,
            database=test_settings.mysql.database
        )
        assert scraper.connection == mock_conn

    @patch('mysql.connector.connect')
    def test_get_tables(self, mock_connect, test_settings):
        """Test retrieving list of tables from database."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [('users',), ('posts',), ('comments',)]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        scraper = MySQLScraper(test_settings.mysql)
        tables = scraper.get_tables()

        assert tables == ['users', 'posts', 'comments']
        mock_cursor.execute.assert_called_once_with("SHOW TABLES")
        mock_cursor.close.assert_called_once()

    @patch('mysql.connector.connect')
    def test_scrape_table(self, mock_connect, test_settings):
        """Test scraping data from a table with mocked database."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        scraper = MySQLScraper(test_settings.mysql)

        with patch.object(scraper, 'get_row_count', return_value=2):
            with patch.object(scraper, 'get_table_schema', return_value=[
                {'Field': 'id'}, {'Field': 'title'}, {'Field': 'content'}
            ]):
                mock_cursor = MagicMock()
                # Simulate two batches with batch_size=1
                mock_cursor.fetchall.side_effect = [
                    [{'id': 1, 'title': 'Title 1', 'content': 'Content 1'}],
                    [{'id': 2, 'title': 'Title 2', 'content': 'Content 2'}]
                ]
                mock_conn.cursor.return_value = mock_cursor

                batches = list(scraper.scrape_table('articles', ['title', 'content'], 'id', batch_size=1))

                assert len(batches) == 2
                assert len(batches[0]) == 1
                assert batches[0][0]['id'] == 1
                assert batches[0][0]['source_table'] == 'articles'
                assert batches[0][0]['content'] == 'Title 1 Content 1'
                assert batches[0][0]['metadata'] == {'id': 1, 'title': 'Title 1', 'content': 'Content 1'}

                assert len(batches[1]) == 1
                assert batches[1][0]['id'] == 2
                assert batches[1][0]['content'] == 'Title 2 Content 2'

    @patch('mysql.connector.connect')
    def test_scrape_table_batching(self, mock_connect, test_settings):
        """Test batching logic in scrape_table."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        scraper = MySQLScraper(test_settings.mysql)

        with patch.object(scraper, 'get_row_count', return_value=3):
            with patch.object(scraper, 'get_table_schema', return_value=[
                {'Field': 'id'}, {'Field': 'text'}
            ]):
                mock_cursor = MagicMock()
                # Simulate one batch with batch_size=3
                mock_cursor.fetchall.return_value = [
                    {'id': 1, 'text': 'Text 1'},
                    {'id': 2, 'text': 'Text 2'},
                    {'id': 3, 'text': 'Text 3'}
                ]
                mock_conn.cursor.return_value = mock_cursor

                batches = list(scraper.scrape_table('posts', ['text'], 'id', batch_size=3))

                assert len(batches) == 1
                assert len(batches[0]) == 3
                assert all(record['source_table'] == 'posts' for record in batches[0])
                assert [record['content'] for record in batches[0]] == ['Text 1', 'Text 2', 'Text 3']

    @patch('mysql.connector.connect')
    def test_error_handling(self, mock_connect, test_settings):
        """Test error handling for connection failures and query errors."""
        # Test connection failure
        mock_connect.side_effect = Exception("Connection failed")
        with pytest.raises(Exception, match="Connection failed"):
            MySQLScraper(test_settings.mysql)

        # Test query error in get_tables
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Query failed")
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        scraper = MySQLScraper(test_settings.mysql)
        with pytest.raises(Exception, match="Query failed"):
            scraper.get_tables()

    @patch('mysql.connector.connect')
    def test_context_manager(self, mock_connect, test_settings):
        """Test context manager for proper resource cleanup."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        with MySQLScraper(test_settings.mysql) as scraper:
            assert scraper.connection == mock_conn

        mock_conn.close.assert_called_once()

    @patch('mysql.connector.connect')
    def test_edge_cases_empty_table(self, mock_connect, test_settings):
        """Test scraping from an empty table."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        scraper = MySQLScraper(test_settings.mysql)

        with patch.object(scraper, 'get_row_count', return_value=0):
            with patch.object(scraper, 'get_table_schema', return_value=[
                {'Field': 'id'}, {'Field': 'text'}
            ]):
                batches = list(scraper.scrape_table('empty_table', ['text'], 'id', batch_size=10))
                assert batches == []

    @patch('mysql.connector.connect')
    def test_edge_cases_null_values(self, mock_connect, test_settings):
        """Test handling of NULL values in text columns."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        scraper = MySQLScraper(test_settings.mysql)

        with patch.object(scraper, 'get_row_count', return_value=1):
            with patch.object(scraper, 'get_table_schema', return_value=[
                {'Field': 'id'}, {'Field': 'title'}, {'Field': 'content'}
            ]):
                mock_cursor = MagicMock()
                mock_cursor.fetchall.return_value = [
                    {'id': 1, 'title': None, 'content': 'Some content'}
                ]
                mock_conn.cursor.return_value = mock_cursor

                batches = list(scraper.scrape_table('articles', ['title', 'content'], 'id', batch_size=10))

                assert len(batches) == 1
                assert batches[0][0]['content'] == 'Some content'  # NULL title skipped

    @patch('mysql.connector.connect')
    def test_edge_cases_special_characters(self, mock_connect, test_settings):
        """Test handling of special characters in content."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        scraper = MySQLScraper(test_settings.mysql)

        with patch.object(scraper, 'get_row_count', return_value=1):
            with patch.object(scraper, 'get_table_schema', return_value=[
                {'Field': 'id'}, {'Field': 'text'}
            ]):
                mock_cursor = MagicMock()
                mock_cursor.fetchall.return_value = [
                    {'id': 1, 'text': 'Special chars: éñüñ@#$%^&*()'}
                ]
                mock_conn.cursor.return_value = mock_cursor

                batches = list(scraper.scrape_table('posts', ['text'], 'id', batch_size=10))

                assert len(batches) == 1
                assert batches[0][0]['content'] == 'Special chars: éñüñ@#$%^&*()'