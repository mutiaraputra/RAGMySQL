import logging
import requests
from typing import Dict, List, Generator, Any, Optional
from datetime import datetime

from config.settings import JSONEndpointConfig

logger = logging.getLogger(__name__)


class JSONScraper:
    """Scraper for extracting data from JSON API endpoints."""

    def __init__(self, config: JSONEndpointConfig):
        """Initialize scraper with JSON endpoint configuration."""
        self.config = config
        self.session = requests.Session()
        
        # Setup headers
        if self.config.headers:
            self.session.headers.update(self.config.headers)
        
        # Setup authentication
        if self.config.auth_token:
            self.session.headers.update({
                'Authorization': f'Bearer {self.config.auth_token}'
            })
        
        logger.info(f"Initialized JSONScraper for endpoint: {self.config.url}")

    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """Get value from nested dictionary using dot notation path."""
        keys = path.split('.')
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value

    def _extract_data_array(self, response_data: Dict) -> List[Dict]:
        """Extract data array from JSON response using configured path."""
        data_array = self._get_nested_value(response_data, self.config.data_path)
        
        if not isinstance(data_array, list):
            logger.warning(f"Data path '{self.config.data_path}' did not return a list. Wrapping in list.")
            data_array = [data_array] if data_array else []
        
        return data_array

    def fetch_data(self) -> Dict:
        """Fetch data from JSON endpoint."""
        try:
            logger.info(f"Fetching data from {self.config.url}")
            
            if self.config.method.upper() == "GET":
                response = self.session.get(
                    str(self.config.url),
                    timeout=self.config.timeout
                )
            elif self.config.method.upper() == "POST":
                response = self.session.post(
                    str(self.config.url),
                    timeout=self.config.timeout
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {self.config.method}")
            
            response.raise_for_status()
            data = response.json()
            
            logger.info("Successfully fetched and parsed JSON data")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch data from endpoint: {e}")
            raise
        except ValueError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise

    def scrape_data(self, batch_size: int = 100) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Scrape data from JSON endpoint and yield batches.
        
        Yields batches of documents with structure:
        {
            "id": <document_id>,
            "content": <concatenated_text>,
            "metadata": <dict_of_metadata>
        }
        """
        try:
            # Fetch data
            response_data = self.fetch_data()
            
            # Extract data array
            data_array = self._extract_data_array(response_data)
            logger.info(f"Extracted {len(data_array)} items from JSON response")
            
            # Process in batches
            batch = []
            for item in data_array:
                if not isinstance(item, dict):
                    logger.warning(f"Skipping non-dict item: {item}")
                    continue
                
                # Extract ID
                doc_id = item.get(self.config.id_field)
                if doc_id is None:
                    logger.warning(f"Item missing ID field '{self.config.id_field}', skipping")
                    continue
                
                # Concatenate content fields
                content_parts = []
                for field in self.config.content_fields:
                    value = item.get(field)
                    if value is not None:
                        content_parts.append(f"{field}: {str(value)}")
                
                content = " | ".join(content_parts) if content_parts else str(item)
                
                # Build metadata
                if self.config.metadata_fields:
                    metadata = {k: item.get(k) for k in self.config.metadata_fields if k in item}
                else:
                    metadata = item.copy()
                
                # Add timestamp
                metadata['scraped_at'] = datetime.utcnow().isoformat()
                
                document = {
                    "id": str(doc_id),
                    "content": content,
                    "metadata": metadata
                }
                
                batch.append(document)
                
                # Yield batch when full
                if len(batch) >= batch_size:
                    logger.info(f"Yielding batch of {len(batch)} documents")
                    yield batch
                    batch = []
            
            # Yield remaining documents
            if batch:
                logger.info(f"Yielding final batch of {len(batch)} documents")
                yield batch
                
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
        logger.info("JSON scraper session closed")