import logging
from typing import List, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import JSONEndpointConfig


class JSONScraper:
    """Scraper for fetching data from JSON API endpoints."""

    def __init__(self, config: JSONEndpointConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup session with retry logic
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Headers sederhana
        self.session.headers.update({
            'User-Agent': 'PostmanRuntime/7.36.0',
            'Accept': 'application/json',
            'Connection': 'keep-alive',
        })

    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch data from JSON endpoint."""
        url = str(self.config.url)
        
        try:
            self.logger.info(f"Fetching data from {url}")
            
            response = self.session.request(
                method=self.config.method,
                url=url,
                timeout=60,
                verify=True
            )
            
            self.logger.info(f"Response status: {response.status_code}")
            self.logger.info(f"Response content-type: {response.headers.get('Content-Type', 'unknown')}")
            
            response.raise_for_status()
            
            # Parse JSON response
            json_response = response.json()
            
            self.logger.info(f"Response status field: {json_response.get('status', 'unknown')}")
            self.logger.info(f"Response count: {json_response.get('count', 0)}")
            
            # Extract data array from response
            # Format: {"status": "success", "count": 10, "data": [...]}
            if isinstance(json_response, dict):
                if json_response.get('status') == 'success' and 'data' in json_response:
                    data = json_response['data']
                elif 'data' in json_response:
                    data = json_response['data']
                elif 'results' in json_response:
                    data = json_response['results']
                elif 'items' in json_response:
                    data = json_response['items']
                else:
                    data = [json_response]
            else:
                data = json_response if isinstance(json_response, list) else [json_response]
            
            if not isinstance(data, list):
                data = [data]
            
            self.logger.info(f"Fetched {len(data)} records from endpoint")
            return data

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error: {e}")
            raise
        except ValueError as e:
            self.logger.error(f"JSON parse error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            raise

    def extract_text_content(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract text content from records for embedding.
        
        Prioritizes 'konten_rag' field if available, otherwise combines all text fields.
        """
        documents = []
        
        for i, record in enumerate(records):
            try:
                record_id = str(record.get('id', i))
                
                # Prioritas: gunakan field 'konten_rag' jika ada
                if 'konten_rag' in record and record['konten_rag']:
                    content = record['konten_rag']
                else:
                    # Fallback: gabungkan semua field text
                    text_parts = []
                    for key, value in record.items():
                        if value is not None and key != 'id':
                            if isinstance(value, str) and value.strip():
                                text_parts.append(f"{key}: {value}")
                            elif isinstance(value, (int, float)):
                                text_parts.append(f"{key}: {value}")
                    content = "\n".join(text_parts)
                
                # Build metadata
                metadata = {
                    "id": record_id,
                    "namapel": record.get('namapel', ''),
                    "barang": record.get('barang', ''),
                    "tglmasuk": record.get('tglmasuk', ''),
                    "tglkeluar": record.get('tglkeluar', ''),
                    "biaya": record.get('biaya', ''),
                }
                
                # Remove empty metadata values
                metadata = {k: v for k, v in metadata.items() if v}
                
                if content and content.strip():
                    documents.append({
                        "id": record_id,
                        "content": content.strip(),
                        "metadata": metadata
                    })
                    self.logger.debug(f"Extracted document {record_id}: {content[:100]}...")
                    
            except Exception as e:
                self.logger.warning(f"Error processing record {i}: {e}")
                continue
        
        self.logger.info(f"Extracted {len(documents)} documents from {len(records)} records")
        return documents