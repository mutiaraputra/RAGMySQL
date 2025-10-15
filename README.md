# RAGMySQL

RAGMySQL adalah proyek open-source yang mengimplementasikan sistem Retrieval Augmented Generation (RAG) lengkap untuk mengintegrasikan data dari database MySQL ke dalam chatbot AI. Sistem ini menggunakan TiDB sebagai vector database dan OpenRouter API untuk akses ke berbagai model LLM (Large Language Models).

## Deskripsi Sistem RAG

Sistem RAGMySQL bekerja melalui pipeline berikut:
1. **MySQL Scraping**: Mengekstrak data teks dari tabel MySQL dengan batching untuk efisiensi memori.
2. **Embedding Generation**: Mengkonversi teks menjadi vektor embeddings menggunakan model sentence-transformers lokal.
3. **TiDB Vector Storage**: Menyimpan embeddings di TiDB dengan indeks HNSW untuk pencarian similarity cepat.
4. **AI Chatbot**: Menggunakan LangChain dan OpenRouter API untuk retrieval augmented generation, menghasilkan respons AI yang akurat berdasarkan data yang diambil.

Sistem ini memungkinkan pengguna untuk membangun chatbot yang dapat menjawab pertanyaan berdasarkan data pribadi dari MySQL, dengan dukungan untuk berbagai provider LLM melalui OpenRouter.

## Arsitektur dan Flow Diagram

### Komponen Utama
- **MySQL Scraper**: Mengekstrak dan memproses data dari MySQL.
- **Text Chunker**: Memecah teks panjang menjadi chunk dengan overlap untuk mempertahankan konteks.
- **Embedding Generator**: Menghasilkan vektor embeddings dari teks menggunakan sentence-transformers.
- **TiDB Vector Store**: Menyimpan dan mengindeks vektor di TiDB dengan HNSW index.
- **RAG ChatBot**: Menggabungkan retrieval dan generation menggunakan LangChain + OpenRouter.
- **Ingestion Pipeline**: Mengorkestrasi seluruh proses ingestion data.

### Flow Diagram
```
MySQL Database
     ↓
MySQL Scraper → Text Chunker → Embedding Generator → TiDB Vector Store
                                                          ↓
User Query → Embedding Generator → TiDB Search → RAG ChatBot → OpenRouter LLM → Response
```

## Prerequisites

- **Python**: Versi 3.9 atau lebih tinggi.
- **MySQL Database**: Database MySQL yang berisi data teks untuk di-scrape.
- **TiDB**: TiDB Cloud atau self-managed versi 8.4+ dengan dukungan VECTOR dan HNSW indexes.
- **OpenRouter API Key**: Diperlukan untuk akses ke LLM providers. Dapatkan dari [OpenRouter](https://openrouter.ai/).
- **Dependencies**: Lihat `requirements.txt` untuk daftar lengkap.

## Instalasi

1. **Clone Repository**:
   ```bash
   git clone https://github.com/mutiaraputra/RAGMySQL.git
   cd RAGMySQL
   ```

2. **Setup Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Pada Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Database**:
   Jalankan script setup untuk membuat tabel vector di TiDB:
   ```bash
   python main.py setup
   ```

## Konfigurasi

1. **Copy Environment File**:
   ```bash
   cp .env.example .env
   ```

2. **Isi Credentials**:
   Edit file `.env` dan isi nilai berikut:
   - **MySQL**: `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`
   - **TiDB**: `TIDB_HOST`, `TIDB_PORT`, `TIDB_USER`, `TIDB_PASSWORD`, `TIDB_DATABASE`, `TIDB_USE_TLS`
   - **OpenRouter**: `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` (default: `anthropic/claude-3.5-sonnet`), `OPENROUTER_BASE_URL` (default: `https://openrouter.ai/api/v1`)
   - **App Settings**: `EMBEDDING_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`), `EMBEDDING_DIMENSION` (default: 384), dll.

   Lihat `.env.example` untuk komentar detail setiap variabel.

## Penjelasan OpenRouter

OpenRouter adalah platform yang menyediakan akses terpadu ke berbagai provider LLM seperti OpenAI (GPT-4), Anthropic (Claude), Meta (Llama), Google (Gemini), dan lainnya. Dengan satu API key, Anda dapat menggunakan model terbaik untuk kebutuhan spesifik:

- **Model Unggulan**: `openai/gpt-4o`, `anthropic/claude-3.5-sonnet` untuk kualitas tinggi.
- **Seimbang**: `openai/gpt-4o-mini`, `anthropic/claude-3-haiku` untuk performa dan biaya.
- **Efisien Biaya**: `meta-llama/llama-3.1-70b-instruct`, `google/gemini-pro`.

OpenRouter menawarkan pricing kompetitif, routing otomatis, dan fallback. Kunjungi [OpenRouter Models](https://openrouter.ai/models) untuk daftar lengkap dan harga terkini.

## Cara Penggunaan

### 1. Jalankan Scraping Data
Scrape data dari tabel MySQL dan simpan ke TiDB:
```bash
python main.py scrape --tables articles,products --config examples/table_config.json
```

### 2. Jalankan Chatbot
Interaksi dengan chatbot AI:
```bash
python main.py chat --top-k 5 --model anthropic/claude-3.5-sonnet
```

### 3. Setup Database
Inisialisasi tabel vector di TiDB:
```bash
python main.py setup --verify-only
```

### Contoh Query
- "Apa itu machine learning?"
- "Jelaskan produk X dari database."
- "Bagaimana cara menggunakan fitur Y?"

## Struktur Direktori

```
RAGMySQL/
├── config/                 # Konfigurasi aplikasi
│   ├── settings.py         # Pydantic settings untuk environment variables
│   └── __init__.py
├── src/                    # Source code utama
│   ├── scraper/            # Modul scraping MySQL
│   ├── embeddings/         # Modul embedding dan chunking
│   ├── vectorstore/        # Modul TiDB vector storage
│   ├── chatbot/            # Modul RAG chatbot dengan OpenRouter
│   └── pipeline/           # Orchestration pipeline
├── scripts/                # CLI scripts
├── tests/                  # Unit dan integration tests
├── examples/               # Contoh konfigurasi dan queries
├── logs/                   # Log files (di-ignore git)
├── main.py                 # Entry point utama
├── requirements.txt        # Dependencies Python
├── .env.example            # Template environment variables
├── .gitignore              # Git ignore rules
├── README.md               # Dokumentasi ini
├── ARCHITECTURE.md         # Dokumentasi arsitektur teknis
└── LICENSE                 # Lisensi proyek
```

### Penjelasan Modul Utama
- **config/settings.py**: Mengelola konfigurasi dari environment variables menggunakan Pydantic.
- **src/scraper/mysql_scraper.py**: Mengekstrak data dari MySQL dengan batching.
- **src/embeddings/generator.py**: Menghasilkan embeddings menggunakan sentence-transformers.
- **src/vectorstore/tidb_store.py**: Interface ke TiDB vector database.
- **src/chatbot/rag_bot.py**: Implementasi RAG chatbot dengan LangChain dan OpenRouter.
- **src/pipeline/ingestion.py**: Mengorkestrasi proses ingestion data.

## Troubleshooting

### Koneksi Database
- **MySQL Connection Failed**: Periksa credentials di `.env` dan pastikan MySQL server berjalan.
- **TiDB Connection Error**: Verifikasi TiDB versi 8.4+, enable TLS jika diperlukan, dan cek firewall.

### API Rate Limits
- **OpenRouter Rate Limit**: Tingkatkan interval request atau pilih model dengan limit lebih tinggi. Lihat [OpenRouter Docs](https://openrouter.ai/docs) untuk limits.

### Vector Dimension Mismatch
- Pastikan `EMBEDDING_DIMENSION` di `.env` sesuai dengan model embedding (default 384 untuk all-MiniLM-L6-v2). Jika error, recreate tabel TiDB.

### OpenRouter Model Selection
- Model tidak tersedia: Periksa [OpenRouter Models](https://openrouter.ai/models) untuk availability. Gunakan fallback model seperti `openai/gpt-4o-mini`.

### Masalah Umum Lain
- **Import Error**: Pastikan semua dependencies terinstall dari `requirements.txt`.
- **Memory Issues**: Kurangi `BATCH_SIZE` untuk dataset besar.
- **Embedding Generation Slow**: Gunakan GPU jika tersedia dengan `device='cuda'`.

Untuk log detail, lihat file di direktori `logs/`.

## Lisensi dan Kontribusi

### Lisensi
Proyek ini menggunakan lisensi MIT. Lihat file `LICENSE` untuk detail.

### Kontribusi
Kontribusi sangat diterima! Silakan:
1. Fork repository.
2. Buat branch fitur baru.
3. Commit perubahan dengan pesan deskriptif.
4. Push ke branch dan buat Pull Request.

Untuk guidelines lengkap, lihat `CONTRIBUTING.md` (jika ada). Pastikan semua tests pass sebelum submit.

---

Untuk pertanyaan atau dukungan, buka issue di GitHub atau hubungi maintainer.
