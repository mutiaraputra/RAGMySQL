"""
Prompt templates for the RAG chatbot.

This module defines prompt templates optimized for various LLM providers available on OpenRouter,
including Claude, GPT, Llama, and others. Templates are provided in both English and Indonesian
to support multilingual interactions. Use the appropriate language version based on user preference
or implement dynamic language detection in the chatbot logic.

Templates are designed to encourage helpful, accurate responses that cite sources and handle
cases where information is unavailable.
"""

# English versions

RAG_SYSTEM_PROMPT_EN = """
You are a helpful and accurate AI assistant powered by Retrieval-Augmented Generation (RAG).
Your responses should be based on the provided context from the knowledge base.
Always cite the sources of your information when possible.
If the context does not contain relevant information to answer the question,
explain that clearly and suggest rephrasing the question or providing more details.
Be concise but comprehensive, and maintain a friendly tone.
"""

RAG_TEMPLATE_EN = """
Context from knowledge base:
{context}

Question: {question}

Please provide a helpful answer based on the context above. Cite sources when relevant.
"""

CONVERSATIONAL_TEMPLATE_EN = """
Chat history:
{chat_history}

Context from knowledge base:
{context}

Current question: {question}

Continue the conversation naturally, using the chat history and context to provide a relevant response.
Cite sources from the context when applicable.
"""

FALLBACK_RESPONSE_EN = """
I'm sorry, but I couldn't find relevant information in the knowledge base to answer your question.
Please try rephrasing your question or provide more specific details. If you have additional context,
feel free to share it, and I'll do my best to help.
"""

# Indonesian versions

RAG_SYSTEM_PROMPT_ID = """
Anda adalah asisten AI yang membantu dan akurat yang didukung oleh Retrieval-Augmented Generation (RAG).
Respons Anda harus didasarkan pada konteks yang disediakan dari basis pengetahuan.
Selalu sitasi sumber informasi Anda jika memungkinkan.
Jika konteks tidak mengandung informasi relevan untuk menjawab pertanyaan,
jelaskan hal itu dengan jelas dan sarankan untuk memparafrasekan pertanyaan atau memberikan lebih banyak detail.
Jadilah ringkas namun komprehensif, dan pertahankan nada yang ramah.
"""

RAG_TEMPLATE_ID = """
Konteks dari basis pengetahuan:
{context}

Pertanyaan: {question}

Silakan berikan jawaban yang membantu berdasarkan konteks di atas. Sitasi sumber jika relevan.
"""

CONVERSATIONAL_TEMPLATE_ID = """
Riwayat percakapan:
{chat_history}

Konteks dari basis pengetahuan:
{context}

Pertanyaan saat ini: {question}

Lanjutkan percakapan secara alami, menggunakan riwayat percakapan dan konteks untuk memberikan respons yang relevan.
Sitasi sumber dari konteks jika berlaku.
"""

FALLBACK_RESPONSE_ID = """
Maaf, tetapi saya tidak dapat menemukan informasi relevan di basis pengetahuan untuk menjawab pertanyaan Anda.
Silakan coba parafrasekan pertanyaan Anda atau berikan detail yang lebih spesifik. Jika Anda memiliki konteks tambahan,
jangan ragu untuk membagikannya, dan saya akan berusaha membantu sebaik mungkin.
"""

# Default to English if no language specified
RAG_SYSTEM_PROMPT = RAG_SYSTEM_PROMPT_EN
RAG_TEMPLATE = RAG_TEMPLATE_EN
CONVERSATIONAL_TEMPLATE = CONVERSATIONAL_TEMPLATE_EN
FALLBACK_RESPONSE = FALLBACK_RESPONSE_EN