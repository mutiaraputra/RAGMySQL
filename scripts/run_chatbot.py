import signal
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from config.settings import get_initialized_settings
from src.chatbot.rag_bot import RAGChatBot
from src.embeddings.generator import EmbeddingGenerator
from src.vectorstore.pinecone_store import PineconeVectorStore

app = typer.Typer()
console = Console()


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    console.print("\n[bold red]Goodbye! 👋[/bold red]")
    sys.exit(0)


@app.command()
def main(
    top_k: int = typer.Option(5, help="Number of top documents to retrieve"),
    model: Optional[str] = typer.Option(None, help="Override Gemini model from config"),
    no_history: bool = typer.Option(False, help="Disable conversation memory"),
):
    """Run the interactive RAG chatbot with Google Gemini."""
    signal.signal(signal.SIGINT, signal_handler)
    run_chatbot(top_k=top_k, model=model, no_history=no_history)


def run_chatbot(top_k: int = 5, model: Optional[str] = None, no_history: bool = False):
    """Callable entrypoint for the chatbot."""
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Get settings
        settings = get_initialized_settings()
        
        # Override model if provided
        if model:
            settings.gemini.model = model

        # Initialize components
        console.print("[yellow]Loading embedding model...[/yellow]")
        embedding_generator = EmbeddingGenerator()
        
        console.print("[yellow]Connecting to Pinecone...[/yellow]")
        vector_store = PineconeVectorStore(
            settings.pinecone,
            settings.app.embedding_dimension
        )
        
        console.print("[yellow]Initializing Gemini chatbot...[/yellow]")
        chatbot = RAGChatBot(
            gemini_config=settings.gemini,
            embedding_generator=embedding_generator,
            vector_store=vector_store,
        )

        # Welcome message
        welcome_text = Text(
            f"Welcome to RAG Chatbot with Google Gemini!\n"
            f"Model: {settings.gemini.model}\n"
            f"Temperature: {settings.gemini.temperature}",
            style="bold green"
        )
        console.print(Panel(welcome_text, title="🤖 RAG Chatbot", border_style="blue"))
        console.print("Type your questions or use commands (/help for list). Press Ctrl+C to exit.\n")

        # Conversation history and last sources
        conversation_history = []
        last_sources = []

        while True:
            try:
                user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    # Handle commands
                    command = user_input.lower()
                    if command == "/exit":
                        console.print("[bold red]Goodbye! 👋[/bold red]")
                        break
                    elif command == "/clear":
                        conversation_history.clear()
                        last_sources.clear()
                        console.print("[yellow]Conversation history and sources cleared.[/yellow]")
                    elif command == "/sources":
                        if last_sources:
                            console.print("[bold magenta]Last Sources:[/bold magenta]")
                            for i, source in enumerate(last_sources, 1):
                                console.print(
                                    f"[dim]{i}. {source['content'][:100]}... "
                                    f"(Score: {source['similarity_score']:.3f})[/dim]"
                                )
                        else:
                            console.print("[yellow]No sources available.[/yellow]")
                    elif command == "/help":
                        help_text = """
Available commands:
/exit - Quit the chatbot
/clear - Clear conversation history and sources
/sources - Show sources from last response
/model - Show current Gemini model configuration
/help - Show this help message
                        """.strip()
                        console.print(Panel(help_text, title="Help", border_style="green"))
                    elif command == "/model":
                        current_settings = get_initialized_settings()
                        model_info = f"""
Model: {current_settings.gemini.model}
Temperature: {current_settings.gemini.temperature}
Max Tokens: {current_settings.gemini.max_output_tokens}
Top-P: {current_settings.gemini.top_p}
Top-K: {current_settings.gemini.top_k}
                        """.strip()
                        console.print(Panel(model_info, title="Gemini Configuration", border_style="blue"))
                    else:
                        console.print(
                            f"[red]Unknown command: {user_input}. "
                            f"Type /help for available commands.[/red]"
                        )
                    continue

                # Process question
                console.print("[dim]Thinking...[/dim]", end="")

                if no_history:
                    response = chatbot.ask(user_input, top_k=top_k)
                else:
                    response = chatbot.chat(user_input, conversation_history, top_k=top_k)

                # Clear thinking message
                console.print("\r" + " " * 20 + "\r", end="")

                # Display answer
                answer_panel = Panel(response["answer"], title="🤖 Gemini", border_style="green")
                console.print(answer_panel)

                # Display sources if available
                if response["sources"]:
                    last_sources = response["sources"]
                    sources_text = "\n".join([
                        f"• {source['content'][:150]}... (Score: {source['similarity_score']:.3f})"
                        for source in response["sources"]
                    ])
                    console.print(Panel(sources_text, title="📚 Sources", border_style="yellow"))
                else:
                    last_sources = []
                    console.print("[dim]No sources found.[/dim]")

                # Display model and confidence
                info_text = f"Model: {response['model_used']}"

                if response.get("confidence_score"):
                    info_text += f" | Confidence: {response['confidence_score']:.3f}"

                console.print(f"[dim]{info_text}[/dim]\n")

            except Exception as e:
                console.print(f"[bold red]Error: {str(e)}[/bold red]")
                console.print("[yellow]Please try again or check your configuration.[/yellow]\n")

    except Exception as e:
        console.print(f"[bold red]Initialization error: {str(e)}[/bold red]")
        console.print("[yellow]Please check your configuration and try again.[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    app()
