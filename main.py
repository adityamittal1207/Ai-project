from graph import create_graph, compile_workflow
from langchain_community.document_loaders import WebBaseLoader, TextLoader, DirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
import os

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule

console = Console()


# server = 'ollama'
# model = 'llama3:instruct'
# model_endpoint = None

models = ["gpt-4o", "gpt-4o-mini", "claude-opus", "claude-haiku", "gemini-2.0-flash", "gemini-1.5-pro", "mistral-large"]
load_dotenv()

model_endpoint = None

iterations = 40

api_keys = {"openai": os.getenv("OPENAI_KEY"), "anthropic": os.getenv("ANTHROPIC_KEY"), "google": os.getenv("GOOGLE_KEY"), "mistral": os.getenv("MISTRAL_KEY")}

print(api_keys)

urls = [
    "https://outofcontrol.substack.com/p/is-ai-a-tradition-machine",
    "https://outofcontrol.substack.com/p/large-language-models-could-re-decentralize",
    "https://www.foxnews.com/opinion/christians-shouldnt-fear-ai-should-partner-with-it"
]

console.print(Panel.fit(
    "AI Viewpoint Analysis Pipeline",
    style="bold green",
    subtitle="Extracting, classifying, and verifying viewpoints from documents"
))

console.print("[bold blue]Loading documents...[/bold blue]")
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
) as progress:
    task = progress.add_task("[cyan]Downloading articles...", total=None)
    # docs = DirectoryLoader("./standard_cases").load()
    docs = WebBaseLoader(urls).load()
    progress.update(task, completed=True)

for i, doc in enumerate(docs):
    doc.metadata = {'Document Identifier': f"Document {i + 1}"}

console.print(f"[green]✓ Loaded {len(docs)} documents[/green]")
console.print()

console.print(Rule(title="[bold]Setting up Analysis Graph[/bold]"))
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
) as progress:
    task = progress.add_task("[cyan]Creating graph and compiling workflow...", total=None)
    embeddings = OpenAIEmbeddings(openai_api_key=api_keys["openai"])

if __name__ == "__main__":
    for model in models:
        graph = create_graph(model = model, model_endpoint=model_endpoint, output_parser=StrOutputParser(), api_keys=api_keys, embeddings=embeddings)
        workflow = compile_workflow(graph)
        progress.update(task, completed=True)
            
        console.print(f"[green]✓ Graph and workflow created successfully for [bold] {model} [/bold] [/green]")
        console.print()


        console.print(Rule(title="[bold]Starting Analysis[/bold]"))
        console.print(f"[bold cyan]Analyzing Document {docs[2].metadata['Document Identifier']}[/bold cyan]")
        console.print()
        
        for i, event in enumerate(workflow.stream({"doc": docs[2]})):
            if i > 0: 
                console.print(f"\n[dim]Step {i} completed[/dim]")
            
        console.print()
        console.print(Panel(f"[bold green]Analysis Complete for [bold cyan] {model} [/bold cyan] ![/bold green]", border_style="green"))

    