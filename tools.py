from state import AgentGraphState
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation
import json
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def parse_viewpoints(state: AgentGraphState):
    viewpoints = []
    lines = state["viewpoint_extractor_response"].strip().split('\n')

    for line in lines:
        line = line.strip().replace("**", "")

        if line.startswith('- Viewpoints:'):
            continue
        elif line.startswith('- '):
            viewpoint = line[2:].strip()
            viewpoints.append(viewpoint)

    state["viewpoints"] = viewpoints

    console.print(Panel.fit(
        "\n".join([f"• {viewpoint}" for viewpoint in viewpoints]),
        title="[bold cyan]Extracted Viewpoints[/bold cyan]",
        border_style="cyan"
    ))
    
    return {"viewpoints": state["viewpoints"]}

def verify_point(point, document_embeddings, original_chunks, embeddings):
    console.print(f"[yellow]Verifying point:[/yellow] {point}")
    
    point_vector = embeddings.embed_query(point)
    similarities = cosine_similarity([point_vector], document_embeddings).flatten()
    
    good_point = np.where(similarities >= 0.8)
    above_threshold = good_point[0]
    top_similarities = sorted(above_threshold, key=lambda idx: similarities[idx], reverse=True)
    
    if len(top_similarities) > 0:
        console.print("[bold green]Top matches:[/bold green]")
        for idx in top_similarities[:5]:
            console.print(f"  • [green]Document Chunk {idx}:[/green] Similarity = {similarities[idx]:.4f}")
    
    is_verified = len(above_threshold) > 0
    return point, "yes" if is_verified else "no", above_threshold

def verify_points_in_document(state: AgentGraphState, embeddings: None, text_splitter: None):
    doc = state["doc"]
    extracted_points = state["classified_viewpoints"]
    
    console.print()
    console.rule(f"[bold magenta]Verifying Points in {doc.metadata['Document Identifier']}[/bold magenta]")
    console.print()
    
    original_chunks = text_splitter.split_documents([doc])
    doc_embeddings = [embeddings.embed_query(chunk.page_content) for chunk in original_chunks]
    
    verified_points = {
        'Pros': [],
        'Cons': [],
        'Neutral': []
    }

    for category, points in extracted_points.items():
        if points:
            console.print(f"[bold]{category}[/bold]:")
            
        for point in points:
            verified_point, verification_response, matches = verify_point(point, doc_embeddings, original_chunks, embeddings)
            if verification_response == "yes":
                verified_points[category].append((verified_point, doc.metadata['Document Identifier']))
                console.print(f"  ✓ [green]Verified:[/green] {verified_point}")
            else:
                console.print(f"  ✗ [red]Not verified:[/red] {verified_point}")
    
    state["verified_points"] = verified_points

    # Print final verified points summary
    console.print()
    console.rule("[bold blue]Final Verified Points Summary[/bold blue]")
    
    for category, points in verified_points.items():
        if points:
            text = Text()
            text.append(f"\n{category}:\n", style="bold")
            for point, doc_id in points:
                text.append(f"• {point} ", style="default")
                text.append(f"(Source: {doc_id})", style="italic cyan")
                text.append("\n")
            console.print(Panel(text, border_style="blue"))
    
    return {"verified_points": state["verified_points"]}