from state import AgentGraphState
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation

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

    print(viewpoints)
    return {"viewpoints": state["viewpoints"]}
def verify_point(point, document_embeddings, original_chunks, embeddings):
    print(f"Verifying point: {point}")
    
    point_vector = embeddings.embed_query(point)
    similarities = cosine_similarity([point_vector], document_embeddings).flatten()
    
    good_point = np.where(similarities >= 0.87)
    above_threshold = good_point[0]
    top_similarities = sorted(above_threshold, key=lambda idx: similarities[idx], reverse=True)
    
    print("Top 5 matches with their text content:")
    for idx in top_similarities[:5]:
        print(f"  - Document Chunk {idx}: Similarity = {similarities[idx]:.4f}")
        # print(f"    Chunk Text: {original_chunks[idx].page_content}\n")
    
    is_verified = len(above_threshold) > 0
    return point, "yes" if is_verified else "no", above_threshold
def verify_points_in_document(state: AgentGraphState, embeddings: None, text_splitter: None):
    doc = state["doc"]
    extracted_points = state["classified_viewpoints"]
    print(f"Verifying points in {doc.metadata['Document Identifier']}")
    original_chunks = text_splitter.split_documents([doc])
    doc_embeddings = [embeddings.embed_query(chunk.page_content) for chunk in original_chunks]
    
    verified_points = {
        'Pros': [],
        'Cons': [],
        'Neutral': []
    }

    for category, points in extracted_points.items():
        for point in points:
            verified_point, verification_response, matches = verify_point(point, doc_embeddings, original_chunks, embeddings)
            if verification_response == "yes":
                verified_points[category].append((verified_point, doc.metadata['Document Identifier']))
    
    state["verified_points"] = verified_points

    print(verified_points)
    
    return {"verified_points": state["verified_points"]}