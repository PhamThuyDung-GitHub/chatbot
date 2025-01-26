import numpy as np
def vector_search(model, query, collection, columns_to_answer, number_docs_retrieval):
    query_embeddings = model.encode([query])
    
    # Fetch results from the collection
    search_results = collection.query(
        query_embeddings=query_embeddings, 
        n_results=number_docs_retrieval
    )  
    metadatas = search_results['metadatas']  
    scores = search_results['distances']  

    # Prepare the search result output
    search_result = ""
    for i, (meta, score) in enumerate(zip(metadatas[0], scores[0]), start=1):  
        search_result += f"\n{i}) Distances: {score:.4f}"  
        for column in columns_to_answer:
            if column in meta:
                search_result += f" {column}: {meta.get(column)}"
        search_result += "\n"

    return metadatas, search_result

def generate_hypothetical_documents(model, query, num_samples=10):
    hypothetical_docs = []
    for _ in range(num_samples):
        enhanced_prompt = f"Write a paragraph that answers the question: {query}"
        # Use the Gemini model stored in session state to generate the document
        response = model.generate_content(enhanced_prompt)
        hypothetical_docs.append(response)
    
    return hypothetical_docs

def encode_hypothetical_documents(documents, encoder_model):
    embeddings = [encoder_model.encode([doc])[0] for doc in documents]
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding

def hyde_search(llm_model, encoder_model, query, collection, columns_to_answer, number_docs_retrieval, num_samples=10):
    
    hypothetical_documents = generate_hypothetical_documents(llm_model, query, num_samples)

    print("hypothetical_documents", hypothetical_documents)
    aggregated_embedding = encode_hypothetical_documents(hypothetical_documents, encoder_model)

   
    search_results = collection.query(
        query_embeddings=aggregated_embedding, 
        n_results=number_docs_retrieval)  # Fetch top 10 results
    
    search_result = ""

    metadatas =  search_results['metadatas']

    i = 0
    for meta in metadatas[0]:
        i += 1
        search_result += f"\n{i})"
        for column in columns_to_answer:
            if column in meta:
                search_result += f" {column.capitalize()}: {meta.get(column)}"

        search_result += "\n"
    return metadatas, search_result