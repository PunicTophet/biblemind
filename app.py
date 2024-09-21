import streamlit as st
import uuid
from collections import deque
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
pc = Pinecone(api_key="7d8508e9-37ba-4594-8851-186353ee9987")
index = pc.Index("bible-app")
connect_str = 'DefaultEndpointsProtocol=https;AccountName=biblemind;AccountKey=zf1gPQJ379T2yJL4V94KeWyqOZoO881CX8qgEVm2uK3mO8x3eZkmMpJPZpjIOD8Z+IHAGi7KuLfb+AStz1Enow==;EndpointSuffix=core.windows.net'
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
blobserviceclient = BlobServiceClient.from_connection_string(connect_str)
import sentence_transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
import anthropic 
from anthropic import Anthropic
client = anthropic.Anthropic(api_key='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')

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Global conversation history
conversation_history = {}
def query_bible(query, top_k=5):
    query_embedding = model.encode(query).tolist()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="bible"
    )
    return results['matches']

def query_commentary(chapter, verse, top_k=5):
    query = f"Commentary on chapter {chapter} verse {verse}"
    query_embedding = model.encode(query).tolist()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="commentary",
        filter={"chapter": chapter, "verse": verse}
    )
    return results['matches']

def display_bible_results(results):
    for result in results:
        st.write(f"**{result['metadata']['book']} {result['metadata']['chapter']}:{result['metadata']['verse']}**")
        st.write(result['metadata']['text'])
        st.write(f"Relevance: {result['score']:.2f}")
        st.write("---")

def display_commentary(chapter, verse):
    st.subheader(f"Commentary on Chapter {chapter}, Verse {verse}")
    commentary_results = query_commentary(chapter, verse)
    
    for result in commentary_results:
        st.write(f"**{result['metadata'].get('author', 'Unknown Commentator')}**")
        st.write(result['metadata']['text'])
        st.write(f"Relevance: {result['score']:.2f}")
        st.write("---")

def main():
    st.title("Bible Verse Lookup")

    # Get user input
    query = st.text_input("Enter a Bible verse or topic:")

    if st.button("Search"):
        # Query the Pinecone index
        results = query_bible(query)

        # Display results
        if results:
            st.subheader("Search Results:")
            display_bible_results(results)
        else:
            st.write("No results found for the query. Please try again.")

if __name__ == "__main__":
    main()
