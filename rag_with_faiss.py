import os
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
import fitz  # PyMuPDF for PDF text extraction

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Extract text from PDF files
def extract_text_from_pdfs(pdf_folder):
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            documents.append(text)
            doc.close()
    return documents


# Build FAISS-based retriever
def build_retriever(documents):
    retriever_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    embeddings = retriever_model.encode(documents, convert_to_tensor=False)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return retriever_model, index, documents


# Load generation model
def load_generator():
    model_name = "sshleifer/distilbart-cnn-12-6"  # Revert to DistilBART, ~139M params, ~0.8GB VRAM
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device).half()  # Use FP16 for 6G VRAM
    model.eval()
    return tokenizer, model


# RAG inference with FAISS
def rag_answer(question, retriever_model, index, documents, tokenizer, model, top_k=1):
    query_embedding = retriever_model.encode([question], convert_to_tensor=False)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]

    # Use a specific DistilBART prompt format for detailed Q&A
    context = " ".join(retrieved_docs)[:400]  # Increase context length for more context
    input_text = f"question: {question} context: {context} provide a detailed and comprehensive answer:"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=300, truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=250,  # Increase to allow longer answers
            num_beams=12,  # Increase beams for better quality
            early_stopping=True,
            temperature=0.7,  # Balanced randomness
            top_k=50,  # Increase top-k for variety
            top_p=0.95,  # Increase for broader sampling
            no_repeat_ngram_size=4,  # Increase to prevent repetition
            repetition_penalty=2.5,  # Increase to penalize repeated tokens
            length_penalty=2.5  # Increase to favor longer answers
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # Remove trailing noise and ensure meaningful output
    if len(answer.split()) < 5 or answer.isspace() or answer.endswith("None") or "None" in answer.split()[-3:]:
        answer = "Sorry, I couldn't generate a clear answer. Please refine your question or try another paper."
    return retrieved_docs[0][:500], answer


# Gradio interface
def gradio_interface(question):
    retrieved_doc, answer = rag_answer(
        question, retriever_model, index, documents, tokenizer, generator_model
    )
    return retrieved_doc, answer


# Main function
def main():
    global retriever_model, index, documents, tokenizer, generator_model

    pdf_folder = "papers"
    if not os.path.exists(pdf_folder):
        raise FileNotFoundError(f"Folder '{pdf_folder}' not found. Please add PDF files.")

    print("Extracting text from PDFs...")
    documents = extract_text_from_pdfs(pdf_folder)
    if not documents:
        raise ValueError("No valid PDF files found.")

    print("Building FAISS retriever...")
    retriever_model, index, documents = build_retriever(documents)
    print("Loading generator model...")
    tokenizer, generator_model = load_generator()

    interface = gr.Interface(
        fn=gradio_interface,
        inputs=gr.Textbox(label="Enter your question about papers"),
        outputs=[
            gr.Textbox(label="Retrieved Document Snippet"),
            gr.Textbox(label="Generated Answer")
        ],
        title="Paper Q&A with RAG (FAISS)",
        description="Ask questions about research papers using FAISS for retrieval."
    )
    interface.launch(debug=True)


if __name__ == "__main__":
    main()