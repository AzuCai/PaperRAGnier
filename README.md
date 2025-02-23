# PaperRAGnizer
A lightweight Retrieval-Augmented Generation (RAG) system for answering questions about research papers, optimized for low-resource environments (6G VRAM). Built with Python, it processes PDF papers, retrieves relevant content using FAISS or Chroma, and generates detailed responses using models like DistilBART or T5. Ideal for researchers, students, and developers exploring NLP and LLMs.

## Features
- **PDF Processing**: Extracts text from research PDF papers to build a knowledge base.
- **Dual Retrieval Options**: Implements two retrieval methods:
  - **FAISS**: Fast, memory-based vector indexing for lightweight similarity search.
  - **Chroma**: Modern vector database for scalable and persistent retrieval.
- **Answer Generation**: Uses lightweight models (e.g., `sshleifer/distilbart-cnn-12-6`, `t5-base`) to generate natural language answers based on retrieved context, optimized for 6G VRAM.
- **Interactive Interface**: Provides a Gradio-based web UI for easy question-answering.
- **Low-Resource Optimization**: Designed to run on a 6G VRAM GPU with FP16 precision, ensuring accessibility on limited hardware.

## Knowledge Points
This project demonstrates practical skills in:
- **Natural Language Processing (NLP)**: Text extraction, embedding generation, and sequence-to-sequence modeling.
- **Retrieval-Augmented Generation (RAG)**: Combines dense vector retrieval with generative AI using FAISS or Chroma.
- **Model Optimization**: Uses lightweight models like DistilBART (`sshleifer/distilbart-cnn-12-6`) or T5-base, optimized for 6G VRAM.
- **Web Deployment**: Integrates Gradio for a user-friendly interface.

## Prerequisites
- **OS**: Windows (tested), Linux, or macOS.
- **Hardware**: GPU with 6G VRAM (optional, CPU fallback available).
- **Anaconda**: Recommended for environment management.
- **Python**: Version 3.9.

## Installation and Deployment
Follow these steps to set up and run the project locally.

### 1. Clone the Repository

git clone https://github.com/AzuCai/PaperRAGnizer.git
cd PaperRAGnizer

### 2. Prepare Papers
Create a folder named papers in the project root.
Download research PDF papers (e.g., from arXiv) and place them in papers.

### 3. Set Up the Environment
Open Anaconda Prompt (Windows) or terminal (Linux/macOS) and run:

#### Create a new environment
conda create -n rag_env python=3.9
conda activate rag_env

#### Install remaining dependencies
pip install transformers sentence-transformers faiss-cpu gradio PyMuPDF chromadb tiktoken blobfile sentencepiece

### 4. Run the Application
Choose one of the two retrieval implementations:

FAISS Version:
python rag_with_faiss.py

Chroma Version:
python rag_with_chroma.py

#### Note: A Gradio interface will launch in your browser. Enter a question (e.g., "What is GPT-4?") to see retrieved snippets and generated answers.


### 5. Future Improvements
Fine-tune the generation model (e.g., sshleifer/distilbart-cnn-12-6 or t5-base) with domain-specific Q&A data.
Add persistent storage for Chroma embeddings.
Support multi-language papers and questions.
Enhance text extraction for complex PDF formats using alternative libraries (e.g., pdfplumber).

### 6. License
MIT License - feel free to use, modify, and share!

### 7. Contact
For questions or suggestions, open an issue or reach out to [chaoquancai2019@gmail.com].
