# 🎓 Advanced RAG Chatbot for NLP & LLM Course

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Gemini](https://img.shields.io/badge/Gemini-2.0%20Flash-orange)
![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-yellow)

**Team 4 Final Project** - *Corso di Laurea Magistrale in Ingegneria Informatica* **Università degli Studi di Salerno (UNISA)** - *Dipartimento DIEM* **Anno Accademico:** 2024/2025

---

## 📖 Project Overview
This project implements a specialized **Retrieval-Augmented Generation (RAG)** chatbot designed to assist students of the "Natural Language Processing and Large Language Models" course.

Unlike standard chatbots, this system is strictly scoped to the course domain, utilizing a **hybrid indexing pipeline** that processes both text and images from course slides to provide accurate, context-aware answers.

### 🚀 Key Features
* **Domain Specificity:** Strictly focused on LLM & NLP topics; handles off-topic queries with humor.
* **Multimodal Ingestion:** Extracts text and generates AI descriptions for images/diagrams in slides (e.g., Transformer architectures), making visual data searchable.
* **Hybrid Indexing:** Combines standard semantic chunking with AI-refined chunks to improve retrieval precision.
* **LLM-as-a-Judge:** Integrated automated evaluation system that benchmarks responses on Relevance, Coherence, Completeness, Clarity, and Accuracy.
* **Desktop App Experience:** Runs as a local windowed application using `pywebview` and `gradio`.

---

## 🛠️ Architecture
The system is built on a modular architecture:

1.  **Preprocessing:** PDFs are processed to extract text and caption images using **Gemini 2.0 Flash**.
2.  **Embedding:** Text is converted into vectors using **BAAI/bge-large-en-v1.5**.
3.  **Vector Store:** Vectors are stored in a local **FAISS** index for efficient similarity search.
4.  **Generation:** **Gemini 2.0 Flash** generates answers based on retrieved context and conversation history.

---

## 📂 Repository Structure

```text
├── src/                     # Input folder for raw PDFs (Slides)
├── LLM_doc/                 # Input folder for text enrichment
├── last_merge/              # FAISS Vector Database (Generated)
├── valutazioni/             # Logs and Evaluation JSONs
├── chatbot2.py              # MAIN APPLICATION (Run this)
├── pre_process.py           # Index Creation Script
├── pdf_2_text.py            # Text & Image Extraction Script
├── recupero_argomenti.py    # AI Text Enrichment Script
├── Relazione_LLM.pdf        # Project Report
└── README.md                # Documentation
