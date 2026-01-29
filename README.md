# In-Context Learning for English–Thai Machine Translation: Domain-Specific Evaluation on WIPO Nice Classification

---

## 📌 Project Overview

This project investigates the effectiveness of **In-context Learning (ICL)** for **English → Thai (En2Th) machine translation**, with a domain-specific focus on **goods and services descriptions** defined by the **WIPO Nice Classification** system.

The primary objective is to evaluate how different context construction strategies—ranging from traditional text-based retrieval to semantic retrieval—affect translation quality when dealing with specialized terminology commonly used in intellectual property and business contexts.

---

## 🎯 Objectives

- Evaluate the performance of **Large Language Models (LLMs)** on English–Thai translation tasks.
- Study the impact of **In-context Learning** under different retrieval strategies.
- Compare **Full-Text Search** and **Retrieval-Augmented Generation (RAG)** approaches for domain adaptation.
- Analyze translation quality across **45 WIPO Nice product and service categories**.
- Deliver a reusable **API-based RAG system** for En2Th translation.

---

## ⚙️ Methodology

### 1. Data Preparation

- **Source**: WIPO Nice Classification dataset covering goods and services categories.
- **Preprocessing**:
  - Cleaning and normalization of parallel English–Thai sentence pairs.
  - Resolution of multiple valid translations using LLM-assisted validation.
  - Sentence chunking based on punctuation to support In-context Learning.

---

### 2. Retrieval Strategies

The system supports multiple context construction strategies for In-context Learning:
- **Full-Text Search**
  - Lexical matching over WIPO documents using a SQLite-based text search engine.
  - Suitable for exact term matching and category-specific keywords.

- **Retrieval-Augmented Generation (RAG)**
  - Contextual documents retrieved and injected into the prompt.
  - Retrieval methods:
    - **Vector Search** using *all-MiniLM-L6-v2*
    - **Semantic Search** using *BGE-m3*
  - Vector storage and similarity search implemented with **FAISS**.
  - Top-*k* relevant documents (*k = 3*) used as context.

---

### 3. Model Inference

- Translation is generated using **instruction-tuned LLMs**.
- **vLLM** is employed for high-performance inference and efficient GPU utilization.
- Output is constrained to structured **JSON format** and filtered to Thai-only responses.

---

## 📊 Evaluation

- **Metric**: BLEU Score
- Comparison across:
  - Retrieval strategies
  - WIPO product and service categories

---

## 🛠️ Tech Stack

- **Retrieval & RAG**: FAISS, LangChain
- **Embedding Models**: all-MiniLM-L6-v2, BGE-m3
- **Inference Engine**: vLLM
- **Evaluation**: PyThaiNLP, NLTK
- **API Framework**: FastAPI

---

## How to use
1. download faiss vectorstore from this url, extract folder and put it in 'backend' directory:

[google drive](https://drive.google.com/file/d/18fpn7uKzeutqeau8b88HhttJnT6NBxbj/view?usp=sharing)

2. run docker
```bash
cd backend
docker compose up -d
```

3. test api (running at port 8000)
```bash
curl -X POST http://localhost:8000/translate_en2th \
  -H "Content-Type: application/json" \
  -d '{
    "wipo_id": 35,
    "english": "Retail services for clothing and footwear.",
  }'
```

---

## 📦 Deliverables

- 📄 Technical report documenting methodology and results
- 🧠 In-context learning evaluation pipeline
- 🔍 RAG-based translation system
- 🚀 Dockerized API service for En2Th translation

---

## 🚀 Future Work

- Fine-tuning LLMs for domain-specific translation
- Expansion to Th2En translation pairs
