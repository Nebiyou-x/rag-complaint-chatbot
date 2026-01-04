Below is a **comprehensive, submission-ready `README.md`** tailored specifically for the **10 Academy – AI Mastery Week 7 RAG Challenge**.
You can copy-paste this directly into your repo and adjust names if needed.

---

# 📊 Intelligent Complaint Analysis for Financial Services

### RAG-Powered Chatbot for Actionable Customer Insights

## 🚀 Project Overview

CrediTrust Financial is a fast-growing digital finance company serving over **500,000 users** across East Africa through a mobile-first platform. With thousands of customer complaints arriving monthly via in-app channels, email, and regulatory portals, internal teams struggle to extract timely, actionable insights from unstructured feedback.

This project delivers an **AI-powered Retrieval-Augmented Generation (RAG) chatbot** that transforms raw complaint narratives into **concise, evidence-backed insights**. The system enables Product Managers, Support, and Compliance teams to ask natural-language questions and instantly understand customer pain points across multiple financial products.

---

## 🎯 Business Objectives

The solution is designed to meet the following KPIs:

* ⏱ **Reduce issue identification time** from days to minutes
* 🧑‍💼 **Empower non-technical teams** to analyze complaints without data analysts
* 🔍 **Enable proactive decision-making** using real-time customer feedback

Example question supported:

> *“Why are customers unhappy with Credit Cards this month?”*

---

## 🧠 Solution Architecture

The system follows a modern **RAG pipeline**:

1. **Data Ingestion & Cleaning**
   CFPB complaint dataset is filtered, cleaned, and normalized.

2. **Text Chunking & Embeddings**
   Long narratives are split into overlapping chunks and embedded using a sentence-transformer model.

3. **Vector Database**
   Embeddings are stored in a persistent vector store (ChromaDB / FAISS) with rich metadata.

4. **Semantic Retrieval**
   User queries retrieve the most relevant complaint chunks using vector similarity search.

5. **LLM Synthesis (later tasks)**
   Retrieved evidence is passed to an LLM to generate grounded, concise answers.

6. **User Interface**
   A simple UI (Gradio/Streamlit) enables natural-language querying.

---

## 🗂 Project Structure

```text
rag-complaint-chatbot/
├── .github/
│   └── workflows/
│       └── unittests.yml          # GitHub Actions CI pipeline
├── data/
│   ├── raw/                       # Original CFPB dataset
│   └── processed/
│       └── filtered_complaints.csv
├── notebooks/
│   ├── __init__.py
│   └── README.md                  # Notebook descriptions
├── src/
│   ├── __init__.py                # Core pipeline code (EDA, chunking, embeddings)
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_chunking.py
│   └── test_embeddings.py
├── vector_store/                  # Persisted FAISS / ChromaDB index
├── app.py                         # Streamlit or Gradio application
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📂 Dataset

**Source:** Consumer Financial Protection Bureau (CFPB)

### Key Features

* Real consumer complaint narratives
* Multiple financial products and issues
* Rich metadata (product, issue, company, date, state)

### Products Used

* Credit Cards
* Personal Loans
* Savings Accounts
* Money Transfers

### Notes

* Complaints without narratives are excluded
* Text is cleaned to improve embedding quality
* Pre-built embeddings are used for large-scale retrieval (≈1.37M chunks)

---

## 🧪 Tasks Completed

### ✅ Task 1: Exploratory Data Analysis & Preprocessing

* Product distribution analysis
* Complaint narrative length analysis
* Removal of empty narratives
* Text normalization (lowercasing, boilerplate removal, special characters)
* Cleaned dataset saved to `data/processed/filtered_complaints.csv`

### ✅ Task 2: Chunking, Embedding & Vector Indexing

* Stratified sampling (10k–15k complaints)
* Recursive text chunking (500 chars, 50 overlap)
* Embedding using `all-MiniLM-L6-v2`
* Vector storage with metadata in ChromaDB / FAISS

---

## 🧠 Embedding Model

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

**Why this model?**

* Strong semantic performance
* Lightweight and fast
* 384-dimensional embeddings (memory efficient)
* Matches the pre-built vector store provided

---

## 🧪 Testing & CI

### Unit Tests

Located in the `tests/` directory:

* Text preprocessing validation
* Chunking behavior and overlap
* Embedding dimensionality checks

### Continuous Integration

* GitHub Actions workflow (`unittests.yml`)
* Automatically runs `pytest` on every push and PR

---

## 🛠 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Nebiyou-x/rag-complaint-chatbot.git
cd rag-complaint-chatbot
```

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

(After completing Tasks 3–4)

```bash
python app.py
```

Then open the local Streamlit/Gradio URL in your browser.




