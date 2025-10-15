# 🧹 Extensive Data Cleaning – Intel DPDK Mailing List

This project focuses on **extensive data cleaning, preprocessing, and structuring** of the **Intel DPDK (Data Plane Development Kit) mailing list archives**, enabling downstream analysis such as **topic modeling, sentiment analysis, and contributor activity tracking**.

---

## 📘 Project Overview

The Intel DPDK mailing list contains thousands of raw email threads discussing performance patches, bug reports, and optimization ideas for Intel’s DPDK framework.  
However, the data is **unstructured**, **redundant**, and contains **HTML noise**, **thread duplications**, and **signature blocks**.

This repository provides a complete workflow for cleaning, normalizing, and preparing this text dataset for further **machine learning or NLP** applications.

---

## 🧩 Key Features

- 🧾 **Raw Email Parsing** – Extracts subject, sender, date, and message body from raw `.mbox` or `.eml` archives.  
- 🧼 **Data Cleaning Pipeline** – Removes HTML tags, quoted replies, special characters, and email signatures.  
- 🔄 **Thread Reconstruction** – Groups messages into discussion threads for context-aware analysis.  
- 🧠 **Text Normalization** – Converts to lowercase, removes stopwords, and performs lemmatization.  
- 📊 **Metadata Extraction** – Extracts and standardizes metadata like author, domain, and patch references.  
- 💾 **Export Formats** – Cleaned data is saved as structured `.csv` or `.json` for analytics or NLP models.

---

## 🏗️ Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| Language | Python 3.x |
| Data Parsing | `mailbox`, `email`, `beautifulsoup4` |
| Data Cleaning | `pandas`, `re`, `nltk`, `spacy` |
| Visualization | `matplotlib`, `seaborn` |
| NLP Support | `scikit-learn`, `sentence-transformers` (optional) |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<anfasmhsn>/Extensive-Data-Cleaning-Intel-DPDK-mailing-List.git
cd Extensive-Data-Cleaning-Intel-DPDK-mailing-List

# Install dependencies
pip install -r requirements.txt
