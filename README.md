# Nexa AI PDF Chatbot

## Introduction

This demo is a PDF chatbot that can answer both common questions and PDF specific questions and generate charts based on the PDF content. It uses a combination of a local LLM, a vector database, and a LoRA model for chart generation. It is built using Nexa SDK.

## Used Models

- [Llama3.2-3B-Instruct](https://nexa.ai/meta/Llama3.2-3B-Instruct/gguf-q4_0/readme)
- [Octopus-v2-PDF](https://nexa.ai/DavidHandsome/Octopus-v2-PDF/gguf-q4_K_M/readme)
- [gemma-2-2b-instruct](https://nexa.ai/google/gemma-2-2b-instruct/gguf-fp16/readme)
- [nomic-embed-text-v1.5](https://nexa.ai/nomic-ai/nomic-embed-text-v1.5/gguf-fp16/readme)
- [Column-Chart-LoRA](https://nexa.ai/DavidHandsome/Column-Chart-LoRA/gguf-fp16/readme)
- [Pie-Chart-LoRA](https://nexa.ai/DavidHandsome/Pie-Chart-LoRA/gguf-fp16/readme)

## Setup

Follow these steps to set up the project:

### 1. Create a New Conda Environment
Create and activate a new Conda environment to manage dependencies:

```
conda create --name pdf_chat python=3.10
conda activate pdf_chat
```

### 2. Install Requirements
install the necessary dependencies:

```
pip install -r requirements.txt
```

### 3. Install Nexa SDK

Follow docs [nexa-sdk](https://github.com/NexaAI/nexa-sdk) to install Nexa SDK.

### 4. Run Streamlit
Run the application using Streamlit:

```
streamlit run app.py
```