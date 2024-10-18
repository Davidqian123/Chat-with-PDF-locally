# Nexa AI PDF Chatbot

## Introduction

This demo showcases <strong>Octopus v2</strong>, the first on-device LLM to achieve GPT-4o-level function calling accuracy for AI agents, while operating at <strong>four times the speed</strong> of GPT-4o. We implemented local RAG techniques to transform PDFs into a vector database, enabling seamless querying and answer retrieval. The fine-tuned Octopus v2 model not only summarizes the retrieved answers but also determines and executes the appropriate function calls based on the input context. Additionally, it can generate visualizations, such as charts, based on the generated responses. The demo leverages <strong>AMD Ryzen AI 300 Series Processors</strong> with <strong>CPU + GPU mixed acceleration</strong>, ensuring optimal performance and efficiency.

## Setup

Follow these steps to set up the project:

### 1. Download and Unzip

Download the project from the source and unzip it.

### 2. Create a New Conda Environment
Create and activate a new Conda environment to manage dependencies:

```
conda create --name pdf_chat python=3.10
conda activate pdf_chat
```

### 3. Install Requirements
install the necessary dependencies:

```
pip install -r requirements.txt
```

### 4. Run Streamlit
Run the application using Streamlit:

```
streamlit run app.py
```

## Design Questions

<strong>Q1 Text:</strong>
What is the pdf about? 
Introduce 2024 AsusZenbook S16 according to pdf content
Create a text slide for this

<strong>Q2 Column Chart:</strong>
What is the gaming performance of the laptopaccording to pdf content
Create a column chart showing dataå

<strong>Q3 Column Chart:</strong> 
What is the 3DMark benchmark performance according to pdf content
Generate a column chart showing data

<strong>Q4 Pie Chart:</strong> 
What is AMD's net revenue in Data Center, Client, Gaming and Embedded according to pdf content
Generate a pie chart showing data

<strong>Q6 Generate ppt:</strong> 
Create a ppt

<strong>Q7 Download ppt:</strong> 
Download PPT
