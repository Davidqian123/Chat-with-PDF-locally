import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os
import json
import ollama
from presentation_generator import PresentationGenerator
from chart_data_generator import execute_chart_generation
from chart_generator import ChartGenerator
from intent_classifier import classify_user_intent
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import onnxruntime_genai as og

avatar_path = "files/avatar.png"

# Load the base model and tokenizer once during initialization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_NAME = "./models/octopus-v2-base"
ONNX_MODEL_COLUMN = "./models/column_chart_onnx"
ONNX_MODEL_PIE = "./models/pie_chart_onnx"

@st.cache_resource
def load_octopus_model():
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16
    ).to(DEVICE)
    print("Base model loaded successfully!")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("Tokenizer loaded successfully!")

    # Load the chart models
    column_chart_model = og.Model(ONNX_MODEL_COLUMN)
    print("Column chart model loaded successfully!")
    pie_chart_model = og.Model(ONNX_MODEL_PIE)
    print("Pie chart model loaded successfully!")

    return base_model, tokenizer, column_chart_model, pie_chart_model

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ppt_generated" not in st.session_state:
        st.session_state.ppt_generated = False
    if "last_response" not in st.session_state:
        st.session_state.last_response = ""
    if "file_uploaded" not in st.session_state:
        st.session_state.file_uploaded = False
    if "pdf_filename" not in st.session_state:
        st.session_state.pdf_filename = ""

def setup_retriever():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    local_db = Chroma(
        persist_directory="./chroma_db", embedding_function=embeddings
    )
    return local_db.as_retriever()


def retrieve_documents(retriever, query):
    docs = retriever.get_relevant_documents(query)
    for doc in docs:
        print(doc.page_content)
    return [doc.page_content for doc in docs]

def call_pdf_qa(query, context):
    system_prompt = (
        "You are a QA assistant. Based on the following context, answer the question using bullet points and include necessary data.\n\n"
        f"Context:\n{context}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    
    print("messages", messages)

    try:
        stream = ollama.chat(
            model="llama3.2",
            messages=messages,
            stream=True
        )
        return stream
    except Exception as e:
        st.error(f"An error occurred while calling QA: {str(e)}")
        return None

def query_information_with_retrieval(prompt, retriever):
    with st.chat_message("assistant", avatar=avatar_path):
        with st.spinner("Generating..."):
            # Retrieve documents and prepare context
            retrieved_docs = retrieve_documents(retriever, prompt)
            context = "\n\n".join(retrieved_docs)
            
            # Get the response stream
            stream = call_pdf_qa(prompt, context)
            if stream is None:
                return
            
            # Create a placeholder for the streaming response
            response_placeholder = st.empty()
            full_response = ""
        
            # Stream the response and update the placeholder
            for chunk in stream:
                content = chunk['message']['content']
                full_response += content
                response_placeholder.markdown(full_response)
            
            # Update session state after streaming is complete
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "avatar": avatar_path
            })
            st.session_state.last_response = full_response


def irrelevant_function(prompt):
    with st.chat_message("assistant", avatar=avatar_path):
        with st.spinner("Generating..."):
            # Get the response stream
            stream = call_common_qa(prompt)
            if stream is None:
                return
            
            # Create a placeholder for the streaming response
            response_placeholder = st.empty()
            full_response = ""
        
            # Stream the response and update the placeholder
            for chunk in stream:
                content = chunk['message']['content']
                full_response += content
                response_placeholder.markdown(full_response)
            
            # Update session state after streaming is complete
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "avatar": avatar_path
            })
            st.session_state.last_response = full_response

def call_common_qa(prompt):
    messages = [
        {"role": "user", "content": prompt},
    ]
    try:
        stream = ollama.chat(
            model="llama3.2",
            messages=messages,
            stream=True
        )
        return stream
    except Exception as e:
        st.error(f"An error occurred while calling QA: {str(e)}")
        return None

def call_title_text_summary(query):
    system_prompt = (
        "You are a helpful assistant that generates a brief, relevant title for the given query in 10 words or less."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=messages,
        )
        return (response['message']['content'])
    
    except Exception as e:
        st.error(f"An error occurred while generating title: {str(e)}")
        return None

def call_main_text_summary(query):
    system_prompt = (
        "You are a helpful assistant that generates a short summary for the given query."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=messages,
        )
        return (response['message']['content'])
    
    except Exception as e:
        st.error(f"An error occurred while generating main text summary: {str(e)}")
        return None

def generate_chart(onnx_model, chart_type):
    """Helper function to generate a chart."""
    result = execute_chart_generation(st.session_state.last_response, onnx_model, chart_type)

    if result is None:
        st.warning("No valid json data was generated from the last response.")
        return None
    
    chart_generator = ChartGenerator()

    slide_data = {
        "title_text": result["text"]["title_text"],
        "main_text": result["text"]["main_text"],
    }

    if chart_type and "chart_data" in result and result["chart_data"]:
        image_path = chart_generator.plot_chart(result["chart_data"])
        slide_data["image_path"] = image_path

    return slide_data


def add_slides_data_to_file(slide_data):
    """Helper function to add slide data to a JSON file."""
    slides_file = "slides_data.json"

    if os.path.exists(slides_file):
        with open(slides_file, "r") as f:
            slides_data = json.load(f)
    else:
        slides_data = []

    slides_data.append(slide_data)

    with open(slides_file, "w") as f:
        json.dump(slides_data, f, indent=2)


def prepare_success_message(slide_data):
    """Helper function to prepare the success message."""
    success_message = "Slide added successfully!\n\n"
    success_message += f"Title: {slide_data['title_text']}\n\n"
    success_message += f"Content: {slide_data['main_text']}\n\n"

    if "image_path" in slide_data:
        success_message += "A chart was generated for this slide."
    else:
        success_message += "No chart was generated for this slide."

    return success_message


def add_to_slides(onnx_model, chart_type=None):
    if st.session_state.last_response:
        with st.spinner("Generating chart and adding to slides..."):
            slide_data = generate_chart(onnx_model, chart_type)

            if slide_data is None:
                return
            
            add_slides_data_to_file(slide_data)

        # Prepare success message using the new helper function
        success_message = prepare_success_message(slide_data)

        # Add success message to chat history, including the image path
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": success_message,
                "avatar": avatar_path,
                "image_path": slide_data.get("image_path"),
            }
        )

        st.success("Chart slide added successfully!")

        # Force a rerun to display the new message
        st.rerun()

        st.success("Slide added successfully!")

def add_to_text_slides():
    if st.session_state.last_response:
        with st.spinner("Generating text slide..."):
            title_text = call_title_text_summary(st.session_state.last_response)
            main_text = call_main_text_summary(st.session_state.last_response)

            slide_data = {
                "title_text": title_text,
                "main_text": main_text,
            }

            add_slides_data_to_file(slide_data)

        # Prepare success message using the new helper function
        success_message = prepare_success_message(slide_data)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": success_message,
                "avatar": avatar_path,
            }
        )

        st.success("Text slide added successfully!")

        # Force a rerun to display the new message
        st.rerun()

        st.success("Slide added successfully!")

def generate_presentation():
    with st.spinner("Generating presentation..."):
        generator = PresentationGenerator("files/amd_ppt_template.pptx")
        generator.generate_presentation(
            "slides_data.json",
            "presentation_with_charts_and_text.pptx",
        )
    st.session_state.ppt_generated = True
    st.success("Presentation generated successfully!")


def display_download_button():
    with open("presentation_with_charts_and_text.pptx", "rb") as file:
        st.download_button(
            label="Download Presentation",
            data=file,
            file_name="presentation_with_charts_and_text.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )


# Main Streamlit App
def main():
    img = Image.open("files/avatar.png")
    
    # Ensure the sidebar is always expanded
    st.set_page_config(
        page_title="Nexa AI PDF Chatbot",
        page_icon=img,
        layout="wide",  # This ensures the sidebar is always expanded
        initial_sidebar_state="expanded"  # The sidebar cannot be collapsed
    )

    # Load the base model and tokenizer once
    base_model, tokenizer, column_chart_model, pie_chart_model = load_octopus_model()

    # Add sidebar with Nexa logo and descriptions
    st.sidebar.image("files/nexa_logo.png", use_column_width=True)  # Adjust the logo path
    st.sidebar.markdown("## Nexa AI's solution")
    st.sidebar.markdown("""
    - Comprehensive On-Device AI Solutions
    - Open-Source AI Model Hub
    - On-Device AI Developer Community
    - SDK for Multi-Modal AI Integration
    """)

    st.title("Nexa AI PDF Chatbot")
    initialize_session_state()
    retriever = setup_retriever()

    # Display the chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])
            if message.get("image_path"):
                st.image(message["image_path"], caption="Generated Chart")

    # Display uploaded PDF information
    if st.session_state.file_uploaded:
        st.info(f"PDF uploaded: {st.session_state.pdf_filename}")

    # File upload area
    if not st.session_state.file_uploaded:
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            with st.spinner("Processing the PDF file..."):
                time.sleep(10)
            st.session_state.file_uploaded = True
            st.session_state.pdf_filename = uploaded_file.name
            st.success("File processed successfully!")
            st.rerun()

    if prompt := st.chat_input("What would you like to know?"):
        # Pass the base_model and tokenizer to classify_user_intent
        st.chat_message("user").markdown(prompt)
        intent = classify_user_intent(prompt, base_model, tokenizer)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        print("intent", intent)
        if intent == "<nexa_0>":        # query_with_pdf 
            query_information_with_retrieval(prompt, retriever)
        elif intent == "<nexa_1>":      # generate_slide_text_content 
            add_to_text_slides()
        elif intent == "<nexa_2>":      # generate_slide_column_chart 
            add_to_slides(column_chart_model, "COLUMN_CLUSTERED")
        elif intent == "<nexa_4>":      # generate_slide_pie_chart
            add_to_slides(pie_chart_model, "PIE")
        elif intent == "<nexa_5>":      # create_presentation 
            generate_presentation()
        elif intent == "<nexa_6>":      # download_presentation
            if st.session_state.ppt_generated:
                display_download_button()
            else:
                st.warning(
                    "No presentation has been generated yet. Please create a presentation first."
                )
        else:                           # irrelevant_function 
            irrelevant_function(prompt)


if __name__ == "__main__":
    main()
