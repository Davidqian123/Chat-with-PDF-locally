import streamlit as st
from langchain_chroma import Chroma
from nexa_embedding import NexaEmbeddings
import os
import json
from presentation_generator import PresentationGenerator
from chart_data_generator import execute_chart_generation
from chart_generator import ChartGenerator
from PIL import Image
from nexa.gguf import NexaTextInference
from prompts import DECISION_MAKING_TEMPLATE
from build_db import create_chroma_db

avatar_path = "files/avatar.jpeg"
persist_directory = "./chroma_db"

@st.cache_resource
def load_models():
    # Load the base model
    chat_model = NexaTextInference(model_path="gemma-2-2b-instruct:fp16")
    print("Chat model loaded successfully!")

    # Load the decision model
    decision_model = NexaTextInference(model_path="DavidHandsome/Octopus-v2-PDF:gguf-q4_K_M")
    print("Decision model loaded successfully!")

    return chat_model, decision_model

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
    embeddings = NexaEmbeddings(model_path="nomic")
    local_db = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )
    return local_db.as_retriever()


def retrieve_documents(retriever, query):
    docs = retriever.get_relevant_documents(query)
    for doc in docs:
        print(doc.page_content)
    return [doc.page_content for doc in docs]


def call_pdf_qa(query, context, chat_model):
    system_prompt = (
        "You are a QA assistant. Based on the following context, answer the question using bullet points and include necessary data.\n\n"
        f"Context:\n{context}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    try:
        stream = chat_model.create_chat_completion(
            messages=messages,
            max_tokens=2048,
            stream=True
        )
        return stream
    except Exception as e:
        st.error(f"An error occurred while calling QA: {str(e)}")
        return None


def query_information_with_retrieval(prompt, retriever, chat_model):
    with st.chat_message("assistant", avatar=avatar_path):
        with st.spinner("Generating..."):
            # Retrieve documents and prepare context
            retrieved_docs = retrieve_documents(retriever, prompt)
            context = "\n\n".join(retrieved_docs)
            
            # Get the response stream
            stream = call_pdf_qa(prompt, context, chat_model)
            if stream is None:
                return
            
            # Create a placeholder for the streaming response
            response_placeholder = st.empty()
            full_response = ""
        
            # Stream the response and update the placeholder
            for chunk in stream:
                content = chunk["choices"][0]["delta"].get("content", "")
                full_response += content
                response_placeholder.markdown(full_response)
            
            # Update session state after streaming is complete
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "avatar": avatar_path
            })
            st.session_state.last_response = full_response

def irrelevant_function(prompt, chat_model):
    with st.chat_message("assistant", avatar=avatar_path):
        with st.spinner("Generating..."):
            # Get the response stream
            stream = call_common_qa(prompt, chat_model)
            if stream is None:
                return
            
            # Create a placeholder for the streaming response
            response_placeholder = st.empty()
            full_response = ""

            # Stream the response and update the placeholder
            for chunk in stream:
                content = chunk["choices"][0]["delta"].get("content", "")
                full_response += content
                response_placeholder.markdown(full_response)

            # Update session state after streaming is complete
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "avatar": avatar_path
            })
            st.session_state.last_response = full_response

def call_common_qa(prompt, chat_model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    try:
        stream = chat_model.create_chat_completion(
            messages=messages,
            max_tokens=2048,
            stream=True
        )
        return stream
    except Exception as e:
        st.error(f"An error occurred while calling QA: {str(e)}")
        return None

def call_title_text_summary(query, chat_model):
    system_prompt = (
        "You are a helpful assistant that generates a brief, relevant title for the given query in 10 words or less."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    try:
        response = chat_model.create_chat_completion(
            messages=messages,
            max_tokens=2048,
        )
        return (response["choices"][0]["message"]["content"])
    
    except Exception as e:
        st.error(f"An error occurred while generating title: {str(e)}")
        return None

def call_main_text_summary(query, chat_model):
    system_prompt = (
        "You are a helpful assistant that generates a short summary for the given query."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    try:
        response = chat_model.create_chat_completion(
            messages=messages,
            max_tokens=2048,
        )
        return (response["choices"][0]["message"]["content"])
    
    except Exception as e:
        st.error(f"An error occurred while generating main text summary: {str(e)}")
        return None

def generate_chart(chart_type):
    """Helper function to generate a chart."""
    result = execute_chart_generation(st.session_state.last_response, chart_type)

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

def classify_user_intent(prompt, decision_model):
    if decision_model is None:
        st.error("Decision model is not loaded. Please refresh the page or contact support.")
        return None

    formatted_prompt = DECISION_MAKING_TEMPLATE.format(input=prompt)
    output = decision_model.create_completion(formatted_prompt, stop=["<nexa_end>"])

    return output["choices"][0]["text"].strip()



def add_to_slides(chart_type=None):
    if st.session_state.last_response:
        with st.spinner("Generating chart and adding to slides..."):
            slide_data = generate_chart(chart_type)

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

def add_to_text_slides(chat_model):
    if st.session_state.last_response:
        with st.spinner("Generating text slide..."):
            title_text = call_title_text_summary(st.session_state.last_response, chat_model)
            main_text = call_main_text_summary(st.session_state.last_response, chat_model)

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

def generate_presentation():
    with st.spinner("Generating presentation..."):
        generator = PresentationGenerator("files/ppt_template.pptx")
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
    img = Image.open("files/avatar.jpeg")
    
    # Ensure the sidebar is always expanded
    st.set_page_config(
        page_title="Nexa AI PDF Chatbot",
        page_icon=img,
        layout="wide",  # This ensures the sidebar is always expanded
        initial_sidebar_state="expanded"  # The sidebar cannot be collapsed
    )

    # Load the models once
    chat_model, decision_model = load_models()

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
                # Save the uploaded file temporarily
                temp_file_path = os.path.join("temp", uploaded_file.name)
                os.makedirs("temp", exist_ok=True)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                # Create the Chroma database
                db = create_chroma_db(pdf_path=temp_file_path)
                # Clean up the temporary file
                os.remove(temp_file_path)

            st.session_state.file_uploaded = True
            st.session_state.pdf_filename = uploaded_file.name
            st.success("File processed successfully!")
            st.rerun()

    if prompt := st.chat_input("What would you like to know?"):
        st.chat_message("user").markdown(prompt)
        intent = classify_user_intent(prompt, decision_model)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        print("intent", intent)
        if intent == "<nexa_0>":        # query_with_pdf 
            query_information_with_retrieval(prompt, retriever, chat_model)
        elif intent == "<nexa_1>":      # generate_slide_text_content 
            add_to_text_slides(chat_model)
        elif intent == "<nexa_2>":      # generate_slide_column_chart 
            add_to_slides("COLUMN_CLUSTERED")
        elif intent == "<nexa_4>":      # generate_slide_pie_chart
            add_to_slides("PIE")
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
            irrelevant_function(prompt, chat_model)


if __name__ == "__main__":
    main()
