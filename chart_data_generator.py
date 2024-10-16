import json
from prompts import COLUMN_CHART_TEMPLATE, PIE_CHART_TEMPLATE
from inference_chart_model import generation_chart_data
import re
import logging

# Set the logging level for the openai logger to WARNING
logging.getLogger().setLevel(logging.ERROR)

def get_template_and_model_path(chart_type: str) -> str:
    if chart_type == "COLUMN_CLUSTERED" or chart_type == None: # Will also use template for pure text calling
        return COLUMN_CHART_TEMPLATE, "./models/column_chart_onnx"
    elif chart_type == "PIE":
        return PIE_CHART_TEMPLATE, "./models/pie_chart_onnx"
    else:
        raise ValueError(f"Invalid chart type: {chart_type}")

def clean_response(raw_response: str) -> dict:
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        return None

def execute_chart_generation(input_text, onnx_model, chart_type):
    template, local_model_path = get_template_and_model_path(chart_type)
    raw_response = generation_chart_data(text = input_text, onnx_model = onnx_model, chat_template = template)
    
    cleaned_response_dict = clean_response(raw_response)
    
    return cleaned_response_dict
    