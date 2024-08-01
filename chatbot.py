import gradio
import query_data

gradio.ChatInterface(query_data.query_rag).launch()