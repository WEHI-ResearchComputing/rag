import gradio
import query_data
import socket

gradio.ChatInterface(query_data.query_rag).launch(ssl_verify=False, server_name=socket.gethostname(), root_path=f"/node/{socket.gethostname()}/7860")