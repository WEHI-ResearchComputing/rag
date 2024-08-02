import gradio
import query_data
import socket
import populate_database
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, help="The port that the app will run on", default=7860)
parser.add_argument("--host", type=str, help="The host the app is running on", default=socket.gethostname())
args = parser.parse_args()

with gradio.Blocks() as demo:
    with gradio.Group():
        with gradio.Row():
            data_path = gradio.Textbox(label="Data Path")
            embedding_model = gradio.Dropdown(
                ["mxbai-embed-large", "nomic-embed-text", "snowflake-arctic-embed:22m"], 
                label="Embedding Model"
            )
            add_data_btn = gradio.Button("Add Data to Database")
        add_data_output = gradio.Textbox(label="Add Data Output")
    add_data_btn.click(
        fn=populate_database.add_data_to_database, 
        inputs=[data_path, embedding_model], 
        outputs=add_data_output
    )
    with gradio.Group():
        llm_model = gradio.Dropdown(
            ["mistral", "mistral-nemo", "phi3:mini", "phi3:medium"],
            label="LLM"
        )

        gradio.ChatInterface(query_data.query_rag, additional_inputs=[llm_model, embedding_model])

demo.launch(
    ssl_verify=False, 
    server_name=args.host, 
    root_path=f"/node/{args.host}/{args.port}",
    server_port=args.port
)