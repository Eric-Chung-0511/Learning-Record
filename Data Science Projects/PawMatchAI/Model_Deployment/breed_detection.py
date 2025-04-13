import re
import gradio as gr
from PIL import Image

def create_detection_tab(predict_fn, example_images):
    with gr.TabItem("Breed Detection"):
        gr.HTML("""
            <div style='
                text-align: center;
                padding: 20px 0;
                margin: 15px 0;
                background: linear-gradient(to right, rgba(66, 153, 225, 0.1), rgba(72, 187, 120, 0.1));
                border-radius: 10px;
            '>
                <p style='
                    font-size: 1.2em;
                    margin: 0;
                    padding: 0 20px;
                    line-height: 1.5;
                    background: linear-gradient(90deg, #4299e1, #48bb78);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-weight: 600;
                '>
                    Upload a picture of a dog, and the model will predict its breed and provide detailed information!
                </p>
                <p style='
                    font-size: 0.9em;
                    color: #666;
                    margin-top: 8px;
                    padding: 0 20px;
                '>
                    Note: The model's predictions may not always be 100% accurate, and it is recommended to use the results as a reference.
                </p>
            </div>
        """)
        
        with gr.Row():
            input_image = gr.Image(label="Upload a dog image", type="pil")
            output_image = gr.Image(label="Annotated Image")

        output = gr.HTML(label="Prediction Results")
        initial_state = gr.State()

        input_image.change(
            predict_fn,
            inputs=input_image,
            outputs=[output, output_image, initial_state]
        )

        gr.Examples(
            examples=example_images,
            inputs=input_image
        )

    return {
        'input_image': input_image,
        'output_image': output_image,
        'output': output,
        'initial_state': initial_state
    }
