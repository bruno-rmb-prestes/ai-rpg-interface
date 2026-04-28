from gradio_client import Client
import traceback

c = Client("http://127.0.0.1:7860/")
try:
    res = c.predict(
        prompt="A magical forest",
        height=1024,
        width=1024,
        num_inference_steps=9,
        randomize_seed=False,
        seed=42,
        api_name="/generate_image"
    )
    print("Success")
except Exception as e:
    print("Error:")
    traceback.print_exc()
