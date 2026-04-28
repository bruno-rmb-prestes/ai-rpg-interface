# AI Image Generator (Unified Local Backend Edition)

A Streamlit app that generates and edits images using local models, completely offline, while leveraging your local GPU resources. The codebase is split into **UI** (`app.py`), **business logic** (`services.py`), and a **unified dynamic backend server** (`unified_backend.py`) for a clear structure and easy maintenance.

## Features

- **Local Privacy** — All inferences (text generation, image generation, and image editing) happen locally on your hardware. No API keys or external services are strictly required.
- **Prompt improvement** — Send your text to a local LLM (like Mistral Nemo 12B running via Ollama/vLLM) to get a richer, more detailed prompt (tuned for D&D-style places and characters).
- **Image generation** — Generate images extremely fast using a local `Tongyi-MAI/Z-Image-Turbo` diffusers backend.
- **Image editing** — Perform instruction-based image edits using a local `FireRedTeam/FireRed-Image-Edit-1.1` backend.
- **Configurable settings** — Width, height, and inference steps in the sidebar.
- **Chat history** — All prompts and generated images are kept in the session.

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd ai-rpg-interface
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Configuration

| Variable          | Required | Description |
|-------------------|----------|-------------|
| `LOCAL_API_BASE`  | No       | The base URL for your local LLM server. Defaults to `http://127.0.0.1:11434/v1` (Ollama's default). |
| `LOCAL_MODEL`     | No       | The name of the LLM model to use. Defaults to `mistral-nemo`. |

Create a `.env` file in the project root to set these variables if you are using custom ports or model names.

## Usage

Running this application requires you to start the backend servers before starting the Streamlit interface. This architecture allows you to distribute the heavy models across multiple GPUs.

### 1. Install and Start the Local LLM (Ollama)
Start your preferred OpenAI-compatible local inference engine (Ollama, vLLM, LMStudio, etc.). 

If you don't have Ollama installed yet, you can install it using their official Linux script (which completely avoids `snap`):
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Once Ollama is installed, run the following command. The very first time you run this, it will automatically download the Mistral Nemo model (~7GB) and then start the server.
```bash
ollama run mistral-nemo
```

### 2. Start the Unified Image Backend
Open a new terminal, activate your virtual environment, and start the unified generation and editing server. This server dynamically swaps the image models in and out of GPU 1's VRAM as needed, while keeping them loaded in your standard System RAM for fast switching.
*(Tip: Use `CUDA_VISIBLE_DEVICES` to target a specific GPU if you have a multi-GPU setup. In this case, we isolate the image models to GPU 1).*
```bash
CUDA_VISIBLE_DEVICES=1 python unified_backend.py
```
*Note: The very first run will download both model weights from Hugging Face.*

### 3. Start the Streamlit UI
In a final terminal, run the application:
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Workflow

1. **Enter a prompt** in the text area (e.g. “a dark tavern with adventurers”).
2. **(Optional)** Click **“✨ Improve prompt”** to have your local LLM expand it into a more detailed prompt.
3. Adjust **Width**, **Height**, and **Inference Steps** in the sidebar.
4. Click **“🎨 Generate”** to create the image.
5. **(Optional)** Click the **“✏️ Edit”** button on a generated image, enter an instruction (e.g. "make it nighttime"), and submit to modify the image using the FireRed backend.

## Project Structure

```
ai-rpg-interface/
├── app.py                 # Streamlit UI: layout, session state, chat, sidebar, event handlers
├── services.py            # Business logic: Local OpenAI clients and Gradio client routes
├── unified_backend.py     # Local Dynamic FastAPI/Gradio server for Z-Image-Turbo and FireRed
├── qwenimage/             # Model architectures required by FireRed-Image-Edit
├── requirements.txt
├── .gitignore
└── README.md
```

## Customizing the prompt improver

The system prompt for the LLM is defined in `services.py` as `IMPROVE_SYSTEM`. You can change it to adjust tone, focus (e.g. D&D, fantasy, realism), or output format. The placeholder `{prompt}` is replaced with the user’s current text before calling the model.
