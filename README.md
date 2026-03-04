# AI Image Generator

A Streamlit app that generates images with [Z-Image-Turbo](https://huggingface.co/spaces/mrfakename/Z-Image-Turbo) on Hugging Face and can enhance your prompts using [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) via the Hugging Face Inference API.

## Features

- **Prompt improvement** — Use the “Improve prompt” button to send your text to Qwen and get a richer, more detailed prompt (tuned for D&D-style places and characters).
- **Image generation** — Generate images from your prompt (or the improved one) with Z-Image-Turbo.
- **Configurable settings** — Width, height, and inference steps in the sidebar.
- **Chat history** — All prompts and generated images are kept in the session.
- **Blocked UI while loading** — Input and buttons are disabled until the improve or generate request finishes.

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

3. Configure your Hugging Face token. Create a `.env` file in the project root with:
   ```
   HF_TOKEN=your_token_here
   ```
   Get a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and ensure it has access to **Inference API** (for prompt improvement) and to use Spaces.

## Configuration

| Variable   | Required | Description |
|-----------|----------|-------------|
| `HF_TOKEN` | Yes     | Hugging Face API token. Used for the image Space (Z-Image-Turbo) and for prompt improvement (Qwen via Inference API). |

Use a `.env` file in the project root. Do not commit `.env`; it should be in `.gitignore`.

## Usage

Run the app:
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Workflow

1. **Enter a prompt** in the text area (e.g. “a dark tavern with adventurers”).
2. **(Optional)** Click **“✨ Improve prompt”** to have Qwen expand it into a more detailed prompt; the text area is replaced with the improved version.
3. Adjust **Width**, **Height**, and **Inference Steps** in the sidebar if you want.
4. Click **“🎨 Generate”** to create the image. The prompt is sent to Z-Image-Turbo and the result appears in the chat.

You can edit the (possibly improved) prompt before generating. Use **“Clear Chat History”** in the sidebar to reset the conversation.

### Tips

- Be descriptive; the improver works best with a clear starting idea.
- Use style keywords (e.g. “photorealistic”, “anime”, “oil painting”).
- Specify lighting and mood for more consistent results.

## Project Structure

```
ai-rpg-interface/
├── app.py           # Main Streamlit app (image gen + prompt improvement)
├── requirements.txt
├── .gitignore
└── README.md
```
Create a `.env` file with `HF_TOKEN` (do not commit it).

## Customizing the prompt improver

The system prompt for Qwen is defined in `app.py` as `IMPROVE_SYSTEM`. You can change it to adjust tone, focus (e.g. D&D, fantasy, realism), or output format. The placeholder `{prompt}` is replaced with the user’s current text before calling the model.

## Notes

- The app uses an SSL workaround for environments with strict or corporate proxies.
- Image generation: Z-Image-Turbo Space on Hugging Face.
- Prompt improvement: Qwen3.5-35B-A3B via Hugging Face Inference API (router with Novita provider).
