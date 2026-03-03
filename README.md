# AI Image Generator

A Streamlit-based chat interface for generating images using the [Z-Image-Turbo](https://huggingface.co/spaces/mrfakename/Z-Image-Turbo) model on Hugging Face.

## Features

- Chat-like interface for entering image prompts
- Configurable image settings (width, height, inference steps)
- Chat history with generated images and seeds
- Fast image generation using Z-Image-Turbo

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd hugging
```

2. Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. (Optional) Configure your Hugging Face token:
```bash
cp .env.example .env
# Edit .env and add your HF token
```

## Configuration

The app uses environment variables for configuration. Copy `.env.example` to `.env` and configure as needed:

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | No | Your Hugging Face API token. Optional for public Spaces, required for private Spaces or higher rate limits. Get yours at https://huggingface.co/settings/tokens |

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Generating Images

1. Type a descriptive prompt in the chat input (e.g., "a beautiful sunset over mountains, photorealistic")
2. Adjust image settings in the sidebar if needed:
   - **Width/Height**: Image dimensions (512-1280px)
   - **Inference Steps**: Quality vs speed trade-off (4-20 steps)
3. Press Enter to generate the image

### Tips for Better Results

- Be descriptive with your prompts
- Include style keywords (e.g., "photorealistic", "anime", "oil painting", "digital art")
- Specify lighting and mood
- Mention specific details you want in the image

## Project Structure

```
hugging/
├── app.py           # Main Streamlit application
├── requirements.txt # Python dependencies
├── .env.example     # Environment variables template
├── .gitignore
└── README.md
```

## Notes

- The app includes SSL certificate bypass for corporate proxy environments
- Images are generated using the Z-Image-Turbo Space on Hugging Face
- HF token is optional for this public Space, but recommended for higher rate limits
- Never commit your `.env` file - it's already in `.gitignore`
