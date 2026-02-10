# Bilingual Text Summarization API

A Flask-based REST API that provides intelligent text summarization for both Indonesian and English content. Supports both local transformer models and cloud-based AI services for flexible deployment options.

## Features

- **Bilingual Support**: Automatic language detection for Indonesian and English text
- **Dual Operation Modes**: 
  - Local models (T5 for Indonesian, BART for English)
  - Google Gemini API for cloud-based processing
- **Long Document Handling**: Recursive chunking with context preservation for texts of any length
- **Smart Summarization**: Sentence-boundary aware text splitting to maintain coherence
- **REST API**: Simple HTTP interface for easy integration into any application
- **Context Preservation**: Maintains narrative flow across document chunks

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Google Gemini API key for cloud mode

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd temp-skripsi
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` to configure your preferences:
```env
# Set to "true" for Gemini API, "false" for local models
USE_GEMINI=false

# Only required if USE_GEMINI=true
GEMINI_API_KEY=your_api_key_here
```

## Usage

### Starting the Server

```bash
python app.py
```

The server will start on `http://localhost:5000` in debug mode.

### API Endpoint

**POST** `/summarize`

**Request Body:**
```json
{
  "text": "Your long text to summarize here...",
  "previous_summary": "Optional: Previous summary for context"
}
```

**Response:**
```json
{
  "summary": "Generated summary of the text..."
}
```

**Parameters:**
- `text` (required): The text content to summarize
- `previous_summary` (optional): Context from a previous summarization to maintain continuity

### Example Usage

**Using curl:**
```bash
curl -X POST http://localhost:5000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your long text here..."
  }'
```

**Using Python requests:**
```python
import requests

response = requests.post(
    "http://localhost:5000/summarize",
    json={"text": "Your long text here..."}
)
print(response.json()["summary"])
```

**With context from previous summary:**
```python
# First chunk
response1 = requests.post(
    "http://localhost:5000/summarize",
    json={"text": "First part of a long document..."}
)
summary1 = response1.json()["summary"]

# Second chunk with context
response2 = requests.post(
    "http://localhost:5000/summarize",
    json={
        "text": "Second part of the document...",
        "previous_summary": summary1
    }
)
final_summary = response2.json()["summary"]
```

## Technical Details

### Models

**Local Mode:**
- **Indonesian**: `cahya/t5-base-indonesian-summarization-cased` (T5-based model)
- **English**: `facebook/bart-large-cnn` (BART-based model)

**Cloud Mode:**
- **Google Gemini 2.5 Flash**: Fast, efficient summarization for both languages

### How It Works

The API implements a sophisticated recursive summarization algorithm:

1. **Language Detection**: Automatically identifies whether the input text is Indonesian or English using the `langdetect` library

2. **Text Chunking**: Splits long texts into manageable chunks based on sentence boundaries:
   - Indonesian (T5): Maximum 512 tokens per chunk
   - English (BART): Maximum 1024 tokens per chunk
   - Respects sentence boundaries to maintain coherence

3. **Recursive Summarization**: 
   - First chunk is summarized independently
   - Each subsequent chunk is summarized with context from the previous summary
   - Context is passed using language-specific prefixes:
     - Indonesian: `"Ringkasan sebelumnya: {summary}. {chunk}"`
     - English: `"Previous summary: {summary}. {chunk}"`
   - Final output combines all context into a coherent summary

4. **Context Preservation**: Reserves 90 tokens for context integration to ensure previous summaries can be properly incorporated

### Token Limits

| Language   | Model | Max Tokens per Chunk | Context Reserve |
|------------|-------|---------------------|-----------------|
| Indonesian | T5    | 512                 | 90              |
| English    | BART  | 1024                | 90              |

When a previous summary is provided, the effective chunk size is reduced by the context reserve amount.

### Summary Generation Parameters

**Indonesian (T5):**
- Min length: 30 words
- Max length: 150 words
- Beam search: 4 beams
- Repetition penalty: 2.0
- Length penalty: 1.0
- No repeat n-gram size: 3

**English (BART):**
- Min length: 30 words
- Max length: 130 words
- Beam search: 4 beams
- Length penalty: 2.0
- No repeat n-gram size: 3

**Gemini API:**
- Target length: 30-130 words
- Language-specific prompts for optimal results

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `USE_GEMINI` | Use Gemini API instead of local models | `false` | No |
| `GEMINI_API_KEY` | Google Gemini API key | - | Yes (if USE_GEMINI=true) |

### Getting a Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create or select a project
3. Generate an API key
4. Add it to your `.env` file

## Performance Considerations

### Local Mode

**Advantages:**
- No API costs
- Works offline
- No rate limits
- Data stays on your server

**Considerations:**
- First run downloads models (~1-2GB total)
- Requires sufficient RAM (4GB+ recommended)
- GPU support available through PyTorch for faster processing
- Slower for very long documents

### Cloud Mode (Gemini API)

**Advantages:**
- Faster processing, especially for long documents
- No model download required
- Lower memory requirements
- Consistent performance

**Considerations:**
- Requires internet connection
- API costs apply
- Rate limits may apply based on your API tier
- Data is sent to Google's servers

## Error Handling

The API returns appropriate HTTP status codes:

- **200 OK**: Successful summarization
- **400 Bad Request**: Missing required `text` field in request body
- **500 Internal Server Error**: Model failure, API issues, or other server errors

Error responses include a JSON object with an `error` field:
```json
{
  "error": "Field 'text' is required"
}
```

## Dependencies

Core dependencies:
- **Flask**: Web framework for the REST API
- **transformers**: HuggingFace library for local models
- **torch**: PyTorch for model execution
- **langdetect**: Automatic language detection
- **google-genai**: Google Gemini API client
- **python-dotenv**: Environment variable management
- **sentencepiece**: Tokenization support
- **protobuf**: Protocol buffer support

See `requirements.txt` for the complete list.

## Project Structure

```
temp-skripsi/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── .env.example       # Environment variable template
├── .env               # Your local configuration (not in git)
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## Common Use Cases

### Summarizing News Articles
Perfect for condensing long news articles into digestible summaries while maintaining key facts.

### Academic Paper Summaries
Generate quick overviews of research papers (ensure proper citation of original work).

### Document Processing Pipelines
Integrate into larger document processing workflows for automated content extraction.

### Chatbot Integration
Use the API to provide quick summaries of long user messages or context.

## Troubleshooting

### Models Not Downloading
If models fail to download on first run, ensure you have:
- Stable internet connection
- Sufficient disk space (~2GB)
- Access to HuggingFace model hub

### Out of Memory Errors
If you encounter memory errors:
- Reduce input text length
- Use Gemini API mode instead of local models
- Close other applications to free up RAM
- Consider running on a machine with more memory

### Gemini API Errors
If using Gemini mode:
- Verify your API key is correct
- Check your API quota and rate limits
- Ensure you have billing enabled (if required)
- Verify internet connectivity

## Future Enhancements

Potential improvements for future versions:
- Support for additional languages
- Batch processing endpoint
- Customizable summary length
- Multiple summary style options
- WebSocket support for streaming summaries
- Model fine-tuning capabilities
- Summary quality metrics

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

[Add your name/info here]

## Acknowledgments

- HuggingFace for the Transformers library
- Google for the Gemini API
- Model creators: Cahya for Indonesian T5 model, Facebook for BART model
