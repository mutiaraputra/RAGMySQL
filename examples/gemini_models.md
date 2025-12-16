# Google Gemini Models

## Available Models

### Gemini 1.5 Flash (Recommended)
- **Model ID**: `gemini-1.5-flash`
- **Best for**: Fast responses, cost-effective
- **Context Window**: 1M tokens
- **Pricing**: Lower cost than Pro

### Gemini 1.5 Pro
- **Model ID**: `gemini-1.5-pro`
- **Best for**: Complex reasoning, high quality
- **Context Window**: 2M tokens
- **Pricing**: Higher quality, higher cost

### Gemini 1.0 Pro
- **Model ID**: `gemini-1.0-pro`
- **Best for**: General purpose
- **Context Window**: 30K tokens

## Configuration Examples

### Fast & Cost-Effective (Flash)
```env
GEMINI__API_KEY=your-api-key
GEMINI__MODEL=gemini-1.5-flash
GEMINI__TEMPERATURE=0.7
GEMINI__MAX_OUTPUT_TOKENS=2048