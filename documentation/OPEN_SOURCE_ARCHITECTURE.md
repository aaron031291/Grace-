# Grace AI - 100% Open Source Architecture

## Overview

Grace AI has been redesigned to be completely independent of proprietary APIs. All embedding and AI functionality now uses open-source alternatives.

## Embedding Solutions

### Primary: Sentence Transformers (HuggingFace)
- **Library**: `sentence-transformers`
- **Default Model**: `all-MiniLM-L6-v2`
- **Features**: 
  - 384-dimensional embeddings
  - Fast inference
  - No API keys required
  - Runs completely offline

### Alternative: Local Transformers
- **Library**: `transformers` + `torch`
- **Default Model**: `bert-base-uncased`
- **Features**:
  - Fully customizable
  - No external dependencies
  - Complete control over model selection

## Removed Dependencies

### OpenAI
- ❌ `openai` package removed
- ❌ API key configuration removed
- ❌ All OpenAI embedding calls removed
- ✅ Replaced with sentence-transformers

### Anthropic
- ❌ `anthropic` package removed
- ❌ Claude API references removed
- ✅ Not used in Grace system

## Configuration Changes

### Before (Proprietary)
```python
EMBEDDING_PROVIDER=openai
EMBEDDING_OPENAI_API_KEY=sk-...
```

### After (Open Source)
```python
EMBEDDING_PROVIDER=huggingface
EMBEDDING_HUGGINGFACE_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Benefits

1. **Privacy**: All data processing happens locally
2. **Cost**: No API fees or usage limits
3. **Reliability**: No dependency on external services
4. **Speed**: Local inference is faster than API calls
5. **Control**: Complete control over models and versions

## Migration Guide

If you were using OpenAI embeddings:

```bash
# 1. Remove old configuration
unset EMBEDDING_OPENAI_API_KEY

# 2. Update .env
EMBEDDING_PROVIDER=huggingface

# 3. Reinstall dependencies
pip install -r requirements.txt

# 4. Regenerate embeddings
python scripts/regenerate_embeddings.py
```

## Performance Comparison

| Feature | OpenAI | HuggingFace | Local |
|---------|--------|-------------|-------|
| Cost | $$$$ | Free | Free |
| Privacy | Low | High | High |
| Speed | Medium | Fast | Fast |
| Offline | No | Yes | Yes |
| Customizable | No | Medium | High |

## Recommended Models

### Small & Fast (Default)
- `sentence-transformers/all-MiniLM-L6-v2` (384d)
- Best for: General use, fast inference

### Medium & Accurate
- `sentence-transformers/all-mpnet-base-v2` (768d)
- Best for: Higher quality embeddings

### Large & Best Quality
- `sentence-transformers/all-distilroberta-v1` (768d)
- Best for: Maximum accuracy

## Support

All Grace functionality now works completely offline with no external API dependencies.
