curl http://localhost:8008/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3-8b",
        "prompt": "San Francisco is a",
        "max_tokens": 32,
        "temperature": 0
    }'