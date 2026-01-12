response = requests.post() 
data = response.json() # start with this

# Token Count Extractions
input_tokens = data.get('usageMetadata', {}).get('promptTokenCount', 0)
output_tokens = data.get('usageMetadata', {}).get('candidatesTokenCount', 0)
total_tokens = data.get('usageMetadata', {}).get('totalTokenCount', 0)
thinking_tokens = data.get('usageMetadata', {}).get('thoughtsTokenCount', 0)

# Structured log entry for LLM token usage
token_log_entry = {
    "model": self.model_id,
    "app_id": self.app_id,
    "token_usage": {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "total_tokens": total_tokens
    }
}
logger.info(json.dumps(log_entry))

if response == 200:
  return data['complet'] # replace response.json() with data in this line



print(f"Input: {input_tokens} | Output: {output_tokens} | Thinking_tokens {thinking_tokens}| Total: {total_tokens}")






fields token_usage.total_tokens
| stats sum(token_usage.total_tokens) as TotalTokensByDay by bin(1d)


stats sum(token_usage.input_tokens) as Input,
      sum(token_usage.output_tokens) as Output,
      sum(token_usage.thinking_tokens) as Thinking
by model

