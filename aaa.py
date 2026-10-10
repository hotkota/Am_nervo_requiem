from llama_cpp import Llama

# Укажите путь к вашей модели в формате GGUF
llm = Llama(model_path="./Qwen3-1.7B-Q8_0.gguf")
output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32)
print(output['choices'][0]['text'])