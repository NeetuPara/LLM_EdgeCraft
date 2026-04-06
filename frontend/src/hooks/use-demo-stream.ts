// Simulates streaming LLM responses character-by-character for demo mode

const RESPONSES: Record<string, string> = {
  default: `I'm EdgeCraft's demo assistant — I'm simulating a fine-tuned language model response.

In the real application, this response would come from your trained model loaded via the Unsloth Studio backend, streamed token-by-token over a server-sent events connection.

You can fine-tune any supported model (Llama 3, Qwen 2.5, Gemma 3, Mistral, DeepSeek, etc.) and chat with it right here — comparing your fine-tuned LoRA adapter against the base model side-by-side.`,

  qlora: `**QLoRA** (Quantized Low-Rank Adaptation) combines two memory-saving techniques:

**1. 4-bit NF4 Quantization**
The base model weights are stored in 4-bit NormalFloat4 format via bitsandbytes, reducing VRAM by ~75% compared to FP16. A 7B model needs ~4GB instead of ~14GB.

**2. LoRA Adapters**
Instead of updating all model parameters, small low-rank matrices (\`A\` and \`B\`) are added to specific layers. The update is: \`W + s·A@B\`, where \`s = alpha/r\`.

**Key insight:** The 4-bit weights are frozen. Only the small LoRA matrices (~0.1% of total params) are trained in FP16/BF16. This gives:
- 4× less VRAM than full fine-tuning
- Near-identical final model quality
- Adapter weights are only ~50–200MB to store

\`\`\`python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    load_in_4bit=True,  # QLoRA
    max_seq_length=2048,
)
model = FastLanguageModel.get_peft_model(model, r=64, lora_alpha=16)
\`\`\``,

  hello: `Hello! I'm a fine-tuned language model running via Unsloth.

I've been trained on custom instruction data, which means I can follow specific formats and styles tailored to your use case. How can I help you today?`,

  python: `\`\`\`python
def fibonacci(n: int) -> list[int]:
    """Generate Fibonacci sequence up to n terms."""
    if n <= 0:
        return []
    if n == 1:
        return [0]

    sequence = [0, 1]
    while len(sequence) < n:
        sequence.append(sequence[-1] + sequence[-2])

    return sequence

# Example usage
print(fibonacci(10))
# Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
\`\`\`

The function builds the sequence iteratively, which is more memory-efficient than recursion for large \`n\`. Time complexity is O(n), space complexity is O(n).`,

  explain: `Sure! Here's a clear explanation:

**Neural networks** are computational systems loosely inspired by the human brain. They consist of:

1. **Neurons (nodes)** — simple mathematical functions that take inputs, apply weights, add a bias, and pass the result through an activation function

2. **Layers** — neurons are organized in layers:
   - *Input layer*: receives raw data
   - *Hidden layers*: learn intermediate representations
   - *Output layer*: produces predictions

3. **Weights** — the learnable parameters. Training adjusts weights to minimize prediction error via backpropagation + gradient descent

**The key idea:** by stacking many layers, networks can learn hierarchical features automatically — edges → shapes → objects in vision, or tokens → grammar → meaning in language.`,

  instruct: `I'm an instruction-following model, which means I'm designed to:

- **Follow explicit instructions** clearly and precisely
- **Complete tasks** across many domains: writing, coding, analysis, Q&A
- **Maintain context** across a multi-turn conversation
- **Decline harmful requests** and stay within safe boundaries

My fine-tuning used supervised learning on human-written (instruction, response) pairs, teaching me to produce helpful, harmless, and honest responses.`,
}

function pickResponse(message: string): string {
  const lower = message.toLowerCase()
  if (lower.includes('qlora') || lower.includes('lora') || lower.includes('quantiz'))
    return RESPONSES.qlora
  if (lower.includes('hello') || lower.includes('hi ') || lower === 'hi' || lower.includes('hey'))
    return RESPONSES.hello
  if (lower.includes('python') || lower.includes('code') || lower.includes('fibonacci') || lower.includes('function'))
    return RESPONSES.python
  if (lower.includes('neural') || lower.includes('explain') || lower.includes('what is') || lower.includes('how does'))
    return RESPONSES.explain
  if (lower.includes('instruct') || lower.includes('fine-tun') || lower.includes('trained'))
    return RESPONSES.instruct
  return RESPONSES.default
}

export async function streamDemoResponse(
  userMessage: string,
  onChunk: (chunk: string) => void,
  onDone: () => void,
  signal?: AbortSignal,
): Promise<void> {
  // Simulate "thinking" delay
  await new Promise(r => setTimeout(r, 400 + Math.random() * 300))

  if (signal?.aborted) return

  const response = pickResponse(userMessage)
  const chunkSize = 4
  const baseDelay = 18

  for (let i = 0; i < response.length; i += chunkSize) {
    if (signal?.aborted) return
    const chunk = response.slice(i, i + chunkSize)
    onChunk(chunk)
    // Variable delay for realism: slower on punctuation
    const char = response[i]
    const delay = ['.', '!', '?', '\n'].includes(char)
      ? baseDelay * 4
      : [',', ';', ':'].includes(char)
        ? baseDelay * 2
        : baseDelay
    await new Promise(r => setTimeout(r, delay))
  }

  onDone()
}
