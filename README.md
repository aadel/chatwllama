# ChatWllama

A LangChain.js integration for [Wllama](https://github.com/jina-ai/wllama) - a local LLM inference library using WebAssembly.

## Installation

This package uses Vite+ toolchain.

### Requirements

In order to use this module:

1. Wllama package should be installed
2. Pre-built wllama.wasm according to documentation on Wllama project

Add the package to dependencies:

```bash
vp add chatwllama
```

The package requires:

- Node.js >= 20
- `@wllama/wllama` - the Wllama WASM module
- `@langchain/core` - LangChain core utilities

## Chat Models

### ChatWllama

`ChatWllama` is a chat model that connects to Wllama models for local LLM inference. It provides:

- Support for streaming and non-streaming responses
- Multiple sampling parameters (temperature, top_p, top_k, min_p, presence_penalty, ...)
- Tool call support

#### Basic Usage

```typescript
import { ChatWllama } from "chatwllama";

const model = new ChatWllama(
  { repo: "some/model", file: "model.Q4_0.gguf" },
  {
    temperature: 0.7,
    max_tokens: 512,
  },
);
```

#### Streaming

ChatWllama supports real-time streaming of response. When streaming is enabled:

- Response tokens are yielded one at a time as they are generated
- Use `stream: true` in the model configuration or set `model.streaming = true`

#### Streaming Example

```typescript
import { HumanMessage } from "@langchain/core/messages";

const result = await model._generate([new HumanMessage({ content: "Hello!" })], {
  streaming: true,
});

// result.generations will contain the concatenated response
```

#### Streaming Configuration

- Set `stream: true` in model initialization or via `model.streaming`
- Abort control via `options.signal` in both streaming and non-streaming modes

## Tool Calling

Tool calling allows the model to invoke external functions during response generation.

#### Tool Calling Example

```typescript
import { ChatWllama } from "chatwllama";
import { HumanMessage } from "@langchain/core/messages";

// Define a function the model can call
const getWeather = async (city: string) => {
  // In a real app, this would call a weather API
  return `The weather in ${city} is 25°C and sunny.`;
};

const model = new ChatWllama(
  { repo: "some/model", file: "model.Q4_0.gguf" },
  {
    temperature: 0.7,
    max_tokens: 512,
  },
);

// Bind tools to the model
const runnable = model.bindTools([
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "Get the weather for a city",
      parameters: {
        type: "object",
        properties: { city: { type: "string" } },
      },
    },
  },
]);

// Invoke with a user message asking about weather
const result = await runnable.invoke({
  messages: [new HumanMessage({ content: "What's the weather in San Francisco?" })],
});

console.log(result.message.content);
// Output: "The weather in San Francisco is 25°C and sunny."
```

## Development

### Running Tests

Execute the test suite with:

```bash
vp test
```

Tests are located in the `tests/` directory and use Vitest.

### Lint & Format

Run the linter and formatter:

```bash
vp check
```

This runs both type checking and formatting checks. The project uses:

- **oxlint** for linting with type-aware checking
- **oxfmt** for formatting

The vite.config.ts is configured with lint and fmt options enabled.
