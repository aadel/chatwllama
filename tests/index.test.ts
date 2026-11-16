import type { BaseMessage } from "@langchain/core/messages";
import type { ChatCompletionResponse } from "@wllama/wllama";

import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";
import { describe, expect, test, vi } from "vite-plus/test";

import { ChatWllama, type WllamaInput } from "../src/index.ts";

vi.mock("@wllama/wllama", () => ({
  Wllama: class {
    createChatCompletion = vi.fn();
    loadModelFromHF = vi.fn();
  },
}));

const HF_MODEL = { repo: "test/repo", file: "model.Q4_0.gguf" };

const createChatWllama = (fields?: Partial<WllamaInput>) =>
  new ChatWllama(HF_MODEL, fields as WllamaInput);

const createMockResponse = (content: string): ChatCompletionResponse =>
  ({
    id: "chatcmpl-test",
    object: "chat.completion",
    created: 1700000000,
    model: "test-model",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop",
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: 4, completion_tokens: 3, total_tokens: 7 },
  }) as ChatCompletionResponse;

const createMockStream = (chunks: Array<Record<string, unknown>>) =>
  vi.fn().mockImplementation(async function* () {
    for (const chunk of chunks) {
      yield chunk;
    }
  });

describe("ChatWllama", () => {
  describe("metadata", () => {
    test("exposes the expected class name", () => {
      expect(ChatWllama.lc_name()).toBe("ChatWllama");
    });

    test("exposes the expected llm type", () => {
      expect(createChatWllama()._llmType()).toBe("wllama");
    });
  });

  describe("constructor", () => {
    test("uses the default wllama config path", () => {
      expect(createChatWllama().wllamaConfigPaths).toEqual({
        default: "../esm/wasm/wllama.wasm",
      });
    });

    test("uses custom wllama config paths when provided", () => {
      expect(
        createChatWllama({ wllamaConfigPaths: { default: "/custom/wllama.wasm" } })
          .wllamaConfigPaths,
      ).toEqual({ default: "/custom/wllama.wasm" });
    });

    test("stores the hugging face model", () => {
      expect(createChatWllama().HFModel).toEqual(HF_MODEL);
    });

    test("applies default sampling parameters", () => {
      const chatwllama = createChatWllama();

      expect(chatwllama.streaming).toBe(false);
      expect(chatwllama.temperature).toBe(1);
      expect(chatwllama.max_tokens).toBeUndefined();
      expect(chatwllama.top_p).toBeUndefined();
      expect(chatwllama.top_k).toBeUndefined();
      expect(chatwllama.seed).toBeUndefined();
      expect(chatwllama.min_p).toBeUndefined();
      expect(chatwllama.presence_penalty).toBeUndefined();
      expect(chatwllama.repetition_penalty).toBeUndefined();
    });

    test("applies sampling parameters from the fields", () => {
      const chatwllama = createChatWllama({
        stream: true,
        temperature: 0.25,
        max_tokens: 128,
        top_p: 0.8,
        top_k: 20,
        seed: 7,
        min_p: 0.8,
        presence_penalty: 0.1,
        repetition_penalty: 1.2,
      });

      expect(chatwllama.streaming).toBe(true);
      expect(chatwllama.temperature).toBe(0.25);
      expect(chatwllama.max_tokens).toBe(128);
      expect(chatwllama.top_p).toBe(0.8);
      expect(chatwllama.top_k).toBe(20);
      expect(chatwllama.seed).toBe(7);
      expect(chatwllama.min_p).toBe(0.8);
      expect(chatwllama.presence_penalty).toBe(0.1);
      expect(chatwllama.repetition_penalty).toBe(1.2);
    });
  });

  describe("invocationParams", () => {
    test("returns the sampling parameters of the instance", () => {
      const chatwllama = createChatWllama({
        stream: true,
        temperature: 0.5,
        max_tokens: 25,
      });

      const params = chatwllama.invocationParams();

      expect(params.stream).toBe(true);
      expect(params.temperature).toBe(0.5);
      expect(params.max_tokens).toBe(25);
    });

    test("returns default invocation parameters", () => {
      const params = createChatWllama().invocationParams();

      expect(params.stream).toBe(false);
      expect(params.temperature).toBe(1);
      expect(params.max_tokens).toBeUndefined();
    });

    test("returns tools and tool choice when provided", () => {
      const chatwllama = createChatWllama();

      const params = chatwllama.invocationParams({
        tools: [
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
        ],
        tool_choice: "auto",
      });

      expect(params.tools).toHaveLength(1);
      expect(params.tool_choice).toBe("auto");
    });

    test("does not return tools when none are provided", () => {
      expect(createChatWllama().invocationParams().tools).toBeUndefined();
    });
  });

  describe("initialize", () => {
    test("loads the model from hugging face", async () => {
      const chatwllama = createChatWllama();
      const loadModelFromHF = vi.fn().mockResolvedValue(undefined);
      (chatwllama as unknown as { _wllama: unknown })._wllama = {
        loadModelFromHF,
      };

      await chatwllama.initialize({ n_ctx: 1024 });

      expect(loadModelFromHF).toHaveBeenCalledWith(HF_MODEL, { n_ctx: 1024 });
    });
  });

  describe("_generate", () => {
    test("returns the generated text for a non streaming completion", async () => {
      const chatwllama = createChatWllama();
      const createChatCompletion = vi
        .fn()
        .mockResolvedValue(createMockResponse("Hello from wllama!"));

      (chatwllama as unknown as { _wllama: unknown })._wllama = {
        createChatCompletion,
      };

      const result = await (
        chatwllama as unknown as {
          _generate: (
            messages: BaseMessage[],
            options: unknown,
          ) => Promise<{ generations: Array<{ text: string }> }>;
        }
      )._generate([new HumanMessage({ content: "Hi" })], {});

      expect(createChatCompletion).toHaveBeenCalledTimes(1);
      expect(createChatCompletion.mock.calls[0][0]).toEqual(
        expect.objectContaining({
          stream: false,
          messages: [{ role: "user", content: "Hi" }],
        }),
      );
      expect(result.generations[0].text).toBe("Hello from wllama!");
    });

    test("streams chunks and concatenates the generated text", async () => {
      const chatwllama = createChatWllama({ stream: true });
      const createChatCompletion = createMockStream([
        {
          id: "chatcmpl-test",
          object: "chat.completion.chunk",
          created: 1700000000,
          model: "test-model",
          choices: [
            {
              index: 0,
              delta: { role: "assistant", content: "Hello" },
              finish_reason: null,
              logprobs: null,
            },
          ],
        },
        {
          id: "chatcmpl-test",
          object: "chat.completion.chunk",
          created: 1700000000,
          model: "test-model",
          choices: [
            {
              index: 0,
              delta: { role: "assistant", content: " from wllama!" },
              finish_reason: "stop",
              logprobs: null,
            },
          ],
        },
      ]);

      (chatwllama as unknown as { _wllama: unknown })._wllama = {
        createChatCompletion,
      };

      const result = await (
        chatwllama as unknown as {
          _generate: (
            messages: BaseMessage[],
            options: unknown,
          ) => Promise<{ generations: Array<{ text: string }> }>;
        }
      )._generate([new HumanMessage({ content: "Hi" })], {});

      expect(createChatCompletion).toHaveBeenCalledWith(expect.objectContaining({ stream: true }));
      expect(result.generations[0].text).toBe("Hello from wllama!");
    });

    test("returns parsed tool calls for a tool response", async () => {
      const chatwllama = createChatWllama();
      const createChatCompletion = vi.fn().mockResolvedValue({
        id: "chatcmpl-test",
        object: "chat.completion",
        created: 1700000000,
        model: "test-model",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: "",
              tool_calls: [
                {
                  id: "call_1",
                  type: "function",
                  function: { name: "get_weather", arguments: '{"city":"Paris"}' },
                },
              ],
            },
            finish_reason: "tool_calls",
            logprobs: null,
          },
        ],
        usage: { prompt_tokens: 4, completion_tokens: 3, total_tokens: 7 },
      } as ChatCompletionResponse);

      (chatwllama as unknown as { _wllama: unknown })._wllama = {
        createChatCompletion,
      };

      const result = await (
        chatwllama as unknown as {
          _generate: (
            messages: BaseMessage[],
            options: unknown,
          ) => Promise<{ generations: Array<{ text: string; message: AIMessage }> }>;
        }
      )._generate([new HumanMessage({ content: "Hi" })], {});

      expect(result.generations[0].message.tool_calls).toEqual([
        { id: "call_1", name: "get_weather", args: { city: "Paris" } },
      ]);
    });

    test("returns an empty generation when the stream yields no chunks", async () => {
      const chatwllama = createChatWllama({ stream: true });
      const createChatCompletion = createMockStream([]);

      (chatwllama as unknown as { _wllama: unknown })._wllama = {
        createChatCompletion,
      };

      const result = await (
        chatwllama as unknown as {
          _generate: (
            messages: BaseMessage[],
            options: unknown,
          ) => Promise<{ generations: Array<{ text: string }> }>;
        }
      )._generate([new HumanMessage({ content: "Hi" })], {});

      expect(result.generations[0].text).toBe("");
    });

    test("handles tool calls with empty arguments", async () => {
      const chatwllama = createChatWllama();
      const createChatCompletion = vi.fn().mockResolvedValue({
        id: "chatcmpl-test",
        object: "chat.completion",
        created: 1700000000,
        model: "test-model",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: "",
              tool_calls: [
                {
                  id: "call_1",
                  type: "function",
                  function: { name: "get_weather", arguments: "" },
                },
              ],
            },
            finish_reason: "tool_calls",
            logprobs: null,
          },
        ],
        usage: { prompt_tokens: 4, completion_tokens: 3, total_tokens: 7 },
      } as ChatCompletionResponse);

      (chatwllama as unknown as { _wllama: unknown })._wllama = {
        createChatCompletion,
      };

      const result = await (
        chatwllama as unknown as {
          _generate: (
            messages: BaseMessage[],
            options: unknown,
          ) => Promise<{ generations: Array<{ text: string; message: AIMessage }> }>;
        }
      )._generate([new HumanMessage({ content: "Hi" })], {});

      expect(result.generations[0].message.tool_calls).toEqual([
        { id: "call_1", name: "get_weather", args: {} },
      ]);
    });
  });

  test("does not crash when usage is missing", async () => {
    const chatwllama = createChatWllama();
    const createChatCompletion = vi.fn().mockResolvedValue({
      id: "chatcmpl-test",
      object: "chat.completion",
      created: 1700000000,
      model: "test-model",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "Hello!" },
          finish_reason: "stop",
          logprobs: null,
        },
      ],
    } as ChatCompletionResponse);

    (chatwllama as unknown as { _wllama: unknown })._wllama = {
      createChatCompletion,
    };

    const result = await (
      chatwllama as unknown as {
        _generate: (
          messages: BaseMessage[],
          options: unknown,
        ) => Promise<{ generations: Array<{ text: string }> }>;
      }
    )._generate([new HumanMessage({ content: "Hi" })], {});

    expect(result.generations[0].text).toBe("Hello!");
  });

  test("streams tool calls and returns parsed tool calls", async () => {
    const chatwllama = createChatWllama({ stream: true });
    const createChatCompletion = createMockStream([
      {
        id: "chatcmpl-test",
        object: "chat.completion.chunk",
        created: 1700000000,
        model: "test-model",
        choices: [
          {
            index: 0,
            delta: {
              role: "assistant",
              content: "",
              tool_calls: [
                {
                  index: 0,
                  id: "call_1",
                  type: "function",
                  function: { name: "get_weather", arguments: '{"city":' },
                },
              ],
            },
            finish_reason: null,
            logprobs: null,
          },
        ],
      },
      {
        id: "chatcmpl-test",
        object: "chat.completion.chunk",
        created: 1700000000,
        model: "test-model",
        choices: [
          {
            index: 0,
            delta: {
              role: "assistant",
              content: "",
              tool_calls: [{ index: 0, type: "function", function: { arguments: '"Paris"}' } }],
            },
            finish_reason: "tool_calls",
            logprobs: null,
          },
        ],
      },
    ]);

    (chatwllama as unknown as { _wllama: unknown })._wllama = {
      createChatCompletion,
    };

    const result = await (
      chatwllama as unknown as {
        _generate: (
          messages: BaseMessage[],
          options: unknown,
        ) => Promise<{ generations: Array<{ text: string; message: AIMessage }> }>;
      }
    )._generate([new HumanMessage({ content: "Hi" })], {});

    expect(result.generations[0].message.tool_calls).toEqual([
      { name: "get_weather", args: { city: "Paris" }, type: "tool_call", id: "call_1" },
    ]);
  });

  test("returns usage metadata from the stream", async () => {
    const chatwllama = createChatWllama({ stream: true });
    const createChatCompletion = createMockStream([
      {
        id: "chatcmpl-test",
        object: "chat.completion.chunk",
        created: 1700000000,
        model: "test-model",
        choices: [
          {
            index: 0,
            delta: { role: "assistant", content: "Hello" },
            finish_reason: null,
            logprobs: null,
          },
        ],
        usage: { prompt_tokens: 2, completion_tokens: 3, total_tokens: 5 },
      },
    ]);

    (chatwllama as unknown as { _wllama: unknown })._wllama = {
      createChatCompletion,
    };

    const result = await (
      chatwllama as unknown as {
        _generate: (
          messages: BaseMessage[],
          options: unknown,
        ) => Promise<{ generations: Array<{ text: string; message: AIMessage }> }>;
      }
    )._generate([new HumanMessage({ content: "Hi" })], {});

    expect(result.generations[0].message.usage_metadata).toEqual({
      input_tokens: 2,
      output_tokens: 3,
      total_tokens: 5,
      input_token_details: {},
      output_token_details: {}
    });
  });

  test("propagates abort signals", async () => {
    const controller = new AbortController();
    controller.abort();
    const chatwllama = createChatWllama({ stream: true });
    const createChatCompletion = createMockStream([
      {
        id: "chatcmpl-test",
        object: "chat.completion.chunk",
        created: 1700000000,
        model: "test-model",
        choices: [
          {
            index: 0,
            delta: { role: "assistant", content: "Hello" },
            finish_reason: null,
            logprobs: null,
          },
        ],
      },
    ]);

    (chatwllama as unknown as { _wllama: unknown })._wllama = {
      createChatCompletion,
    };

    await expect(
      (
        chatwllama as unknown as {
          _generate: (
            messages: BaseMessage[],
            options: unknown,
          ) => Promise<{ generations: Array<{ text: string }> }>;
        }
      )._generate([new HumanMessage({ content: "Hi" })], { signal: controller.signal }),
    ).rejects.toThrow();
  });

  test("propagates errors from the model client", async () => {
    const chatwllama = createChatWllama();
    const createChatCompletion = vi.fn().mockRejectedValue(new Error("model load failed"));
    (chatwllama as unknown as { _wllama: unknown })._wllama = {
      createChatCompletion,
    };

    await expect(
      (
        chatwllama as unknown as {
          _generate: (
            messages: BaseMessage[],
            options: unknown,
          ) => Promise<{ generations: Array<{ text: string }> }>;
        }
      )._generate([new HumanMessage({ content: "Hi" })], {}),
    ).rejects.toThrow("model load failed");
  });
});

describe("_call", () => {
  test("joins the streamed chunks into a response", async () => {
    const chatwllama = createChatWllama({ stream: true });
    const createChatCompletion = createMockStream([
      {
        id: "chatcmpl-test",
        object: "chat.completion.chunk",
        created: 1700000000,
        model: "test-model",
        choices: [
          {
            index: 0,
            delta: { role: "assistant", content: "Hello" },
            finish_reason: null,
            logprobs: null,
          },
        ],
      },
      {
        id: "chatcmpl-test",
        object: "chat.completion.chunk",
        created: 1700000000,
        model: "test-model",
        choices: [
          {
            index: 0,
            delta: { role: "assistant", content: " from wllama!" },
            finish_reason: "stop",
            logprobs: null,
          },
        ],
      },
    ]);

    (chatwllama as unknown as { _wllama: unknown })._wllama = {
      createChatCompletion,
    };

    const response = await chatwllama._call([new HumanMessage({ content: "Hi" })], {});

    expect(response).toBe("Hello from wllama!");
  });
});

describe("_convertToWllama", () => {
  test("converts langchain messages to wllama chat completion messages", () => {
    const chatwllama = createChatWllama();
    const convertToWllama = (
      chatwllama as unknown as {
        _convertToWllama: (messages: BaseMessage[]) => unknown[];
      }
    )._convertToWllama.bind(chatwllama);

    expect(
      convertToWllama([
        new SystemMessage({ content: "You are helpful." }),
        new HumanMessage({ content: "Hi!" }),
        new AIMessage({ content: "Hello!" }),
        new ToolMessage({ content: "25C", tool_call_id: "call_1" }),
      ]),
    ).toEqual([
      { role: "system", content: "You are helpful." },
      { role: "user", content: "Hi!" },
      { role: "assistant", content: "Hello!" },
      { role: "tool", content: "25C", tool_call_id: "call_1" },
    ]);
  });

  test("includes assistant tool calls when present", () => {
    const chatwllama = createChatWllama();
    const convertToWllama = (
      chatwllama as unknown as {
        _convertToWllama: (messages: BaseMessage[]) => unknown[];
      }
    )._convertToWllama.bind(chatwllama);

    expect(
      convertToWllama([
        new HumanMessage({ content: "What's the weather in Paris?" }),
        new AIMessage({
          content: "",
          tool_calls: [{ id: "call_1", name: "get_weather", args: { city: "Paris" } }],
        }),
      ]),
    ).toEqual([
      { role: "user", content: "What's the weather in Paris?" },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "get_weather", arguments: '{"city":"Paris"}' },
          },
        ],
      },
    ]);
  });

  test("throws when the message content is not a string", () => {
    const chatwllama = createChatWllama();
    const convertToWllama = (
      chatwllama as unknown as {
        _convertToWllama: (messages: BaseMessage[]) => unknown[];
      }
    )._convertToWllama.bind(chatwllama);

    const nonStringContentMessage = {
      type: "human",
      content: [{ type: "text", text: "Hi!" }],
    } as unknown as BaseMessage;

    expect(() =>
      convertToWllama([new HumanMessage({ content: "Hi" }), nonStringContentMessage]),
    ).toThrow("ChatWllama does not support non-string message content in sessions.");
  });

  test("throws for generic messages", () => {
    const chatwllama = createChatWllama();
    const convertToWllama = (
      chatwllama as unknown as {
        _convertToWllama: (messages: BaseMessage[]) => unknown[];
      }
    )._convertToWllama.bind(chatwllama);

    const genericMessage = { type: "generic", content: "Hi!" } as unknown as BaseMessage;

    expect(() => convertToWllama([genericMessage])).toThrow("Generic messages are not supported.");
  });
});

describe("bindTools", () => {
  test("returns a runnable bound to the tools", () => {
    const chatwllama = createChatWllama();

    const runnable = chatwllama.bindTools([
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

    expect(runnable).toBeDefined();
  });
});
