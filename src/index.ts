import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import type { BaseLanguageModelCallOptions } from "@langchain/core/language_models/base";
import type { ChatGeneration } from "@langchain/core/outputs";
import type {
  ChatCompletionAssistantMessage,
  ChatCompletionSystemMessage,
  ChatCompletionUserMessage,
  ChatCompletionToolMessage,
  ChatCompletionTool,
  ChatCompletionToolCall,
  ChatCompletionChoice,
  ChatCompletionChunkDelta,
  HuggingFaceParams,
  ChatCompletionUsage,
} from "@wllama/wllama";

import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import {
  BaseChatModel,
  type BindToolsInput,
  type BaseChatModelParams,
} from "@langchain/core/language_models/chat_models";
import {
  BaseMessage,
  HumanMessageChunk,
  ToolMessageChunk,
  FunctionMessageChunk,
  ChatMessageChunk,
  AIMessage,
  ToolMessage,
  AIMessageChunk,
  type ToolCallChunk,
  type MessageStructure,
  type MessageToolSet,
  type MessageType,
  type ToolCall,
} from "@langchain/core/messages";
import { ChatGenerationChunk, type ChatResult } from "@langchain/core/outputs";
import { Runnable } from "@langchain/core/runnables";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import {
  Wllama,
  type ChatCompletionMessage,
  type ChatCompletionToolChoice,
  type DownloadOptions,
  type LoadModelParams,
} from "@wllama/wllama";
import { Logger, LogLevel } from "tslog";

const DEFAULT_TEMPERATURE = 1.0;

const CONFIG_PATHS = {
  default: "../esm/wasm/wllama.wasm",
};

export interface WllamaConfigPaths {
  default: string;
}

type ChatCompletionOptionsParameters = Parameters<Wllama["createChatCompletion"]>;
type ChatCompletionOptions = ChatCompletionOptionsParameters[0] & {
  presence_penalty?: number;
  repetition_penalty?: number;
};

export interface WllamaInput extends BaseChatModelParams, Omit<ChatCompletionOptions, "messages"> {
  /**
   * Wllama parameters
   */
  HFModel: HuggingFaceParams;
  /**
   * Whether or not to stream responses.
   */
  wllamaConfigPaths: WllamaConfigPaths;
}

export interface WllamaCallOptions extends BaseLanguageModelCallOptions {
  tools?: BindToolsInput[];
  /**
   * Whether or not to specify what tool the model should use
   * @default "auto"
   */
  tool_choice?: ChatCompletionToolChoice;
  response_format?: {
    type: "json_object";
    schema: Record<string, unknown>;
  };
}

/**
 * To use this module you need to have the `@wllama/wllama` module installed.
 */
export class ChatWllama
  extends BaseChatModel<WllamaCallOptions>
  implements Omit<WllamaInput, "stream">
{
  static logger: Logger<object>;

  private _wllama: Wllama;

  protected inputs?: WllamaInput;

  get wllama(): Wllama {
    return this._wllama;
  }
  protected set wllama(value: Wllama) {
    this._wllama = value;
  }

  wllamaConfigPaths: WllamaConfigPaths;

  streaming?: boolean = false;

  temperature: number;

  top_p?: number;

  top_k?: number;

  seed?: number;

  max_tokens?: number;

  min_p?: number;

  presence_penalty?: number;

  repetition_penalty?: number;

  HFModel: HuggingFaceParams;

  static lc_name() {
    return "ChatWllama";
  }

  _llmType() {
    return "wllama";
  }

  static {
    // Initialize the logger configuration
    ChatWllama.logger = new Logger({
      minLevel: LogLevel.DEBUG, // Set minimum severity threshold
      name: "ChatWllama",
    });
  }

  constructor(model: HuggingFaceParams, fields?: Omit<WllamaInput, "HFModelParams">) {
    super(fields ?? {});
    this.wllamaConfigPaths = fields?.wllamaConfigPaths ?? CONFIG_PATHS;
    this.inputs = fields;
    this.HFModel = model;
    this.streaming = fields?.stream ?? this.streaming;
    this.temperature = fields?.temperature ?? DEFAULT_TEMPERATURE;
    this.max_tokens = fields?.max_tokens;
    this.top_k = fields?.top_k;
    this.top_p = fields?.top_p;
    this.seed = fields?.seed;
    this.min_p = fields?.min_p ?? undefined;
    this.presence_penalty = fields?.presence_penalty ?? undefined;
    this.repetition_penalty = fields?.repetition_penalty ?? undefined;

    this._wllama = new Wllama(this.wllamaConfigPaths);
  }

  async initialize(params?: LoadModelParams & DownloadOptions & { useCache?: boolean }) {
    await this._wllama.loadModelFromHF(this.HFModel, { ...params });
  }

  /**
   * Get the parameters used to invoke the model
   */
  override invocationParams(
    options?: this["ParsedCallOptions"],
  ): Omit<ChatCompletionOptions, "messages" | "stream"> & { stream?: boolean } {
    const tools =
      (options?.tools?.map((tool) => convertToOpenAITool(tool)) as ChatCompletionTool[]) ?? [];
    const hasTools = tools.length > 0;
    const parameters: Omit<ChatCompletionOptions, "messages" | "stream"> & { stream?: boolean } = {
      stream: this.streaming,
      temperature: this.temperature,
      top_p: this.top_p,
      top_k: this.top_k,
      seed: this.seed,
      max_tokens: this.max_tokens,
      min_p: this.min_p,
      presence_penalty: this.presence_penalty,
      repetition_penalty: this.repetition_penalty,
    };
    if (hasTools) {
      parameters.tools = tools;
    }
    const toolChoice = options?.tool_choice;
    if (toolChoice !== undefined) {
      parameters.tool_choice = toolChoice;
    }

    return parameters;
  }

  private _convertDeltaToMessageChunk(
    delta: ChatCompletionChunkDelta,
    usage?: ChatCompletionUsage | null,
    defaultRole?: "assistant",
  ) {
    const rawToolCalls = delta.tool_calls?.length
      ? (delta.tool_calls?.map((toolCall) => {
          if (toolCall.index === undefined || toolCall.index === null)
            throw Error(`Invalid tool call ${JSON.stringify(toolCall)}`);

          const validId = toolCall.id && toolCall.id !== "" ? toolCall.id : undefined;
          const validToolCall = toolCall as ChatCompletionToolCall;
          return {
            ...validToolCall,
            ...(validId !== null && { id: validId }),
            type: "function",
          };
        }) as (ChatCompletionToolCall & { index: number; type: "function" })[])
      : undefined;

    const role = delta.role || defaultRole || "assistant";
    const content = delta.content ?? "";
    const additional_kwargs = {
      ...(rawToolCalls ? { tool_calls: rawToolCalls } : {}),
      ...("reasoning_content" in delta ? { reasoning: delta?.reasoning_content } : {}),
    };

    const usageMetadata = {
      input_tokens: usage?.prompt_tokens ?? 0,
      output_tokens: usage?.completion_tokens ?? 0,
      total_tokens: usage?.total_tokens ?? 0,
    };

    switch (role) {
      case "user" as string:
        return new HumanMessageChunk({ content });

      case "assistant" as string: {
        // Extract tool call chunks creation
        const toolCallChunks: ToolCallChunk[] = [];
        if (rawToolCalls && rawToolCalls?.length > 0) {
          for (const rawToolCallChunk of rawToolCalls) {
            let collectedToolCall: ToolCallChunk = {};
            collectedToolCall.index = rawToolCallChunk.index;
            if (rawToolCallChunk.id) collectedToolCall.id = rawToolCallChunk.id;
            if (rawToolCallChunk.function?.name)
              collectedToolCall.name = rawToolCallChunk.function.name;
            if (rawToolCallChunk.function?.arguments)
              collectedToolCall.args = rawToolCallChunk.function.arguments;

            toolCallChunks.push(collectedToolCall);
          }
        }

        return new AIMessageChunk({
          content,
          tool_call_chunks: toolCallChunks,
          additional_kwargs,
          usage_metadata: usageMetadata,
        });
      }

      case "tool" as string:
        if (rawToolCalls) {
          return new ToolMessageChunk({
            content,
            additional_kwargs,
            tool_call_id: rawToolCalls[0].id,
          });
        }
        return null;

      case "function" as string:
        return new FunctionMessageChunk({
          content,
          additional_kwargs,
        });

      default:
        return new ChatMessageChunk({ content, role });
    }
  }

  async *_streamResponseChunks(
    baseMessages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    let defaultRole;
    const messages: ChatCompletionMessage[] = this._convertToWllama(baseMessages);

    ChatWllama.logger.debug("Options", options);

    let fullContent: string = "";

    const stream = await this.wllama.createChatCompletion({
      messages,
      ...this.invocationParams(options),
      stream: true,
    });

    let finishReason = null;
    let messageChunks: AIMessageChunk[] = [];

    for await (const chunk of stream) {
      const choice = chunk.choices[0] as ChatCompletionChoice &
        Record<"delta", ChatCompletionMessage>;

      ChatWllama.logger.debug("Choice", choice);

      if (choice && !("delta" in choice)) {
        continue;
      }

      const delta = choice?.delta as ChatCompletionChunkDelta & { reasoning_content: string };

      if (!delta) {
        continue;
      }

      finishReason = choice.finish_reason ?? finishReason;

      if (delta.content) fullContent += delta.content;

      const usageMetadata = chunk.usage;

      const messageChunk = this._convertDeltaToMessageChunk(
        delta,
        usageMetadata,
        defaultRole,
      ) as AIMessageChunk;

      messageChunks.push(messageChunk);
      defaultRole = delta.role || defaultRole;

      if (delta.reasoning_content) {
        ChatWllama.logger.info("assistant", "(reasoning)", delta.reasoning_content);
      }

      if (finishReason === "stop") {
        if (fullContent) ChatWllama.logger.info("assistant", fullContent);
        else ChatWllama.logger.info("assistant", "(no content)");
      }

      // Last chunk has undefined content
      if (
        messageChunk === null ||
        (!delta.content && !delta.tool_calls && delta.role === "assistant")
      ) {
        continue;
      }

      const text = delta.content ?? "";
      const generationChunk = new ChatGenerationChunk({
        text: text,
        message: messageChunk,
      });

      yield generationChunk;

      ChatWllama.logger.debug("Delta", delta);

      await runManager?.handleLLMNewToken(text);
    } // for await (const chunk of stream)
  }

  private _convertToWllama(
    baseMessages: BaseMessage<MessageStructure<MessageToolSet>, MessageType>[],
  ): ChatCompletionMessage[] {
    return baseMessages.map((message) => {
      if (typeof message.content !== "string") {
        throw new Error("ChatWllama does not support non-string message content in sessions.");
      }
      const langChainType = message.type;
      let wllamaMessage: ChatCompletionMessage;
      if (langChainType === "ai") {
        const assistantMessage: ChatCompletionAssistantMessage = {
          role: "assistant",
          content: message.content,
        };
        const toolCalls = (message as AIMessage).tool_calls;
        if (toolCalls && toolCalls.length > 0) {
          assistantMessage.tool_calls = toolCalls.map((toolCall) => ({
            id: toolCall.id ?? "",
            type: "function" as const,
            function: {
              name: toolCall.name,
              arguments: JSON.stringify(toolCall.args ?? {}),
            },
          }));
        }
        wllamaMessage = assistantMessage;
      } else if (langChainType === "human") {
        wllamaMessage = {
          role: "user",
          content: message.content,
        } as ChatCompletionUserMessage;
      } else if (langChainType === "system") {
        wllamaMessage = {
          role: "system",
          content: message.content,
        } as ChatCompletionSystemMessage;
      } else if (langChainType === "tool") {
        wllamaMessage = {
          role: "tool",
          content: message.content,
          tool_call_id: (message as ToolMessage).tool_call_id,
        } as ChatCompletionToolMessage;
      } else {
        throw new Error("Generic messages are not supported.");
      }
      return wllamaMessage;
    });
  }

  async _generate(
    baseMessages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    options.signal?.throwIfAborted();

    if (this.streaming) {
      const messageChunks: AIMessageChunk[] = [];
      for await (const chunk of this._streamResponseChunks(baseMessages, options, runManager)) {
        options.signal?.throwIfAborted();
        messageChunks.push(chunk.message as AIMessageChunk);
      }
      const finalChunk = messageChunks.reduce<AIMessageChunk>(
        (acc, messageChunk) => acc.concat(messageChunk),
        new AIMessageChunk({ content: "" }),
      );

      const nonChunkMessage = new AIMessage({
        id: finalChunk.id,
        content: finalChunk.content ?? "",
        additional_kwargs: finalChunk.additional_kwargs,
        tool_calls: finalChunk.tool_calls,
        response_metadata: finalChunk.response_metadata,
        usage_metadata: finalChunk.usage_metadata,
      });

      return {
        generations: [
          {
            text: typeof nonChunkMessage.content === "string" ? nonChunkMessage.content : "",
            message: nonChunkMessage,
          },
        ],
      };
    } else {
      return { generations: await this._generateNonStreaming(baseMessages, options) };
    }
  }

  async _generateNonStreaming(
    baseMessages: BaseMessage[],
    options: this["ParsedCallOptions"],
  ): Promise<ChatGeneration[]> {
    options.signal?.throwIfAborted();

    const messages: ChatCompletionMessage[] = this._convertToWllama(baseMessages);

    const response = await this.wllama.createChatCompletion({
      messages,
      ...this.invocationParams(options),
      stream: false,
    });

    const { message } = response.choices[0];
    const { usage } = response;

    if (message.tool_calls) {
      const tool_calls: ToolCall[] = [];
      for (let tc of message.tool_calls) {
        let args: Record<string, unknown> = {};
        if (tc.function.arguments) {
          try {
            args = JSON.parse(tc.function.arguments);
          } catch {
            args = {};
          }
        }
        tool_calls.push({
          id: tc.id,
          name: tc.function.name,
          args,
        });
      }

      const nonChunkMessage = {
        text: "",
        message: new AIMessage({
          id: response.id,
          content: message.content ?? "",
          tool_calls: tool_calls,
          usage_metadata: {
            input_tokens: usage?.prompt_tokens ?? 0,
            output_tokens: usage?.completion_tokens ?? 0,
            total_tokens: usage?.total_tokens ?? 0,
          },
        }),
      };

      const generations: ChatGeneration[] = [nonChunkMessage];

      return generations;
    } else {
      return [
        {
          text: message.content ?? "",
          message: new AIMessage({
            id: response.id,
            content: message.content ?? "",
            usage_metadata: {
              input_tokens: usage?.prompt_tokens ?? 0,
              output_tokens: usage?.completion_tokens ?? 0,
              total_tokens: usage?.total_tokens ?? 0,
            },
          }),
        },
      ];
    }
  }

  async _call(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): Promise<string> {
    const chunks = [];
    for await (const chunk of this._streamResponseChunks(messages, options, runManager)) {
      chunks.push(chunk.text);
    }
    return chunks.join("");
  }

  override bindTools(
    tools: BindToolsInput[],
    kwargs?: Partial<this["ParsedCallOptions"]>,
  ): Runnable<BaseLanguageModelInput, AIMessageChunk, WllamaCallOptions> {
    return this.withConfig({
      tools: tools.map((tool) => convertToOpenAITool(tool)),
      ...kwargs,
    });
  }
}
