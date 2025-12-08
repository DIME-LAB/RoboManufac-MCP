import OpenAI from 'openai';
import { encoding_for_model, get_encoding } from 'tiktoken';
import type {
  ModelProvider,
  TokenCounter,
  Tool,
  Message,
  TokenUsage,
  SummarizationConfig,
  MessageStreamEvent,
} from '../model-provider.js';

// OpenAI model context window limits (in tokens)
const OPENAI_MODEL_CONTEXT_WINDOWS: Record<string, number> = {
  'gpt-5': 200000,
  'gpt-4o': 128000,
  'gpt-4o-mini': 128000,
  'gpt-4-turbo': 128000,
  'gpt-4': 8192,
  'gpt-4-32k': 32768,
  'gpt-3.5-turbo': 16385,
  'gpt-3.5-turbo-16k': 16385,
  'o1-preview': 200000,
  'o1-mini': 128000,
  // Add other OpenAI model variants as needed
};

// OpenAI Token Counter Implementation
export class OpenAITokenCounter implements TokenCounter {
  private encoder: any;
  private maxTokens: number;
  private modelName: string;
  private config: SummarizationConfig;

  constructor(
    modelName: string = 'gpt-5',
    config: Partial<SummarizationConfig> = {},
  ) {
    this.modelName = modelName;
    this.maxTokens =
      OPENAI_MODEL_CONTEXT_WINDOWS[modelName] ||
      OPENAI_MODEL_CONTEXT_WINDOWS['gpt-5'] ||
      200000;

    // OpenAI models use cl100k_base encoding
    try {
      // Map OpenAI model names to tiktoken model names
      const tiktokenModel = modelName.startsWith('gpt-5') ? 'gpt-4' : // gpt-5 uses gpt-4 encoding
                           modelName.startsWith('gpt-4') ? 'gpt-4' :
                           modelName.startsWith('gpt-3.5') ? 'gpt-3.5-turbo' :
                           modelName.startsWith('o1') ? 'gpt-4' : // o1 models use gpt-4 encoding
                           'gpt-4'; // Default to gpt-4 encoding
      this.encoder = encoding_for_model(tiktokenModel);
    } catch (error) {
      // Fallback: try to get cl100k_base directly
      try {
        this.encoder = get_encoding('cl100k_base');
      } catch (e) {
        // Last resort fallback - will use character estimation
        this.encoder = null;
      }
    }

    this.config = {
      threshold: 80, // Default: summarize at 80% of context window
      recentMessagesToKeep: 10, // Default: keep last 10 messages
      enabled: true,
      ...config,
    };
  }

  countTokens(text: string): number {
    if (!this.encoder) {
      // Fallback estimation: ~4 characters per token (rough approximation)
      return Math.ceil(text.length / 4);
    }

    try {
      const tokens = this.encoder.encode(text);
      return tokens.length;
    } catch (error) {
      // Fallback to rough estimation if encoding fails
      return Math.ceil(text.length / 4);
    }
  }

  countMessageTokens(message: { role: string; content: string }): number {
    // Count tokens for the message structure
    // Format: role + content + overhead
    const roleTokens = this.countTokens(message.role);
    const contentTokens = this.countTokens(message.content);
    // Add overhead for message structure (approximately 4 tokens)
    return roleTokens + contentTokens + 4;
  }

  getUsage(currentTokens: number): TokenUsage {
    const percentage = (currentTokens / this.maxTokens) * 100;

    let suggestion: 'continue' | 'warn' | 'break';
    if (percentage < 60) {
      suggestion = 'continue';
    } else if (percentage < 80) {
      suggestion = 'warn';
    } else {
      suggestion = 'break';
    }

    return {
      current: currentTokens,
      limit: this.maxTokens,
      percentage: Math.round(percentage * 100) / 100, // Round to 2 decimal places
      suggestion,
    };
  }

  shouldSummarize(currentTokens: number): boolean {
    if (!this.config.enabled) {
      return false;
    }

    const usage = this.getUsage(currentTokens);
    return usage.percentage >= this.config.threshold;
  }

  getContextWindow(): number {
    return this.maxTokens;
  }

  getModelName(): string {
    return this.modelName;
  }

  updateModel(modelName: string): void {
    this.modelName = modelName;
    this.maxTokens =
      OPENAI_MODEL_CONTEXT_WINDOWS[modelName] ||
      OPENAI_MODEL_CONTEXT_WINDOWS['gpt-5'] ||
      200000;
  }

  getConfig(): SummarizationConfig {
    return { ...this.config };
  }

  updateConfig(config: Partial<SummarizationConfig>): void {
    this.config = { ...this.config, ...config };
  }
}

// OpenAI Provider Implementation
export class OpenAIProvider implements ModelProvider {
  private openaiClient: OpenAI;

  constructor() {
    this.openaiClient = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
  }

  getProviderName(): string {
    return 'openai';
  }

  getDefaultModel(): string {
    return 'gpt-5';
  }

  getContextWindow(model: string): number {
    return (
      OPENAI_MODEL_CONTEXT_WINDOWS[model] ||
      OPENAI_MODEL_CONTEXT_WINDOWS['gpt-5'] ||
      200000
    );
  }

  getToolType(): any {
    // Return a dummy value - this method is for type compatibility only
    // The actual Tool type is handled at compile time
    return undefined;
  }

  createTokenCounter(
    model: string,
    config?: Partial<SummarizationConfig>,
  ): TokenCounter {
    return new OpenAITokenCounter(model, config);
  }

  async *createMessageStream(
    messages: Message[],
    model: string,
    tools: Tool[],
    maxTokens: number,
  ): AsyncIterable<MessageStreamEvent> {
    // Convert generic Tool[] to OpenAI function format
    const openaiFunctions = tools.map((tool) => ({
      type: 'function' as const,
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.input_schema,
      },
    }));

    // Convert generic Message[] to OpenAI format
    const openaiMessages = messages.map((msg) => {
      // Handle tool role messages (OpenAI-specific)
      if (msg.role === 'tool' && msg.tool_call_id) {
        return {
          role: 'tool' as const,
          tool_call_id: msg.tool_call_id,
          content: msg.content,
        };
      }
      // Handle assistant messages with tool_calls (OpenAI-specific)
      if (msg.role === 'assistant' && msg.tool_calls && msg.tool_calls.length > 0) {
        return {
          role: 'assistant' as const,
          content: msg.content || null,
          tool_calls: msg.tool_calls.map(tc => ({
            id: tc.id,
            type: 'function' as const,
            function: {
              name: tc.name,
              arguments: tc.arguments,
            },
          })),
        };
      }
      return {
        role: msg.role as 'user' | 'assistant' | 'system',
        content: msg.content,
      };
    });

    const stream = await this.openaiClient.chat.completions.create({
      model: model,
      messages: openaiMessages,
      max_completion_tokens: maxTokens,
      tools: openaiFunctions.length > 0 ? openaiFunctions : undefined,
      stream: true,
    });

    // Track tool calls by index to handle streaming chunks properly
    const toolCallTracker = new Map<number, { name?: string; id?: string; arguments: string }>();
    let messageStarted = false;

    // Convert OpenAI stream events to a normalized format
    // OpenAI uses a different event structure than Anthropic, so we normalize it
    for await (const chunk of stream) {
      const choice = chunk.choices?.[0];
      if (!choice) continue;

      // Yield message_start on first chunk (OpenAI doesn't send this, but our processor expects it)
      if (!messageStarted) {
        yield {
          type: 'message_start',
        } as MessageStreamEvent;
        messageStarted = true;
      }

      const delta = choice.delta;
      
      // Handle text content
      if (delta.content) {
        yield {
          type: 'content_block_delta',
          delta: {
            type: 'text_delta',
            text: delta.content,
          },
        } as MessageStreamEvent;
      }

      // Handle tool calls
      if (delta.tool_calls && delta.tool_calls.length > 0) {
        for (const toolCall of delta.tool_calls) {
          const index = toolCall.index;
          
          // Initialize tracker for this tool call index if not exists
          if (!toolCallTracker.has(index)) {
            toolCallTracker.set(index, { arguments: '' });
          }
          
          const tracker = toolCallTracker.get(index)!;
          
          // Handle tool call ID (comes first)
          if (toolCall.id && !tracker.id) {
            tracker.id = toolCall.id;
          }
          
          // Handle tool name (must come before arguments)
          if (toolCall.function?.name) {
            if (!tracker.name) {
              // First time we see the name - yield start event
              tracker.name = toolCall.function.name;
              yield {
                type: 'content_block_start',
                content_block: {
                  type: 'tool_use',
                  name: toolCall.function.name,
                  id: tracker.id, // Include tool call ID if available
                },
              } as MessageStreamEvent;
            }
          }
          
          // Handle tool arguments (may come in chunks, but only after we have the name)
          if (toolCall.function?.arguments && tracker.name) {
            tracker.arguments += toolCall.function.arguments;
            yield {
              type: 'content_block_delta',
              delta: {
                type: 'input_json_delta',
                partial_json: toolCall.function.arguments,
              },
            } as MessageStreamEvent;
          }
        }
      }

      // Handle finish reason
      if (choice.finish_reason) {
        // Clear tracker when message finishes
        toolCallTracker.clear();
        
        if (choice.finish_reason === 'tool_calls') {
          yield {
            type: 'message_delta',
            delta: {
              stop_reason: 'tool_use',
            },
          } as MessageStreamEvent;
        } else {
          yield {
            type: 'message_delta',
            delta: {
              stop_reason: choice.finish_reason,
            },
          } as MessageStreamEvent;
        }
        
        yield {
          type: 'message_stop',
        } as MessageStreamEvent;
      }
    }
  }

  // Helper method to create a non-streaming message (for summarization)
  async createMessage(
    messages: Message[],
    model: string,
    maxTokens: number,
  ): Promise<{ content: Array<{ type: string; text: string }> }> {
    const openaiMessages = messages.map((msg) => ({
      role: msg.role as 'user' | 'assistant' | 'system',
      content: msg.content,
    }));

    const response = await this.openaiClient.chat.completions.create({
      model: model,
      messages: openaiMessages,
      max_completion_tokens: maxTokens,
    });

    // Convert OpenAI response to a format compatible with Claude's response structure
    return {
      content: [
        {
          type: 'text',
          text: response.choices[0]?.message?.content || '',
        },
      ],
    };
  }
}

