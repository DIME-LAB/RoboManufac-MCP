import { Anthropic } from '@anthropic-ai/sdk';

import {
  StdioClientTransport,
  StdioServerParameters,
} from '@modelcontextprotocol/sdk/client/stdio.js';
import {
  ListToolsResultSchema,
  CallToolResultSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import chalk from 'chalk';
import { Tool } from '@anthropic-ai/sdk/resources/index.mjs';
import { Stream } from '@anthropic-ai/sdk/streaming.mjs';
import { consoleStyles, Logger, LoggerOptions } from './logger.js';
import { TokenCounter, SummarizationConfig } from './token-counter.js';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

type MCPClientOptions = StdioServerParameters & {
  loggerOptions?: LoggerOptions;
  summarizationConfig?: Partial<SummarizationConfig>;
  model?: string;
};

type MultiServerConfig = {
  name: string;
  config: StdioServerParameters;
};

type ServerConnection = {
  name: string;
  client: Client;
  transport: StdioClientTransport;
  tools: Tool[];
};

export class MCPClient {
  private anthropicClient: Anthropic;
  private messages: Message[] = [];
  private servers: Map<string, ServerConnection> = new Map();
  private tools: Tool[] = [];
  private logger: Logger;
  private serverConfigs: MultiServerConfig[];
  private tokenCounter: TokenCounter;
  private currentTokenCount: number = 0;
  private model: string;

  constructor(
    serverConfigs: StdioServerParameters | StdioServerParameters[],
    options?: { loggerOptions?: LoggerOptions; summarizationConfig?: Partial<SummarizationConfig>; model?: string },
  ) {
    this.anthropicClient = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
    });

    // Support both single server (backward compatibility) and multiple servers
    const configs = Array.isArray(serverConfigs) ? serverConfigs : [serverConfigs];
    this.serverConfigs = configs.map((config, index) => ({
      name: `server-${index}`,
      config,
    }));

    this.logger = new Logger(options?.loggerOptions ?? { mode: 'verbose' });
    
    // Initialize model (default to claude-haiku-4-5-20251001)
    this.model = options?.model || 'claude-haiku-4-5-20251001';
    
    // Initialize token counter
    this.tokenCounter = new TokenCounter(this.model, options?.summarizationConfig);
  }

  // Constructor for multiple named servers
  static createMultiServer(
    servers: Array<{ name: string; config: StdioServerParameters }>,
    options?: { loggerOptions?: LoggerOptions; summarizationConfig?: Partial<SummarizationConfig>; model?: string },
  ): MCPClient {
    const client = Object.create(MCPClient.prototype);
    client.anthropicClient = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
    });
    client.messages = [];
    client.servers = new Map();
    client.tools = [];
    client.logger = new Logger(options?.loggerOptions ?? { mode: 'verbose' });
    client.serverConfigs = servers;
    client.model = options?.model || 'claude-haiku-4-5-20251001';
    client.currentTokenCount = 0;
    client.tokenCounter = new TokenCounter(client.model, options?.summarizationConfig);
    return client;
  }

  async start() {
    const connectionErrors: Array<{ name: string; error: any }> = [];

    // Connect to all servers with individual error handling
    for (const serverConfig of this.serverConfigs) {
      try {
        this.logger.log(`Connecting to server "${serverConfig.name}"...\n`, {
          type: 'info',
        });

        const client = new Client(
          { name: 'cli-client', version: '1.0.0' },
          { capabilities: {} },
        );
        const transport = new StdioClientTransport(serverConfig.config);

        await client.connect(transport);
        
        // Give the server process a moment to fully initialize
        await new Promise(resolve => setTimeout(resolve, 200));

        const connection: ServerConnection = {
          name: serverConfig.name,
          client,
          transport,
          tools: [],
        };

        this.servers.set(serverConfig.name, connection);
        this.logger.log(`✓ Connected to "${serverConfig.name}"\n`, {
          type: 'info',
        });
      } catch (error) {
        connectionErrors.push({ name: serverConfig.name, error });
        this.logger.log(
          `✗ Failed to connect to "${serverConfig.name}": ${error}\n`,
          { type: 'warning' },
        );
        // Continue with other servers
      }
    }

    // Check if we have at least one successful connection
    if (this.servers.size === 0) {
      this.logger.log(
        'Failed to connect to any servers. Please check your server configurations.\n',
        { type: 'error' },
      );
      if (connectionErrors.length > 0) {
        this.logger.log('Connection errors:\n', { type: 'error' });
        connectionErrors.forEach(({ name, error }) => {
          this.logger.log(`  ${name}: ${error}\n`, { type: 'error' });
        });
      }
      process.exit(1);
    }

    // Log warnings for failed connections
    if (connectionErrors.length > 0) {
      this.logger.log(
        `Warning: ${connectionErrors.length} server(s) failed to connect, continuing with ${this.servers.size} server(s)\n`,
        { type: 'warning' },
      );
    }

    // Initialize tools from all successfully connected servers
    await this.initMCPTools();
    
    this.logger.log(
      `Connected to ${this.servers.size} server(s): ${Array.from(this.servers.keys()).join(', ')}\n`,
      { type: 'info' },
    );
  }

  async stop() {
    const closePromises = Array.from(this.servers.values()).map((connection) =>
      connection.client.close().catch(() => {
        // Ignore errors during cleanup
      }),
    );
    await Promise.all(closePromises);
    this.servers.clear();
  }

  private async initMCPTools() {
    const allTools: Tool[] = [];

    // Load tools from each server and prefix with server name
    for (const [serverName, connection] of this.servers.entries()) {
      try {
        const toolsResults = await connection.client.request(
          { method: 'tools/list' },
          ListToolsResultSchema,
        );

        const serverTools = toolsResults.tools.map(
          ({ inputSchema, name, description }) => {
            // Prefix tool name with server name to avoid conflicts
            // Use double underscore as separator (colon not allowed in Anthropic tool names)
            const prefixedName = `${serverName}__${name}`;
            return {
              name: prefixedName,
              description: `[${serverName}] ${description}`,
              input_schema: inputSchema,
            };
          },
        );

        connection.tools = serverTools;
        allTools.push(...serverTools);
      } catch (error) {
        this.logger.log(
          `Failed to load tools from server "${serverName}": ${error}\n`,
          { type: 'warning' },
        );
      }
    }

    this.tools = allTools;
    this.logger.log(
      `Loaded ${allTools.length} tool(s) from ${this.servers.size} server(s)\n`,
      { type: 'info' },
    );
  }

  private formatToolCall(toolName: string, args: any): string {
    return (
      '\n' +
      consoleStyles.tool.bracket('[') +
      consoleStyles.tool.name(toolName) +
      consoleStyles.tool.bracket('] ') +
      consoleStyles.tool.args(JSON.stringify(args, null, 2)) +
      '\n'
    );
  }

  private formatJSON(json: string): string {
    return json
      .replace(/"([^"]+)":/g, chalk.blue('"$1":'))
      .replace(/: "([^"]+)"/g, ': ' + chalk.green('"$1"'));
  }

  private shouldSummarize(): boolean {
    return this.tokenCounter.shouldSummarize(this.currentTokenCount);
  }

  // Public method to get token usage status (for testing/debugging)
  getTokenUsage() {
    return this.tokenCounter.getUsage(this.currentTokenCount);
  }

  // Public method to manually trigger summarization (for testing)
  async manualSummarize(): Promise<void> {
    await this.autoSummarize();
  }

  // Public method to set test mode (lower threshold for easier testing)
  setTestMode(enabled: boolean = true, testThreshold: number = 5) {
    if (enabled) {
      this.tokenCounter.updateConfig({
        threshold: testThreshold, // Very low threshold for testing
        enabled: true,
      });
      this.logger.log(
        `\n🧪 Test mode enabled: Summarization will trigger at ${testThreshold}% (${Math.round(this.tokenCounter.getContextWindow() * testThreshold / 100)} tokens)\n`,
        { type: 'info' },
      );
    } else {
      this.tokenCounter.updateConfig({
        threshold: 80, // Back to normal
      });
      this.logger.log('\n🧪 Test mode disabled: Summarization threshold reset to 80%\n', {
        type: 'info',
      });
    }
  }

  private async autoSummarize(): Promise<void> {
    if (!this.tokenCounter.getConfig().enabled) {
      return;
    }

    const config = this.tokenCounter.getConfig();
    const recentCount = config.recentMessagesToKeep;

    // Need at least recentCount + 1 messages to summarize
    if (this.messages.length <= recentCount) {
      return;
    }

    this.logger.log(
      `\n⚠️ Context window approaching limit (${this.tokenCounter.getUsage(this.currentTokenCount).percentage}% used). Summarizing conversation...\n`,
      { type: 'warning' },
    );

    try {
      // Keep recent messages
      const recentMessages = this.messages.slice(-recentCount);
      const oldMessages = this.messages.slice(0, -recentCount);

      // Create summarization prompt
      const messagesToSummarize = oldMessages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      // Call API to summarize
      const summaryResponse = await this.anthropicClient.messages.create({
        model: this.model,
        max_tokens: 2000,
        messages: [
          ...messagesToSummarize,
          {
            role: 'user',
            content:
              'Summarize the above conversation concisely, preserving key decisions, context, important information, and any tool usage patterns. Focus on what was accomplished and what context is needed to continue the conversation.',
          },
        ],
      });

      const summaryText =
        summaryResponse.content[0]?.type === 'text'
          ? summaryResponse.content[0].text
          : JSON.stringify(summaryResponse.content);

      // Recalculate token count
      // Remove old messages from count
      let oldTokenCount = 0;
      for (const msg of oldMessages) {
        oldTokenCount += this.tokenCounter.countMessageTokens(msg);
      }

      // Count summary message
      const summaryMessage: Message = {
        role: 'user',
        content: `[Previous conversation summary: ${summaryText}]`,
      };
      const summaryTokenCount =
        this.tokenCounter.countMessageTokens(summaryMessage);

      // Update messages and token count
      this.messages = [summaryMessage, ...recentMessages];
      this.currentTokenCount =
        this.currentTokenCount - oldTokenCount + summaryTokenCount;

      this.logger.log(
        `✓ Conversation summarized. Context reduced from ${oldMessages.length} to 1 summary message. Token usage: ${this.tokenCounter.getUsage(this.currentTokenCount).percentage}%\n`,
        { type: 'info' },
      );
    } catch (error) {
      this.logger.log(
        `Failed to summarize conversation: ${error}\n`,
        { type: 'error' },
      );
      // Continue without summarization - let API handle the limit
    }
  }

  private async processStream(
    stream: Stream<Anthropic.Messages.RawMessageStreamEvent>,
  ): Promise<void> {
    let currentMessage = '';
    let currentToolName = '';
    let currentToolInputString = '';
    let assistantMessageAdded = false;

    this.logger.log(consoleStyles.assistant);
    for await (const chunk of stream) {
      switch (chunk.type) {
        case 'message_start':
          // Reset flags for new message
          assistantMessageAdded = false;
          currentMessage = '';
          currentToolName = '';
          currentToolInputString = '';
          continue;

        case 'content_block_stop':
          continue;

        case 'content_block_start':
          if (chunk.content_block?.type === 'tool_use') {
            currentToolName = chunk.content_block.name;
          }
          break;

        case 'content_block_delta':
          if (chunk.delta.type === 'text_delta') {
            this.logger.log(chunk.delta.text);
            currentMessage += chunk.delta.text;
          } else if (chunk.delta.type === 'input_json_delta') {
            if (currentToolName && chunk.delta.partial_json) {
              currentToolInputString += chunk.delta.partial_json;
            }
          }
          break;

        case 'message_delta':
          // Only add assistant message once when we have content and haven't added it yet
          if (currentMessage && !assistantMessageAdded) {
            const assistantMessage: Message = {
              role: 'assistant',
              content: currentMessage,
            };
            this.messages.push(assistantMessage);
            assistantMessageAdded = true;
            // Count tokens for assistant message
            this.currentTokenCount += this.tokenCounter.countMessageTokens(assistantMessage);
          }

          if (chunk.delta.stop_reason === 'tool_use') {
            let toolArgs = {};
            try {
              toolArgs = currentToolInputString
                ? JSON.parse(currentToolInputString)
                : {};
            } catch (parseError) {
              // JSON parsing failed - feed error back to agent so it can fix it
              const errorMessage: Message = {
                role: 'user',
                content: `Error parsing tool arguments for "${currentToolName}": ${parseError instanceof Error ? parseError.message : String(parseError)}\n\nInvalid JSON: ${currentToolInputString}\n\nPlease fix the tool call with valid JSON arguments.`,
              };
              this.messages.push(errorMessage);
              this.currentTokenCount += this.tokenCounter.countMessageTokens(errorMessage);
              
              this.logger.log(
                `\n⚠️ JSON parse error for tool "${currentToolName}": ${parseError instanceof Error ? parseError.message : String(parseError)}\n`,
                { type: 'error' },
              );
              
              // Continue conversation so agent can see the error and fix it
              const errorStream = await this.anthropicClient.messages.create({
                messages: this.messages,
                model: this.model,
                max_tokens: 8192,
                tools: this.tools,
                stream: true,
              });
              await this.processStream(errorStream);
              return; // Exit early since we've handled the error
            }

            this.logger.log(
              this.formatToolCall(currentToolName, toolArgs) + '\n',
            );

            // Extract server name and actual tool name from prefixed name
            // Format: "server-name__tool-name" (double underscore separator)
            const [serverName, actualToolName] = currentToolName.includes('__')
              ? currentToolName.split('__', 2)
              : [null, currentToolName];

            let toolResult;
            try {
              if (serverName && this.servers.has(serverName)) {
                // Route to the specific server
                const connection = this.servers.get(serverName)!;
                toolResult = await connection.client.request(
                  {
                    method: 'tools/call',
                    params: {
                      name: actualToolName,
                      arguments: toolArgs,
                    },
                  },
                  CallToolResultSchema,
                );
              } else {
                // Fallback: try to find the tool in any server (backward compatibility)
                let found = false;
                for (const [name, connection] of this.servers.entries()) {
                  const tool = connection.tools.find((t) => t.name === currentToolName || t.name.endsWith(`__${currentToolName}`));
                  if (tool) {
                    const actualName = tool.name.includes('__') ? tool.name.split('__')[1] : tool.name;
                    toolResult = await connection.client.request(
                      {
                        method: 'tools/call',
                        params: {
                          name: actualName,
                          arguments: toolArgs,
                        },
                      },
                      CallToolResultSchema,
                    );
                    found = true;
                    break;
                  }
                }
                if (!found || !toolResult) {
                  throw new Error(`Tool "${currentToolName}" not found in any server`);
                }
              }
            } catch (toolError) {
              // Tool execution failed - feed error back to agent so it can handle it
              const errorMessage: Message = {
                role: 'user',
                content: `Error executing tool "${currentToolName}": ${toolError instanceof Error ? toolError.message : String(toolError)}\n\nTool arguments: ${JSON.stringify(toolArgs, null, 2)}\n\nPlease handle this error and continue.`,
              };
              this.messages.push(errorMessage);
              this.currentTokenCount += this.tokenCounter.countMessageTokens(errorMessage);
              
              this.logger.log(
                `\n⚠️ Tool execution error for "${currentToolName}": ${toolError instanceof Error ? toolError.message : String(toolError)}\n`,
                { type: 'error' },
              );
              
              // Continue conversation so agent can see the error and handle it
              const errorStream = await this.anthropicClient.messages.create({
                messages: this.messages,
                model: this.model,
                max_tokens: 8192,
                tools: this.tools,
                stream: true,
              });
              await this.processStream(errorStream);
              return; // Exit early since we've handled the error
            }

            const formattedResult = this.formatJSON(
              JSON.stringify(toolResult.content.flatMap((c) => c.text)),
            );

            const toolResultMessage: Message = {
              role: 'user',
              content: formattedResult,
            };
            this.messages.push(toolResultMessage);
            // Count tokens for tool result message
            this.currentTokenCount += this.tokenCounter.countMessageTokens(toolResultMessage);

            // Check if we need to summarize before continuing
            if (this.shouldSummarize()) {
              await this.autoSummarize();
            }

            const nextStream = await this.anthropicClient.messages.create({
              messages: this.messages,
              model: this.model,
              max_tokens: 8192,
              tools: this.tools,
              stream: true,
            });
            await this.processStream(nextStream);
          }
          break;

        case 'message_stop':
          // Ensure assistant message is added if it wasn't added in message_delta
          if (currentMessage && !assistantMessageAdded) {
            const assistantMessage: Message = {
              role: 'assistant',
              content: currentMessage,
            };
            this.messages.push(assistantMessage);
            assistantMessageAdded = true;
            // Count tokens for assistant message
            this.currentTokenCount += this.tokenCounter.countMessageTokens(assistantMessage);
          }
          break;

        default:
          this.logger.log(`Unknown event type: ${JSON.stringify(chunk)}\n`, {
            type: 'warning',
          });
      }
    }
  }

  async processQuery(query: string) {
    try {
      // Check if we need to summarize before adding new message
      if (this.shouldSummarize()) {
        await this.autoSummarize();
      }

      const userMessage: Message = { role: 'user', content: query };
      this.messages.push(userMessage);
      
      // Count tokens for user message
      this.currentTokenCount += this.tokenCounter.countMessageTokens(userMessage);
      
      // Log token usage after each message (for testing/debugging)
      const usage = this.tokenCounter.getUsage(this.currentTokenCount);
      this.logger.log(
        `[Token usage: ${usage.current}/${usage.limit} (${usage.percentage}%)]\n`,
        { type: 'info' },
      );

      // Check again after adding message (in case we're very close to limit)
      if (this.shouldSummarize()) {
        await this.autoSummarize();
      }

      const stream = await this.anthropicClient.messages.create({
        messages: this.messages,
        model: this.model,
        max_tokens: 8192,
        tools: this.tools,
        stream: true,
      });
      await this.processStream(stream);

      return this.messages;
    } catch (error) {
      this.logger.log('\nError during query processing: ' + error + '\n', {
        type: 'error',
      });
      if (error instanceof Error) {
        this.logger.log(
          consoleStyles.assistant +
            'I apologize, but I encountered an error: ' +
            error.message +
            '\n',
        );
      }
    }
  }
}
