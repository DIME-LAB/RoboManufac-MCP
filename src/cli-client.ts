import { StdioServerParameters } from '@modelcontextprotocol/sdk/client/stdio.js';
import { ListToolsResultSchema } from '@modelcontextprotocol/sdk/types.js';
import readline from 'readline/promises';
import { MCPClient } from './index.js';
import { consoleStyles, Logger } from './logger.js';

const EXIT_COMMAND = 'exit';

export class MCPClientCLI {
  private rl: readline.Interface | null = null;
  private client: MCPClient;
  private logger: Logger;
  private isShuttingDown = false;

  constructor(
    serverConfig: StdioServerParameters | Array<{ name: string; config: StdioServerParameters }>,
  ) {
    if (Array.isArray(serverConfig)) {
      // Multiple servers
      this.client = MCPClient.createMultiServer(serverConfig);
    } else {
      // Single server (backward compatibility)
      this.client = new MCPClient(serverConfig);
    }
    this.logger = new Logger({ mode: 'verbose' });
    
    // Set up signal handlers for graceful shutdown
    this.setupSignalHandlers();
  }

  private setupSignalHandlers() {
    const cleanup = async () => {
      if (this.isShuttingDown) {
        return;
      }
      this.isShuttingDown = true;
      
      this.logger.log('\n\nShutting down gracefully...\n', { type: 'info' });
      
      try {
        // Close readline first
        if (this.rl) {
          this.rl.close();
          this.rl = null;
        }
        
        // Close MCP client connection
        await this.client.stop();
      } catch (error) {
        // Ignore errors during cleanup
      }
      
      process.exit(0);
    };

    // Handle SIGINT (Ctrl+C)
    process.on('SIGINT', () => {
      void cleanup();
    });

    // Handle SIGTERM
    process.on('SIGTERM', () => {
      void cleanup();
    });
  }

  async start() {
    try {
      this.logger.log(consoleStyles.separator + '\n', { type: 'info' });
      this.logger.log('🤖 Interactive Claude CLI\n', { type: 'info' });
      this.logger.log(`Type your queries or "${EXIT_COMMAND}" to exit\n`, {
        type: 'info',
      });
      this.logger.log(
        `\nTesting commands:\n` +
        `  /token-status or /tokens - Show current token usage\n` +
        `  /summarize or /summarize-now - Manually trigger summarization\n` +
        `  /test-mode [threshold] - Enable test mode (default: 5% threshold)\n` +
        `  /test-mode off - Disable test mode\n` +
        `  /todo-on - Enable todo mode (agent will track tasks)\n` +
        `  /todo-off - Disable todo mode\n` +
        `  /tools or /tools-select - Interactive tool selection mode\n` +
        `  /tools-enable-all - Enable all tools from all servers\n` +
        `  /tools-disable-all - Disable all tools from all servers\n` +
        `  /tools-list - List all tools with their enabled/disabled status\n` +
        `  /tools-enable-server <server-name> - Enable all tools from a server\n` +
        `  /tools-disable-server <server-name> - Disable all tools from a server\n`,
        { type: 'info' },
      );
      this.logger.log(consoleStyles.separator + '\n', { type: 'info' });
      
      // Wait for MCP client to fully connect before creating readline
      await this.client.start();
      
      // Create readline interface after MCP connection is established
      this.rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
      });

      await this.chat_loop();
    } catch (error) {
      if (!this.isShuttingDown) {
        this.logger.log('Failed to initialize tools: ' + error + '\n', {
          type: 'error',
        });
      }
    } finally {
      await this.cleanup();
    }
  }

  private async cleanup() {
    if (this.isShuttingDown) {
      return;
    }
    this.isShuttingDown = true;

    try {
      if (this.rl) {
        this.rl.close();
        this.rl = null;
      }
      
      await this.client.stop();
    } catch (error) {
      // Ignore errors during cleanup
    }
  }

  private async chat_loop() {
    if (!this.rl) {
      throw new Error('Readline interface not initialized');
    }
    
    while (true) {
      try {
        if (this.isShuttingDown) {
          break;
        }
        
        const query = (await this.rl.question(consoleStyles.prompt)).trim();
        
        if (this.isShuttingDown) {
          break;
        }
        
        if (query.toLowerCase() === EXIT_COMMAND) {
          this.logger.log('\nGoodbye! 👋\n', { type: 'warning' });
          break;
        }

        // Handle special commands for testing
        if (query.toLowerCase() === '/token-status' || query.toLowerCase() === '/tokens') {
          const usage = this.client.getTokenUsage();
          this.logger.log(
            `\n📊 Token Usage Status:\n` +
            `  Current: ${usage.current} tokens\n` +
            `  Limit: ${usage.limit} tokens\n` +
            `  Usage: ${usage.percentage}%\n` +
            `  Status: ${usage.suggestion}\n` +
            `  Messages: ${this.client['messages'].length}\n`,
            { type: 'info' },
          );
          continue;
        }

        if (query.toLowerCase() === '/summarize' || query.toLowerCase() === '/summarize-now') {
          this.logger.log('\n🔧 Manually triggering summarization...\n', { type: 'info' });
          await this.client.manualSummarize();
          const usage = this.client.getTokenUsage();
          this.logger.log(
            `\n📊 Token Usage After Summarization:\n` +
            `  Current: ${usage.current} tokens\n` +
            `  Usage: ${usage.percentage}%\n`,
            { type: 'info' },
          );
          continue;
        }

        if (query.toLowerCase().startsWith('/test-mode')) {
          const parts = query.split(' ');
          if (parts.length > 1 && parts[1] === 'off') {
            this.client.setTestMode(false);
          } else {
            const threshold = parts.length > 1 ? parseFloat(parts[1]) : 5;
            this.client.setTestMode(true, threshold);
          }
          continue;
        }

        if (query.toLowerCase() === '/todo-on') {
          try {
            if (!this.client.isTodoServerConfigured()) {
              this.logger.log(
                '\nTodo server not configured. Please add "todo" server to mcp_config.json before using this feature.\n',
                { type: 'error' },
              );
              continue;
            }
            
            await this.client.enableTodoMode();
            
            // Send system prompt to agent (marked as system prompt so it doesn't trigger clear)
            const systemPrompt = 'You are now in todo mode. When the user provides a task, you must: 1) Decompose the task into actionable todos using create-todo, 2) As you complete each task, mark it complete using complete-todo. You cannot exit until all todos are completed or skipped using skip-todo.';
            await this.client.processQuery(systemPrompt, true);
            
            // Mark todo mode as initialized so future user queries will auto-clear
            this.client.setTodoModeInitialized(true);
          } catch (error) {
            this.logger.log(
              `\nFailed to enable todo mode: ${error}\n`,
              { type: 'error' },
            );
          }
          continue;
        }

        if (query.toLowerCase() === '/todo-off') {
          try {
            await this.client.disableTodoMode();
          } catch (error) {
            this.logger.log(
              `\nFailed to disable todo mode: ${error}\n`,
              { type: 'error' },
            );
          }
          continue;
        }

        // Handle tool management commands
        if (query.toLowerCase() === '/tools-enable-all') {
          try {
            await this.client.enableAllTools();
          } catch (error) {
            this.logger.log(
              `\nFailed to enable all tools: ${error}\n`,
              { type: 'error' },
            );
          }
          continue;
        }

        if (query.toLowerCase() === '/tools-disable-all') {
          try {
            await this.client.disableAllTools();
          } catch (error) {
            this.logger.log(
              `\nFailed to disable all tools: ${error}\n`,
              { type: 'error' },
            );
          }
          continue;
        }

        if (query.toLowerCase() === '/tools-list') {
          try {
            await this.displayToolsList();
          } catch (error) {
            this.logger.log(
              `\nFailed to list tools: ${error}\n`,
              { type: 'error' },
            );
          }
          continue;
        }

        if (query.toLowerCase().startsWith('/tools-enable-server')) {
          try {
            const parts = query.split(' ');
            if (parts.length < 2) {
              this.logger.log(
                '\nUsage: /tools-enable-server <server-name>\n',
                { type: 'error' },
              );
              continue;
            }
            const serverName = parts.slice(1).join(' ');
            await this.client.enableServerTools(serverName);
          } catch (error) {
            this.logger.log(
              `\nFailed to enable server tools: ${error}\n`,
              { type: 'error' },
            );
          }
          continue;
        }

        if (query.toLowerCase().startsWith('/tools-disable-server')) {
          try {
            const parts = query.split(' ');
            if (parts.length < 2) {
              this.logger.log(
                '\nUsage: /tools-disable-server <server-name>\n',
                { type: 'error' },
              );
              continue;
            }
            const serverName = parts.slice(1).join(' ');
            await this.client.disableServerTools(serverName);
          } catch (error) {
            this.logger.log(
              `\nFailed to disable server tools: ${error}\n`,
              { type: 'error' },
            );
          }
          continue;
        }

        if (query.toLowerCase() === '/tools' || query.toLowerCase() === '/tools-select') {
          try {
            await this.interactiveToolSelection();
          } catch (error) {
            this.logger.log(
              `\nFailed to open tool selection: ${error}\n`,
              { type: 'error' },
            );
          }
          continue;
        }

        await this.client.processQuery(query);
        this.logger.log('\n' + consoleStyles.separator + '\n');
      } catch (error: any) {
        // Check if readline was closed (happens during shutdown)
        if (error?.code === 'ERR_USE_AFTER_CLOSE' || this.isShuttingDown) {
          break;
        }
        this.logger.log('\nError: ' + error + '\n', { type: 'error' });
      }
    }
  }

  private async displayToolsList(): Promise<void> {
    const toolManager = this.client.getToolManager();
    const toolStates = toolManager.getToolStates();
    
    // Get all tools from all servers
    const allTools: Array<{ name: string; server: string; enabled: boolean }> = [];
    
    // Access private servers map through a workaround
    const servers = (this.client as any).servers as Map<string, any>;
    
    for (const [serverName, connection] of servers.entries()) {
      // Get all tools from server (including disabled ones)
      try {
        const toolsResults = await connection.client.request(
          { method: 'tools/list' },
          ListToolsResultSchema,
        );
        
        for (const tool of toolsResults.tools) {
          const prefixedName = `${serverName}__${tool.name}`;
          const enabled = toolManager.isToolEnabled(prefixedName);
          allTools.push({
            name: tool.name,
            server: serverName,
            enabled,
          });
        }
      } catch (error) {
        // Ignore errors for individual servers
      }
    }
    
    // Group by server
    const toolsByServer = new Map<string, Array<{ name: string; enabled: boolean }>>();
    for (const tool of allTools) {
      if (!toolsByServer.has(tool.server)) {
        toolsByServer.set(tool.server, []);
      }
      toolsByServer.get(tool.server)!.push({ name: tool.name, enabled: tool.enabled });
    }
    
    this.logger.log('\n📋 Tools Status:\n', { type: 'info' });
    
    for (const [serverName, tools] of toolsByServer.entries()) {
      const enabledCount = tools.filter(t => t.enabled).length;
      this.logger.log(
        `\n[${serverName}] (${enabledCount}/${tools.length} enabled):\n`,
        { type: 'info' },
      );
      
      for (const tool of tools) {
        const status = tool.enabled ? '✓' : '✗';
        const statusColor = tool.enabled ? 'green' : 'red';
        this.logger.log(
          `  ${status} ${tool.name}\n`,
          { type: tool.enabled ? 'info' : 'warning' },
        );
      }
    }
    
    this.logger.log('\n');
  }

  private async interactiveToolSelection(): Promise<void> {
    if (!this.rl) {
      throw new Error('Readline interface not initialized');
    }
    
    const toolManager = this.client.getToolManager();
    
    // Collect all tools from all servers
    const allTools: Array<{ name: string; server: string; toolName: string; enabled: boolean }> = [];
    const servers = (this.client as any).servers as Map<string, any>;
    const serverList: string[] = [];
    
    for (const [serverName, connection] of servers.entries()) {
      serverList.push(serverName);
      try {
        const toolsResults = await connection.client.request(
          { method: 'tools/list' },
          ListToolsResultSchema,
        );
        
        for (const tool of toolsResults.tools) {
          const prefixedName = `${serverName}__${tool.name}`;
          const enabled = toolManager.isToolEnabled(prefixedName);
          allTools.push({
            name: tool.name,
            server: serverName,
            toolName: prefixedName,
            enabled,
          });
        }
      } catch (error) {
        // Ignore errors
      }
    }
    
    // Update state for new tools
    const toolObjects = allTools.map(t => ({
      name: t.toolName,
      description: `[${t.server}] ${t.name}`,
      input_schema: {},
    }));
    toolManager.updateStateForNewTools(toolObjects as any);
    
    // Create index mapping
    const indexToTool = new Map<number, typeof allTools[0]>();
    let toolIndex = 1;
    
    // Group tools by server
    const toolsByServer = new Map<string, typeof allTools>();
    for (const tool of allTools) {
      if (!toolsByServer.has(tool.server)) {
        toolsByServer.set(tool.server, []);
      }
      toolsByServer.get(tool.server)!.push(tool);
    }
    
    const sortedServers = Array.from(toolsByServer.entries()).sort((a, b) => 
      a[0].localeCompare(b[0])
    );
    
    // Clear screen before entering the loop
    process.stdout.write('\x1B[2J\x1B[0f');
    
    while (true) {
      // Clear and display
      process.stdout.write('\x1B[2J\x1B[0f'); // Clear screen
      
      // Use a single write to avoid duplication issues
      let displayText = '\n🔧 Tool Selection\n';
      displayText += 'Available Servers and Tools:\n';
      
      toolIndex = 1;
      indexToTool.clear();
      
      for (let serverIdx = 0; serverIdx < sortedServers.length; serverIdx++) {
        const [serverName, serverTools] = sortedServers[serverIdx];
        const enabledCount = serverTools.filter(t => t.enabled).length;
        const totalCount = serverTools.length;
        
        let serverStatus = '✓';
        if (enabledCount === 0) {
          serverStatus = '✗';
        } else if (enabledCount < totalCount) {
          serverStatus = '~';
        }
        
        displayText += `\nS${serverIdx + 1}. ${serverStatus} [${serverName}] (${enabledCount}/${totalCount} enabled):\n`;
        
        for (const tool of serverTools) {
          const status = tool.enabled ? '✓' : '✗';
          displayText += `  ${toolIndex}. ${status} ${tool.name}\n`;
          indexToTool.set(toolIndex, tool);
          toolIndex++;
        }
      }
      
      displayText += `\nCommands:\n` +
        `  Enter numbers separated by commas or ranges (e.g., 1,3,5-8) to toggle tools\n` +
        `  Enter S + number (e.g., S1, s2) to toggle all tools in a server\n` +
        `  a or all - Enable all tools\n` +
        `  n or none - Disable all tools\n` +
        `  s or save - Save changes and return\n` +
        `  q or quit - Cancel and return\n`;
      
      // Write everything at once to avoid duplication
      process.stdout.write(displayText);
      
      const selection = (await this.rl.question('> ')).trim().toLowerCase();
      
      if (selection === 's' || selection === 'save') {
        // Reload tools to apply changes
        await (this.client as any).initMCPTools();
        this.logger.log('\n✓ Changes saved\n', { type: 'info' });
        break;
      }
      
      if (selection === 'q' || selection === 'quit') {
        // Reload original state
        toolManager.loadState();
        this.logger.log('\n✗ Changes cancelled\n', { type: 'warning' });
        break;
      }
      
      if (selection === 'a' || selection === 'all') {
        const toolObjects = allTools.map(t => ({
          name: t.toolName,
          description: `[${t.server}] ${t.name}`,
          input_schema: {},
        }));
        toolManager.enableAllTools(toolObjects as any);
        // Update all enabled statuses
        for (const tool of allTools) {
          tool.enabled = true;
        }
        continue;
      }
      
      if (selection === 'n' || selection === 'none') {
        const toolObjects = allTools.map(t => ({
          name: t.toolName,
          description: `[${t.server}] ${t.name}`,
          input_schema: {},
        }));
        toolManager.disableAllTools(toolObjects as any);
        // Update all enabled statuses
        for (const tool of allTools) {
          tool.enabled = false;
        }
        continue;
      }
      
      // Handle server toggle (S1, s2, etc.)
      if (selection.match(/^s\d+$/i)) {
        const serverNum = parseInt(selection.slice(1)) - 1;
        if (serverNum >= 0 && serverNum < sortedServers.length) {
          const [serverName, serverTools] = sortedServers[serverNum];
          const allEnabled = serverTools.every(t => t.enabled);
          const newState = !allEnabled;
          
          for (const tool of serverTools) {
            toolManager.setToolEnabled(tool.toolName, newState);
            // Update the enabled status in allTools array
            tool.enabled = newState;
          }
          
          // Continue loop to refresh display immediately
          continue;
        }
      }
      
      // Handle tool number selection
      if (selection.match(/^[\d,\-\s]+$/)) {
        const parts = selection.split(',').map(p => p.trim());
        const indices: number[] = [];
        
        for (const part of parts) {
          if (part.includes('-')) {
            const [start, end] = part.split('-').map(n => parseInt(n.trim()));
            if (!isNaN(start) && !isNaN(end)) {
              for (let i = start; i <= end; i++) {
                indices.push(i);
              }
            }
          } else {
            const num = parseInt(part);
            if (!isNaN(num)) {
              indices.push(num);
            }
          }
        }
        
        let toggledCount = 0;
        for (const idx of indices) {
          if (indexToTool.has(idx)) {
            const tool = indexToTool.get(idx)!;
            toolManager.toggleTool(tool.toolName);
            // Update the enabled status in allTools array
            tool.enabled = toolManager.isToolEnabled(tool.toolName);
            toggledCount++;
          }
        }
        
        if (toggledCount > 0) {
          // Continue loop to refresh display immediately
          continue;
        }
      }
      
      this.logger.log('\nInvalid selection. Please try again.\n', { type: 'error' });
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
}
