"""
Entry point for MCP Scheduler.
"""
import asyncio
import argparse
import logging
import os
import signal
import sys
import traceback
import threading
import json
import re

from mcp_scheduler.config import Config
from mcp_scheduler.persistence import Database
from mcp_scheduler.executor import Executor
from mcp_scheduler.scheduler import Scheduler
from mcp_scheduler.server import SchedulerServer
from mcp_scheduler.utils import setup_logging

# For well-known endpoint
from mcp_scheduler.well_known import setup_well_known
import aiohttp

# Import our custom JSON parser utilities
try:
    from mcp_scheduler.json_parser import patch_fastmcp_parser, install_stdio_wrapper
except ImportError:
    # Define dummies if the module isn't available
    def patch_fastmcp_parser():
        return False
    def install_stdio_wrapper():
        return False

# Global variables for cleanup
scheduler = None
server = None
scheduler_task = None

def log_to_stderr(message):
    """Log messages to stderr instead of stdout to avoid interfering with stdio transport."""
    print(message, file=sys.stderr, flush=True)

def run_scheduler_in_thread(scheduler_instance):
    """Run the scheduler in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def scheduler_loop():
        await scheduler_instance.start()
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            await scheduler_instance.stop()
    
    loop.run_until_complete(scheduler_loop())
    loop.close()

def handle_sigterm(signum, frame):
    """Handle SIGTERM gracefully"""
    log_to_stderr("Received SIGTERM signal. Shutting down...")
    if scheduler:
        try:
            # Create a new event loop for clean shutdown
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(scheduler.stop())
            loop.close()
        except Exception as e:
            log_to_stderr(f"Error during shutdown: {e}")
    sys.exit(0)

# Custom stdout wrapper to ensure only JSON goes to stdout
class JSONRPCStdout:
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        
    def write(self, data):
        # Only allow JSON data through stdout for MCP protocol
        if data and data.strip():
            # Check if it looks like JSON (starts with { or [)
            if data.strip().startswith('{') or data.strip().startswith('['):
                self.original_stdout.write(data)
                self.original_stdout.flush()
            else:
                # Non-JSON data should go to stderr
                sys.stderr.write(f"Attempted to write non-JSON to stdout: {data[:100]}\n")
                sys.stderr.flush()
        
    def flush(self):
        self.original_stdout.flush()
        
    def __getattr__(self, name):
        return getattr(self.original_stdout, name)

# Custom input wrapper to handle malformed JSON
class SafeJsonStdin:
    def __init__(self, original_stdin):
        self.original_stdin = original_stdin
        
    def readline(self):
        line = self.original_stdin.readline()
        if not line:
            return line
            
        # Skip obvious non-JSON lines (file paths, Python code, etc.)
        stripped_line = line.strip()
        if (stripped_line.startswith('D:/') or 
            stripped_line.startswith('C:/') or 
            stripped_line.startswith('/') or
            '.py' in stripped_line or
            '.venv' in stripped_line or
            (not stripped_line.startswith('{') and not stripped_line.startswith('['))):
            log_to_stderr(f"Skipping non-JSON input: {stripped_line[:100]}...")
            return '\n'  # Return empty line to avoid blocking
            
        # Only try to fix lines that look like JSON
        if stripped_line.startswith('{') or stripped_line.startswith('['):
            try:
                # Parse the JSON to ensure it's valid
                json.loads(stripped_line)
                return line
            except json.JSONDecodeError as e:
                log_to_stderr(f"Fixing malformed JSON input: {e}")
                # Try to extract the ID if present
                try:
                    id_match = re.search(r'"id"\s*:\s*(\d+)', stripped_line)
                    if id_match:
                        id_val = id_match.group(1)
                        return f'{{"jsonrpc":"2.0","id":{id_val},"error":{{"code":-32700,"message":"Parse error"}}}}\n'
                    # Default response for malformed JSON
                    return '{"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"Parse error"}}\n'
                except Exception as ex:
                    log_to_stderr(f"Error creating error response: {ex}")
                    # Ultimate fallback
                    return '{"jsonrpc":"2.0","id":null,"error":{"code":-32603,"message":"Internal error"}}\n'
                    
        return line
        
    def __getattr__(self, name):
        return getattr(self.original_stdin, name)

def start_well_known_server():
    app = aiohttp.web.Application()
    setup_well_known(app)
    # Use the same address/port as the MCP server if possible
    from mcp_scheduler.config import Config
    config = Config()
    aiohttp.web.run_app(app, host=config.server_address, port=config.server_port + 1)

def main():
    """Main entry point."""
    # Save original stdout for JSON-RPC communication
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    parser = argparse.ArgumentParser(description="MCP Scheduler Server")
    parser.add_argument(
        "--address", 
        default=None,
        help="Server address (default: localhost or from config/env)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=None,
        help="Server port (default: 8080 or from config/env)"
    )
    parser.add_argument(
        "--transport", 
        choices=["sse", "stdio"], 
        default=None,
        help="Transport mode (sse or stdio) (default: sse or from config/env)"
    )
    parser.add_argument(
        "--log-level", 
        default=None,
        help="Logging level (default: INFO or from config/env)"
    )
    parser.add_argument(
        "--log-file", 
        default=None,
        help="Log file path (default: None or from config/env)"
    )
    parser.add_argument(
        "--db-path", 
        default=None,
        help="SQLite database path (default: scheduler.db or from config/env)"
    )
    parser.add_argument(
        "--config", 
        default=None,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--ai-model", 
        default=None,
        help="AI model to use for AI tasks (default: gpt-4o or from config/env)"
    )
    parser.add_argument(
        "--version", 
        action="store_true",
        help="Show version and exit"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with full traceback"
    )
    parser.add_argument(
        "--fix-json", 
        action="store_true",
        help="Enable JSON fixing for malformed messages"
    )
    parser.add_argument(
        "--ai-provider", 
        default=None,
        choices=["openai", "azure", "anthropic", "local"],
        help="AI provider (openai, azure, anthropic, local) (default: openai)"
    )
    parser.add_argument(
        "--openai-base-url", 
        default=None,
        help="OpenAI compatible API base URL (default: https://api.openai.com/v1)"
    )
    parser.add_argument(
        "--openai-api-key", 
        default=None,
        help="OpenAI compatible API key"
    )
    
    args = parser.parse_args()
    
    # Set environment variables for config
    if args.ai_provider:
        os.environ["MCP_SCHEDULER_AI_PROVIDER"] = args.ai_provider

    if args.openai_base_url:
        os.environ["MCP_SCHEDULER_OPENAI_BASE_URL"] = args.openai_base_url

    if args.openai_api_key:
        os.environ["MCP_SCHEDULER_OPENAI_API_KEY"] = args.openai_api_key

    if args.config:
        os.environ["MCP_SCHEDULER_CONFIG_FILE"] = args.config
    
    if args.address:
        os.environ["MCP_SCHEDULER_ADDRESS"] = args.address
    
    if args.port:
        os.environ["MCP_SCHEDULER_PORT"] = str(args.port)
    
    if args.transport:
        os.environ["MCP_SCHEDULER_TRANSPORT"] = args.transport
    
    if args.log_level:
        os.environ["MCP_SCHEDULER_LOG_LEVEL"] = args.log_level
    
    if args.log_file:
        os.environ["MCP_SCHEDULER_LOG_FILE"] = args.log_file
    
    if args.db_path:
        os.environ["MCP_SCHEDULER_DB_PATH"] = args.db_path
    
    if args.ai_model:
        os.environ["MCP_SCHEDULER_AI_MODEL"] = args.ai_model
    
    # Enable debug mode
    debug_mode = args.debug
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_sigterm)
    
    try:
        # Apply stdout wrapper for stdio transport to ensure only JSON goes to stdout
        if args.transport == "stdio" or (not args.transport and os.getenv("MCP_SCHEDULER_TRANSPORT", "stdio") == "stdio"):
            log_to_stderr("Using stdio transport - applying stdout wrapper")
            sys.stdout = JSONRPCStdout(original_stdout)
        
        # Install JSON fixing patches if requested or always in debug mode
        if args.fix_json or debug_mode:
            # Try to patch the FastMCP parser directly
            if not patch_fastmcp_parser():
                # If direct patching fails, use stdin wrapper as fallback
                if not install_stdio_wrapper():
                    # Last resort: replace stdin with our custom wrapper
                    log_to_stderr("Installing basic JSON fix wrapper")
                    sys.stdin = SafeJsonStdin(sys.stdin)
                    
        # Load configuration
        config = Config()
        
        # Show version and exit if requested
        if args.version:
            log_to_stderr(f"{config.server_name} version {config.server_version}")
            sys.exit(0)
        
        # Configure logging - ensure it goes to a file or stderr, not stdout
        if not config.log_file and config.transport == "stdio":
            # Force a log file when using stdio transport if none was specified
            config.log_file = "mcp_scheduler.log"
            
        setup_logging(config.log_level, config.log_file)
        
        # Print configuration for debugging
        log_to_stderr(f"Configuration: AI Provider={config.ai_provider}, Model={config.ai_model}, Transport={config.transport}")
        
        # Initialize components
        database = Database(config.db_path)
        executor = Executor(config)
        
        global scheduler
        scheduler = Scheduler(database, executor)
        
        global server
        server = SchedulerServer(scheduler, config)
        
        # Start the scheduler in a separate thread
        scheduler_thread = threading.Thread(
            target=run_scheduler_in_thread,
            args=(scheduler,),
            daemon=True  # Make thread a daemon so it exits when the main thread exits
        )
        scheduler_thread.start()
        log_to_stderr("Scheduler started in background thread")

        # If transport is SSE, start well-known server in separate thread
        if config.transport == "sse":
            log_to_stderr(f"Starting well-known server on port {config.server_port + 1}")
            threading.Thread(target=start_well_known_server, daemon=True).start()

        # Start the MCP server (this will block with stdio transport)
        log_to_stderr(f"Starting MCP server with {config.transport} transport")
        server.start()
        
    except KeyboardInterrupt:
        log_to_stderr("Interrupted by user")
        sys.exit(0)
    except json.JSONDecodeError as e:
        log_to_stderr(f"JSON parsing error: {e}")
        if debug_mode:
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        log_to_stderr(f"Error during initialization: {e}")
        if debug_mode:
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

if __name__ == "__main__":
    main()