import asyncio
import logging
import shlex
import subprocess
import platform
import os
from datetime import datetime
import aiohttp
from typing import Optional, Tuple, Dict, Any

from openai import OpenAI

from .task import Task, TaskExecution, TaskStatus, TaskType

logger = logging.getLogger(__name__)

class Executor:
    """Task executor for running scheduled tasks."""
    
    def __init__(self, config):
        """Initialize the task executor with configuration."""
        self.config = config
        self.execution_timeout = config.execution_timeout
        self.is_windows = platform.system() == "Windows"
        self.ai_client = None
        
        self._init_ai_client()
    
    def _init_ai_client(self):
        """Initialize AI client based on configuration."""
        ai_config = self.config.get_ai_config()

        # 调试信息
        logger.info(f"AI Provider: {self.config.ai_provider}")
        logger.info(f"AI Model: {self.config.ai_model}")
        logger.info(f"Base URL: {ai_config.get('base_url')}")
        logger.info(f"API Key present: {bool(ai_config.get('api_key'))}")
        logger.info(f"Full AI config: {ai_config}")
        
        if not ai_config.get("api_key"):
            logger.warning("No API key configured for AI tasks")
            return
        
        try:
            if self.config.ai_provider in ["openai", "local"]:
                self.ai_client = OpenAI(
                    api_key=ai_config["api_key"],
                    base_url=ai_config.get("base_url"),
                    organization=ai_config.get("organization")
                )
                logger.info(f"Initialized {self.config.ai_provider} client with model: {self.config.ai_model}")
                
            elif self.config.ai_provider == "azure":
                from openai import AzureOpenAI
                self.ai_client = AzureOpenAI(
                    api_key=ai_config["api_key"],
                    api_version=ai_config["api_version"],
                    azure_endpoint=ai_config["base_url"].replace(f"/openai/deployments/{self.config.ai_model}", "")
                )
                logger.info(f"Initialized Azure OpenAI client with model: {self.config.ai_model}")
                
            elif self.config.ai_provider == "anthropic":
                self.ai_client = OpenAI(
                    api_key=ai_config["api_key"],
                    base_url=ai_config["base_url"]
                )
                logger.info(f"Initialized Anthropic client with model: {self.config.ai_model}")
            
            # 测试API连接
            if self.ai_client:
                self._test_ai_connection()
                
        except Exception as e:
            logger.error(f"Failed to initialize AI client: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _test_ai_connection(self):
        """Test AI API connection."""
        try:
            logger.info("Testing AI API connection...")
            # 简单的测试请求
            test_response = self.ai_client.chat.completions.create(
                model=self.config.ai_model,
                messages=[{"role": "user", "content": "Say 'test successful'"}],
                max_tokens=10
            )
            logger.info("AI API connection test successful")
            return True
        except Exception as e:
            logger.error(f"AI API connection test failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        
    async def execute_task(self, task: Task) -> TaskExecution:
        """Execute a task based on its type."""
        logger.info(f"Executing task: {task.id} ({task.name})")
        
        execution = TaskExecution(task_id=task.id)
        
        try:
            if task.type.value == "shell_command":
                output, error = await self._execute_shell_command(task.command)
                if error:
                    execution.status = TaskStatus.FAILED
                    execution.error = error
                else:
                    execution.status = TaskStatus.COMPLETED
                    execution.output = output
                    
            elif task.type.value == "api_call":
                output, error = await self._execute_api_call(
                    task.api_url, 
                    task.api_method, 
                    task.api_headers, 
                    task.api_body
                )
                if error:
                    execution.status = TaskStatus.FAILED
                    execution.error = error
                else:
                    execution.status = TaskStatus.COMPLETED
                    execution.output = output
                    
            elif task.type.value == "ai":
                output, error = await self._execute_ai_task(task.prompt)
                if error:
                    execution.status = TaskStatus.FAILED
                    execution.error = error
                else:
                    execution.status = TaskStatus.COMPLETED
                    execution.output = output
                    
            elif task.type.value == "reminder":
                output, error = await self._execute_reminder_task(
                    task.reminder_title or task.name,
                    task.reminder_message
                )
                if error:
                    execution.status = TaskStatus.FAILED
                    execution.error = error
                else:
                    execution.status = TaskStatus.COMPLETED
                    execution.output = output
            
            else:
                execution.status = TaskStatus.FAILED
                execution.error = f"Unsupported task type: {task.type.value}"
                
        except Exception as e:
            logger.exception(f"Error executing task {task.id}")
            execution.status = TaskStatus.FAILED
            execution.error = str(e)
        
        execution.end_time = datetime.utcnow()
        return execution
    
    async def _execute_shell_command(self, command: str) -> Tuple[Optional[str], Optional[str]]:
        """Execute a shell command with timeout."""
        if not command:
            return None, "No command specified"
        
        # Determine if we need to use shell mode
        use_shell = self.is_windows
        
        # These commands are shell builtins and need shell=True
        shell_commands = ['start', 'cd', 'dir', 'echo', 'set', 'type', 'copy', 'del', 'md', 'rd', 'ren', 'cls']
        
        # If command starts with any of these, use shell mode
        if any(command.strip().lower().startswith(cmd) for cmd in shell_commands):
            use_shell = True
        
        # If pipe or redirect is in command, use shell mode
        if '|' in command or '>' in command or '<' in command:
            use_shell = True
            
        logger.info(f"Executing command: {command} (shell mode: {use_shell})")
        
        try:
            if use_shell:
                # Use shell mode for Windows or shell-specific commands
                if self.is_windows:
                    # Force cmd.exe on Windows
                    full_command = f"cmd.exe /c {command}"
                else:
                    full_command = command
                
                process = await asyncio.create_subprocess_shell(
                    full_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    shell=True
                )
            else:
                # Use direct execution for standard commands
                try:
                    args = shlex.split(command)
                except ValueError as e:
                    return None, f"Invalid command syntax: {str(e)}"
                
                process = await asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=self.execution_timeout
                )
                
                if process.returncode != 0:
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    return None, f"Command failed with exit code {process.returncode}: {error_msg}"
                
                return stdout.decode().strip(), None
                
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except Exception:
                    pass
                return None, f"Command timed out after {self.execution_timeout} seconds"
                
        except Exception as e:
            return None, f"Failed to execute command: {str(e)}"
    
    async def _execute_api_call(self, url: str, method: str, headers: dict, body: dict) -> Tuple[Optional[str], Optional[str]]:
        """Execute an API call."""
        if not url:
            return None, "No URL specified"
        
        if not method:
            method = "GET"
            
        method = method.upper()
        
        try:
            async with aiohttp.ClientSession() as session:
                request_kwargs = {
                    "headers": headers or {},
                }
                
                if method in ["POST", "PUT", "PATCH"] and body:
                    request_kwargs["json"] = body
                
                async with session.request(
                    method, 
                    url, 
                    **request_kwargs,
                    timeout=aiohttp.ClientTimeout(total=self.execution_timeout)
                ) as response:
                    response_text = await response.text()
                    
                    if response.status >= 400:
                        return None, f"API call failed with status {response.status}: {response_text}"
                    
                    return response_text, None
                    
        except aiohttp.ClientError as e:
            return None, f"API call failed: {str(e)}"
        except asyncio.TimeoutError:
            return None, f"API call timed out after {self.execution_timeout} seconds"
    
    async def _execute_ai_task(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """Execute an AI task using configured provider."""
        if not prompt:
            return None, "No prompt specified"
        
        # 详细的调试信息
        logger.info(f"=== AI Task Execution Debug ===")
        logger.info(f"AI Client: {self.ai_client}")
        logger.info(f"AI Client type: {type(self.ai_client)}")
        logger.info(f"Prompt: {prompt}")
        
        if not self.ai_client:
            logger.error("No AI client configured for AI tasks")
            return None, "No AI client configured for AI tasks"
        
        try:
            ai_config = self.config.get_ai_config()
            
            logger.info(f"Using AI provider: {self.config.ai_provider}, model: {self.config.ai_model}")
            
            # 准备请求参数
            request_params = {
                "model": self.config.ai_model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant executing scheduled tasks."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": ai_config.get("max_tokens", 2000),
                "temperature": ai_config.get("temperature", 0.7)
            }
            
            logger.info(f"Request parameters: {request_params}")
            
            # 测试API连接
            logger.info("Making API call to DeepSeek...")
            
            # 添加超时控制
            try:
                completion = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.ai_client.chat.completions.create,
                        **request_params
                    ),
                    timeout=30.0  # 30秒超时
                )
                
                response = completion.choices[0].message.content
                logger.info(f"AI response received: {response}")
                return response, None
                
            except asyncio.TimeoutError:
                logger.error("AI API call timed out after 30 seconds")
                return None, "AI API call timed out"
            
        except Exception as e:
            logger.error(f"AI task failed: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, f"AI task failed: {str(e)}"

    async def _execute_reminder_task(self, title: str, message: str) -> Tuple[Optional[str], Optional[str]]:
        """Execute a reminder task that displays a desktop notification with sound."""
        if not message:
            return None, "No message specified for reminder"
        
        os_type = platform.system()
        command = None
        
        try:
            # Generate platform-specific notification commands
            if os_type == "Windows":
                # Escape single quotes for VBScript
                safe_title = title.replace("'", "''")
                safe_message = message.replace("'", "''")
                
                # Create a temporary HTML file for the notification
                temp_dir = os.path.join(os.environ.get('TEMP', ''), '')
                temp_html = os.path.join(temp_dir, 'notification.hta')
                
                # Use VBScript instead of JavaScript for playing sounds
                with open(temp_html, 'w') as f:
                    f.write(f'''
                    <html>
                    <head>
                    <title>{safe_title}</title>
                    <hta:application
                        id="notification"
                        applicationname="Notification"
                        border="thin"
                        borderstyle="normal"
                        caption="yes"
                        contextmenu="no"
                        icon=""
                        maximizebutton="no"
                        minimizebutton="yes"
                        navigable="no"
                        showintaskbar="yes"
                        singleinstance="yes"
                        sysmenu="yes"
                        version="1.0"
                        windowstate="normal"
                    />
                    <style>
                        body {{
                            font-family: Arial, sans-serif;
                            padding: 20px;
                            background-color: #f0f0f0;
                            width: 300px;
                            height: 150px;
                        }}
                        h2 {{
                            color: #333;
                            margin-top: 0;
                        }}
                        p {{
                            color: #555;
                        }}
                        button {{
                            padding: 5px 15px;
                            margin-top: 15px;
                            border: none;
                            background-color: #0078d7;
                            color: white;
                            cursor: pointer;
                        }}
                    </style>
                    <script language="VBScript">
                        Sub Window_OnLoad
                            ' Play the notification sound
                            Set oShell = CreateObject("WScript.Shell")
                            oShell.Run "rundll32 user32.dll,MessageBeep", 0, False
                        End Sub
                        
                        Sub CloseButton_OnClick
                            window.close
                        End Sub
                        
                        Sub Window_OnLoad
                            ' Auto-close after 30 seconds
                            setTimeout 30000, "window.close"
                        End Sub
                        
                        Sub setTimeout(ms, expr)
                            CreateObject("WScript.Shell").Run "ping -n " & Int(ms/1000 + 2) & " 127.0.0.1 > nul && mshta vbscript:close(Execute(""window.close""))", 0, False
                        End Sub
                    </script>
                    </head>
                    <body>
                        <h2>{safe_title}</h2>
                        <p>{safe_message}</p>
                        <button onclick="CloseButton_OnClick">OK</button>
                    </body>
                    </html>
                    ''')
                
                # Use mshta to display the notification
                command = f'start mshta.exe "{temp_html}"'
                
                # Also run a direct MessageBeep as backup
                backup_command = f'rundll32 user32.dll,MessageBeep'
                await self._execute_shell_command(backup_command)
                
            elif os_type == "Darwin":  # macOS
                # Escape double quotes in the osascript command
                safe_title = title.replace('"', '\\"')
                safe_message = message.replace('"', '\\"')
                # Use the "default" sound which is guaranteed to work
                command = f'osascript -e \'display notification "{safe_message}" with title "{safe_title}" sound name "default"\''
                
            else:  # Linux and others
                # Escape quotes in the notify-send command
                safe_title = title.replace('"', '\\"')
                safe_message = message.replace('"', '\\"')
                
                # Try paplay with notify-send for sound on Linux
                notify_cmd = f'notify-send -u normal "{safe_title}" "{safe_message}"'
                sound_cmd = 'paplay /usr/share/sounds/freedesktop/stereo/message.oga'
                
                # Chain commands together
                command = f'{notify_cmd} && {sound_cmd}'
                
                # Check if notify-send exists
                notify_send_check = await asyncio.create_subprocess_shell(
                    "which notify-send",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await notify_send_check.communicate()
                
                if notify_send_check.returncode != 0:
                    # Fallback to zenity with sound
                    zenity_check = await asyncio.create_subprocess_shell(
                        "which zenity",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    await zenity_check.communicate()
                    
                    if zenity_check.returncode == 0:
                        command = f'zenity --info --title="{safe_title}" --text="{safe_message}" & {sound_cmd}'
                    else:
                        # Last resort: try xmessage with a sound command
                        xmessage_check = await asyncio.create_subprocess_shell(
                            "which xmessage",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        await xmessage_check.communicate()
                        
                        if xmessage_check.returncode == 0:
                            command = f'xmessage -center "{safe_title}: {safe_message}" & {sound_cmd}'
                        else:
                            return None, f"No notification command available on this system ({os_type})"
            
            if command:
                # Execute the notification command
                output, error = await self._execute_shell_command(command)
                if error:
                    return None, f"Notification failed: {error}"
                
                return f"Displayed notification: {title}", None
            else:
                return None, "Failed to create notification command"
                
        except Exception as e:
            logger.exception("Error in reminder task")
            return None, f"Reminder task failed: {str(e)}"