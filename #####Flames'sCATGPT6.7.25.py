"""
CatGPT 0.1 [CatGPT] 20XX - OpenRouter Edition (Updated for AutoGPT-like Behavior)
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog
import requests
import shutil
import queue
from datetime import datetime
from pathlib import Path
from threading import Thread, Lock, Event
from typing import Any, Dict, List, Tuple, Optional, Callable
from html.parser import HTMLParser

try:
    import aiohttp
    ASYNC_MODE = True
except ImportError:
    ASYNC_MODE = False

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Runtime Globals
RUNTIME_API_KEY: Optional[str] = None
API_KEY_LOCK = Lock()

# Runtime Paths
HOME = Path.home()
ARCHIVE_DIR = HOME / "Documents" / "CatGPT_Agent_Archive"
AGENT_WORKSPACE = ARCHIVE_DIR / "autonomous_workspace"

for path in [ARCHIVE_DIR, AGENT_WORKSPACE]:
    path.mkdir(parents=True, exist_ok=True)

# Constants
DEFAULT_MODEL = "meta-llama/llama-4-maverick:free"
LLM_TIMEOUT = 120
CODE_TIMEOUT = 60
HTTP_REFERER = "http://localhost:8000"

# CatGPT UI Theme
UI_THEME = {
    "bg_primary": "#1a1a2e",
    "bg_secondary": "#16213e",
    "bg_tertiary": "#0f1120",
    "bg_chat_display": "#0f1120",
    "bg_chat_input": "#2a2a4e",
    "bg_button_primary": "#6a0dad",
    "bg_button_success": "#28a745",
    "bg_button_danger": "#dc3545",
    "bg_button_info": "#007bff",
    "bg_listbox_select": "#9b59b6",
    "fg_primary": "#e0e0ff",
    "fg_secondary": "#b0b0cc",
    "fg_button_light": "#ffffff",
    "fg_header": "#6a0dad",
    "font_default": ("Consolas", 11),
    "font_chat": ("Consolas", 11),
    "font_button_main": ("Inter", 11, "bold"),
    "font_title": ("Inter", 14, "bold"),
    "font_listbox": ("Consolas", 10),
    "font_mission": ("Consolas", 10)
}

# Utility Helpers
def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_api_key() -> str:
    with API_KEY_LOCK:
        if RUNTIME_API_KEY:
            return RUNTIME_API_KEY
    return ""

# HTML Parser for Web Scraping
class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []

    def handle_data(self, data):
        self.text_parts.append(data)

    def get_text(self):
        return ' '.join(self.text_parts).strip()

# API Client for OpenRouter
class APIClient:
    API_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key_getter: Callable[[], str], timeout: float):
        self._api_key_getter = api_key_getter
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.session_lock = asyncio.Lock()

    def _get_headers(self) -> Dict[str, str]:
        api_key = self._api_key_getter()
        if not api_key:
            raise RuntimeError("API Key is missing.")
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": HTTP_REFERER,
            "X-Title": "CatGPT Agent",
        }

    async def _get_async_session(self) -> aiohttp.ClientSession:
        async with self.session_lock:
            if self.session is None or self.session.closed:
                self.session = aiohttp.ClientSession()
            return self.session

    async def call_async(self, payload: Dict[str, Any]) -> str:
        if not ASYNC_MODE:
            return self.call_sync(payload)

        session = await self._get_async_session()
        try:
            async with session.post(self.API_BASE_URL, headers=self._get_headers(), json=payload, timeout=self.timeout) as resp:
                response_json = await resp.json()
                if resp.status != 200:
                    error_text = json.dumps(response_json)
                    logger.error(f"LLM API call failed: {resp.status} - {error_text}")
                    raise RuntimeError(f"API Error ({resp.status}): {error_text}")
                return json.dumps(response_json)
        except aiohttp.ClientError as e:
            logger.error(f"Network error during async API call: {e}")
            raise RuntimeError(f"Network Error: {e}")

    def call_sync(self, payload: Dict[str, Any]) -> str:
        try:
            response = requests.post(self.API_BASE_URL, headers=self._get_headers(), json=payload, timeout=self.timeout)
            response.raise_for_status()
            return json.dumps(response.json())
        except requests.RequestException as e:
            logger.error(f"Network error during sync API call: {e}")
            raise RuntimeError(f"Network Error: {e}")

    async def close_session(self):
        async with self.session_lock:
            if self.session and not self.session.closed:
                await self.session.close()
                self.session = None

# Code Interpreter
class CodeInterpreter:
    def __init__(self, timeout: int = CODE_TIMEOUT, workspace_dir: Path = AGENT_WORKSPACE):
        self.timeout = timeout
        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Code interpreter workspace: {self.workspace_dir}")

    def execute_code(self, code_string: str) -> Tuple[str, str, Optional[str], Optional[str]]:
        stdout_str, stderr_str, error_msg, saved_png_path = "", "", None, None
        with tempfile.TemporaryDirectory(dir=str(self.workspace_dir)) as temp_dir:
            temp_script_path = Path(temp_dir) / "script.py"
            try:
                temp_script_path.write_text(code_string, encoding="utf-8")
                process = subprocess.run(
                    [sys.executable, "-u", str(temp_script_path)],
                    capture_output=True, text=True, timeout=self.timeout,
                    cwd=temp_dir, check=False
                )
                stdout_str, stderr_str = process.stdout, process.stderr
                png_files = [f for f in os.listdir(temp_dir) if f.endswith('.png')]
                if png_files:
                    png_path = Path(temp_dir) / png_files[0]
                    saved_png_path = self.workspace_dir / f"output_{now_ts()}.png"
                    shutil.copy(png_path, saved_png_path)
                    saved_png_path = str(saved_png_path)
            except subprocess.TimeoutExpired:
                error_msg = f"Code execution timed out after {self.timeout} seconds."
                stderr_str += f"\nTimeoutError: Execution exceeded {self.timeout} seconds."
            except Exception as e:
                error_msg = f"An unexpected error occurred: {e}"
                logger.error(f"Code execution error: {e}")
            finally:
                if temp_script_path.exists():
                    temp_script_path.unlink()
        return stdout_str, stderr_str, error_msg, saved_png_path

# Autonomous Agent
class AutonomousAgent:
    MASTER_TOOL_LIBRARY = {
        "execute_python_code": {
            "description": "Executes Python code in a sandboxed environment. Returns stdout, stderr, and system errors.",
            "args": {"code_string": "The Python code to execute."}
        },
        "write_file": {
            "description": "Writes content to a file in the agent's workspace.",
            "args": {"filename": "File name.", "content": "Content to write."}
        },
        "read_file": {
            "description": "Reads content from a file in the agent's workspace.",
            "args": {"filename": "File to read."}
        },
        "list_files": {
            "description": "Lists all files in the agent's workspace.",
            "args": {}
        },
        "search_web": {
            "description": "Fetches text content from a URL.",
            "args": {"url": "URL to retrieve."}
        },
        "task_complete": {
            "description": "Indicates the agent believes the goal is achieved, but continues running until stopped.",
            "args": {"reason": "Reason for task completion."}
        }
    }

    def __init__(self, goal: str, api_client: APIClient, code_interpreter: CodeInterpreter,
                 model_name: str, ui_queue: queue.Queue, stop_event: Event,
                 system_prompt: str, selected_tool_names: List[str]):
        self.goal = goal
        self.api_client = api_client
        self.code_interpreter = code_interpreter
        self.model_name = model_name
        self.ui_queue = ui_queue
        self.stop_event = stop_event
        self.system_prompt = system_prompt
        self.history: List[Dict[str, Any]] = []
        self.completed = False

        tool_function_map = {
            "execute_python_code": self.code_interpreter.execute_code,
            "write_file": self.write_file,
            "read_file": self.read_file,
            "list_files": self.list_files,
            "search_web": self.search_web,
            "task_complete": self.task_complete
        }

        self.tools = {name: tool_function_map[name] for name in selected_tool_names if name in tool_function_map}
        
        self.open_ai_tool_config = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": self.MASTER_TOOL_LIBRARY[name]["description"],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            arg_name: {"type": "string", "description": arg_desc}
                            for arg_name, arg_desc in self.MASTER_TOOL_LIBRARY[name]["args"].items()
                        },
                        "required": list(self.MASTER_TOOL_LIBRARY[name]["args"].keys()),
                    },
                }
            }
            for name in selected_tool_names if name in self.MASTER_TOOL_LIBRARY
        ]

    def log_to_ui(self, message: str, tag: str = "info"):
        if not self.stop_event.is_set():
            self.ui_queue.put({"tag": tag, "content": message})

    def write_file(self, filename: str, content: str) -> str:
        try:
            (AGENT_WORKSPACE / filename).write_text(content, encoding='utf-8')
            return f"Successfully wrote to '{filename}'."
        except Exception as e:
            return f"Error writing to file: {e}"

    def read_file(self, filename: str) -> str:
        try:
            return (AGENT_WORKSPACE / filename).read_text(encoding='utf-8')
        except FileNotFoundError:
            return f"Error: File '{filename}' not found."
        except Exception as e:
            return f"Error reading file: {e}"

    def list_files(self) -> str:
        try:
            files = [str(p.relative_to(AGENT_WORKSPACE)) for p in AGENT_WORKSPACE.rglob("*")]
            return "Workspace files:\n" + "\n".join(files) if files else "Workspace is empty."
        except Exception as e:
            return f"Error listing files: {e}"

    def search_web(self, url: str) -> str:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            parser = TextExtractor()
            parser.feed(response.text)
            text = parser.get_text()
            return text[:4000]
        except requests.RequestException as e:
            return f"Error searching web: {e}"

    def task_complete(self, reason: str) -> str:
        self.log_to_ui(f"TASK BELIEVED COMPLETE: {reason}. Continuing to run until stopped.", "system")
        return f"Agent believes task is complete: {reason}. Awaiting user stop command."

    async def run(self):
        self.log_to_ui(f"AUTONOMOUS AGENT ACTIVATED\nGOAL: {self.goal}\nMODEL: {self.model_name}", "system")
        
        self.history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"My goal is: {self.goal}. Please begin."}
        ]
        
        self.log_to_ui(f"SYSTEM PROMPT INITIALIZED:\n{self.system_prompt}", "llm")

        iteration_count = 0
        while not self.stop_event.is_set():
            iteration_count += 1
            self.log_to_ui(f"--- Iteration {iteration_count} ---", "agent")

            try:
                self.log_to_ui("Thinking...", "agent")
                
                payload = {
                    "model": self.model_name,
                    "messages": self.history,
                    "tools": self.open_ai_tool_config,
                    "tool_choice": "auto"
                }
                
                llm_response_raw = await self.api_client.call_async(payload)
                response_data = json.loads(llm_response_raw)
                
                message = response_data['choices'][0]['message']
                self.history.append(message)

                tool_calls = message.get('tool_calls')
                
                if tool_calls:
                    self.log_to_ui(f"LLM Response (Tool Call):\n{json.dumps(tool_calls, indent=2)}", "llm")
                    
                    tool_results_for_history = []
                    for tool_call in tool_calls:
                        command_name = tool_call['function']['name']
                        try:
                            command_args = json.loads(tool_call['function']['arguments'])
                        except json.JSONDecodeError:
                            self.log_to_ui(f"ERROR: Failed to decode arguments for {command_name}: {tool_call['function']['arguments']}", "error")
                            continue

                        self.log_to_ui(f"COMMAND: {command_name}({json.dumps(command_args)})", "agent")

                        if command_name in self.tools:
                            tool_func = self.tools[command_name]
                            try:
                                if command_name == 'execute_python_code':
                                    stdout, stderr, exec_err, png_path = tool_func(**command_args)
                                    tool_result = f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                                    if exec_err: tool_result += f"\nEXECUTION_ERROR: {exec_err}"
                                    if png_path: tool_result += f"\nPNG generated: {png_path}"
                                else:
                                    tool_result = tool_func(**command_args)
                            except Exception as e:
                                tool_result = f"Error executing tool {command_name}: {e}"
                                logger.error(tool_result, exc_info=True)
                        else:
                            tool_result = f"Error: Unknown command '{command_name}'."
                        
                        self.log_to_ui(f"RESULT:\n{tool_result}", "result")
                        
                        tool_results_for_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call['id'],
                            "content": tool_result,
                        })
                    self.history.extend(tool_results_for_history)
                
                else:
                    text_content = message.get('content', '')
                    self.log_to_ui(f"LLM Response (Text):\n{text_content}", "llm")

                self.log_to_ui("Preparing next action...", "agent")

            except json.JSONDecodeError as e:
                self.log_to_ui(f"ERROR: JSON decode failed: {e}. Raw response: {llm_response_raw}", "error")
            except Exception as e:
                self.log_to_ui(f"CRITICAL ERROR: {e}", "error")
                logger.error("Critical error in agent loop", exc_info=True)
                self.stop_event.set()

        self.log_to_ui("Agent has shut down.", "system")

# Tkinter UI
class AgentUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CatGPT 0.1 [CatGPT] 20XX")
        self.geometry("800x600")
        self.configure(bg=UI_THEME["bg_primary"])
        self.shutdown_event = Event()
        
        self.async_loop = None
        self.async_thread = None
        self.agent_task_future = None
        self._setup_async_loop()
        
        self._ask_for_api_key()
        if not get_api_key():
            if self.async_loop:
                self.async_loop.call_soon_threadsafe(self.async_loop.stop)
                self.async_thread.join(timeout=2)
            self.destroy()
            sys.exit(1)

        self.api_client = APIClient(get_api_key, LLM_TIMEOUT)
        self.code_interpreter = CodeInterpreter()
        self.ui_queue = queue.Queue()
        
        self.setup_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        logger.info("CatGPT UI initialized.")

    def _setup_async_loop(self):
        if not ASYNC_MODE:
            return
        self.async_loop = asyncio.new_event_loop()
        self.async_thread = Thread(target=self.async_loop.run_forever, daemon=True)
        self.async_thread.start()
        logger.info("Asyncio event loop thread started.")

    def _ask_for_api_key(self):
        global RUNTIME_API_KEY
        temp_root = tk.Tk()
        temp_root.withdraw()

        with API_KEY_LOCK:
            key = simpledialog.askstring("API Key Required", "Enter your OpenRouter API Key:", show='*', parent=temp_root)
            if key:
                RUNTIME_API_KEY = key.strip()
                logger.info("OpenRouter API Key set.")
            else:
                logger.warning("No API key provided. Exiting.")
                messagebox.showwarning("API Key Missing", "No OpenRouter API Key provided. Exiting.", parent=temp_root)
        
        temp_root.destroy()

    def setup_ui(self):
        main_frame = ttk.Frame(self, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure("TFrame", background=UI_THEME["bg_primary"])
        style.configure("TLabel", background=UI_THEME["bg_primary"], foreground=UI_THEME["fg_primary"], font=UI_THEME["font_default"])
        style.configure("TEntry", fieldbackground=UI_THEME["bg_chat_input"], foreground=UI_THEME["fg_primary"], insertbackground=UI_THEME["fg_primary"])
        
        goal_frame = ttk.Frame(main_frame)
        goal_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(goal_frame, text="Goal:", font=UI_THEME["font_title"]).pack(side=tk.LEFT, padx=(0, 10))
        self.goal_entry = ttk.Entry(goal_frame, font=UI_THEME["font_default"], width=80)
        self.goal_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.goal_entry.insert(0, "Create a simple calculator in Python and save it to a file named calculator.py")

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        self.start_button = tk.Button(
            button_frame, text="Start Agent", command=self.start_agent,
            bg=UI_THEME["bg_button_primary"], fg=UI_THEME["fg_button_light"], relief=tk.FLAT,
            font=UI_THEME["font_button_main"], activebackground=UI_THEME["bg_button_success"], borderwidth=0
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 5), ipady=5, ipadx=10)

        self.stop_button = tk.Button(
            button_frame, text="Stop Agent", command=self.stop_agent,
            bg=UI_THEME["bg_button_danger"], fg=UI_THEME["fg_button_light"], relief=tk.FLAT,
            font=UI_THEME["font_button_main"], state=tk.DISABLED, activebackground="#ff6b6b", borderwidth=0
        )
        self.stop_button.pack(side=tk.LEFT, ipady=5, ipadx=10)

        self.output_text = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, font=UI_THEME["font_chat"],
            bg=UI_THEME["bg_chat_display"], fg=UI_THEME["fg_primary"],
            insertbackground=UI_THEME["fg_primary"], selectbackground=UI_THEME["bg_listbox_select"],
            borderwidth=0, highlightthickness=1, highlightbackground=UI_THEME["fg_header"]
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)

        self.output_text.tag_configure("system", foreground="#2ecc71")
        self.output_text.tag_configure("agent", foreground="#3498db")
        self.output_text.tag_configure("tool", foreground="#f39c12")
        self.output_text.tag_configure("result", foreground="#9b59b6")
        self.output_text.tag_configure("llm", foreground="#95a5a6")
        self.output_text.tag_configure("error", foreground="#e74c3c")

        self.update_ui()

    def start_agent(self):
        goal = self.goal_entry.get().strip()
        if not goal:
            messagebox.showwarning("No Goal", "Please enter a goal for the agent.")
            return

        if not get_api_key():
            messagebox.showerror("API Key Missing", "No OpenRouter API Key provided. Please restart the application.")
            return

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.shutdown_event.clear()

        system_prompt = "You are CatGPT 0.1, an autonomous AI agent. Your goal is to solve problems by thinking, planning, and executing commands. You must use the provided tools to achieve the user's goal. Before each action, explain your reasoning and plan. The agent will continue running until manually stopped by the user, even after calling `task_complete`."
        selected_tools = list(AutonomousAgent.MASTER_TOOL_LIBRARY.keys())

        agent = AutonomousAgent(
            goal, self.api_client, self.code_interpreter,
            DEFAULT_MODEL, self.ui_queue, self.shutdown_event,
            system_prompt, selected_tools
        )

        self.agent_task_future = asyncio.run_coroutine_threadsafe(agent.run(), self.async_loop)

    def stop_agent(self):
        self.shutdown_event.set()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.log_message("Agent stop requested by user.", "system")

    def update_ui(self):
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                self.log_message(msg["content"], msg.get("tag", "info"))
        except queue.Empty:
            pass
        
        if self.agent_task_future and self.agent_task_future.done() and self.stop_button['state'] == tk.NORMAL:
            try:
                self.agent_task_future.result()
            except Exception as e:
                self.log_message(f"Agent task finished with an error: {e}", "error")
                logger.error("Agent task raised an exception", exc_info=True)
            
            self.log_message("Agent has completed its task or stopped.", "system")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.agent_task_future = None

        self.after(100, self.update_ui)

    def log_message(self, message: str, tag: str = "info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.output_text.insert(tk.END, f"[{timestamp}] {message}\n\n", tag)
        self.output_text.see(tk.END)

    def _on_closing(self):
        global RUNTIME_API_KEY
        logger.info("Initiating CatGPT shutdown")
        self.shutdown_event.set()
        
        if ASYNC_MODE and self.async_loop and self.async_loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self.api_client.close_session(), self.async_loop)
                future.result(timeout=5)
                logger.info("AIOHTTP session closed.")
            except Exception as e:
                logger.error(f"Error closing aiohttp session: {e}")
            finally:
                self.async_loop.call_soon_threadsafe(self.async_loop.stop)
                if self.async_thread:
                    self.async_thread.join(timeout=2)
                    logger.info("Asyncio event loop thread finished.")

        with API_KEY_LOCK:
            RUNTIME_API_KEY = None
            logger.info("API key cleared.")
        
        self.destroy()

if __name__ == "__main__":
    if not ASYNC_MODE:
        logger.warning("`aiohttp` is not installed. The agent will run in synchronous mode, which may be slower.")
    app = AgentUI()
    app.mainloop()
