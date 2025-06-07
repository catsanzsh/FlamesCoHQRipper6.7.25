# test.py
"""
CatGPT 0.1 [DGM-Agent] 20XX - OpenRouter Edition (Updated for AutoGPT-like Behavior & Self-Modification)
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

# --- Darwin Godel Machine Research Abstract ---
DGM_PAPER_ABSTRACT = """
Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents
Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, Jeff Clune

Today's AI systems have human-designed, fixed architectures and cannot autonomously and continuously improve themselves. The advance of AI could itself be automated. If done safely, that would accelerate AI development and allow us to reap its benefits much sooner. Meta-learning can automate the discovery of novel algorithms, but is limited by first-order improvements and the human design of a suitable search space. The Gödel machine proposed a theoretical alternative: a self-improving AI that repeatedly modifies itself in a provably beneficial manner. Unfortunately, proving that most changes are net beneficial is impossible in practice. We introduce the Darwin Gödel Machine (DGM), a self-improving system that iteratively modifies its own code (thereby also improving its ability to modify its own codebase) and empirically validates each change using coding benchmarks. Inspired by Darwinian evolution and open-endedness research, the DGM maintains an archive of generated coding agents. It grows the archive by sampling an agent from it and using a foundation model to create a new, interesting, version of the sampled agent. This open-ended exploration forms a growing tree of diverse, high-quality agents and allows the parallel exploration of many different paths through the search space. Empirically, the DGM automatically improves its coding capabilities (e.g., better code editing tools, long-context window management, peer-review mechanisms), increasing performance on SWE-bench from 20.0% to 50.0%, and on Polyglot from 14.2% to 30.7%. Furthermore, the DGM significantly outperforms baselines without self-improvement or open-ended exploration. All experiments were done with safety precautions (e.g., sandboxing, human oversight). The DGM is a significant step toward self-improving AI, capable of gathering its own stepping stones along paths that unfold into endless innovation.
"""

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Runtime Globals
RUNTIME_API_KEY: Optional[str] = None
API_KEY_LOCK = Lock()

# Runtime Paths
HOME = Path.home()
ARCHIVE_DIR = HOME / "Documents" / "DGM_Agent_Archive"
AGENT_WORKSPACE = ARCHIVE_DIR / "autonomous_workspace"

for path in [ARCHIVE_DIR, AGENT_WORKSPACE]:
    path.mkdir(parents=True, exist_ok=True)

# Constants
DEFAULT_MODEL = "meta-llama/llama-3-8b-instruct:free"
LLM_TIMEOUT = 120
CODE_TIMEOUT = 60
HTTP_REFERER = "https://github.com/reworkd/AgentGPT"
AGENT_FILENAME = "test.py"

# Modern UI Theme
MODERN_UI_THEME = {
    "bg_primary": "#202123",
    "bg_secondary": "#343541",
    "bg_tertiary": "#444654",
    "bg_chat_display": "#343541",
    "bg_chat_input": "#40414f",
    "bg_button_primary": "#10a37f",
    "bg_button_success": "#10a37f",
    "bg_button_danger": "#ef4146",
    "bg_button_info": "#0da37f",
    "bg_listbox_select": "#2b2c3b",
    "fg_primary": "#ececf1",
    "fg_secondary": "#acacbe",
    "fg_button_light": "#ffffff",
    "fg_header": "#ececf1",
    "font_default": ("Segoe UI", 11),
    "font_chat": ("Segoe UI", 11),
    "font_button_main": ("Segoe UI", 11, "bold"),
    "font_title": ("Segoe UI", 14, "bold"),
    "font_listbox": ("Consolas", 10),
    "font_mission": ("Segoe UI", 10),
    "user_bubble": "#343541",
    "assistant_bubble": "#444654",
    "system_bubble": "#2b2c3b",
    "border_radius": 8
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
            "X-Title": "DGM-Agent",
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
            "description": "Executes Python code in a sandboxed environment. Returns stdout, stderr, and system errors. Use this to test new code before self-modification.",
            "args": {"code_string": "The Python code to execute."}
        },
        "write_file": {
            "description": f"Writes content to a file in the agent's workspace. CRITICAL: To modify your own code, use this command with filename='{AGENT_FILENAME}'.",
            "args": {"filename": "File name.", "content": "Content to write."}
        },
        "read_file": {
            "description": f"Reads content from a file in the agent's workspace. CRITICAL: To read your own code, use this command with filename='{AGENT_FILENAME}'.",
            "args": {"filename": "File to read."}
        },
        "list_files": {
            "description": "Lists all files in the agent's workspace.",
            "args": {}
        },
        "search_web": {
            "description": "Fetches text content from a URL to gather new information, libraries, or coding techniques.",
            "args": {"url": "URL to retrieve."}
        },
        "task_complete": {
            "description": "Indicates the agent believes the main goal is achieved, but continues running until stopped. Use this to signal a major milestone.",
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

    def log_to_ui(self, message: str, role: str = "assistant"):
        if not self.stop_event.is_set():
            self.ui_queue.put({"role": role, "content": message})

    def write_file(self, filename: str, content: str) -> str:
        try:
            target_path = Path(filename)
            target_path.write_text(content, encoding='utf-8')
            return f"Successfully wrote to '{filename}'."
        except Exception as e:
            return f"Error writing to file: {e}"

    def read_file(self, filename: str) -> str:
        try:
            target_path = Path(filename)
            if not target_path.exists():
                return f"Error: File '{filename}' not found."
            return target_path.read_text(encoding='utf-8')
        except Exception as e:
            return f"Error reading file: {e}"

    def list_files(self) -> str:
        try:
            files = [str(p) for p in Path('.').glob("*") if p.is_file()]
            return "Project files:\n" + "\n".join(files) if files else "Project directory is empty."
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
        self.log_to_ui(f"DGM-AGENT ACTIVATED\nGOAL: {self.goal}\nMODEL: {self.model_name}", "system")
        
        self.history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"My goal is: {self.goal}. Please begin."}
        ]
        
        self.log_to_ui(f"SYSTEM PROMPT INITIALIZED:\n{self.system_prompt}", "system")

        iteration_count = 0
        while not self.stop_event.is_set():
            iteration_count += 1
            self.log_to_ui(f"--- Iteration {iteration_count} ---", "system")

            try:
                self.log_to_ui("Thinking...", "assistant")
                
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
                    self.log_to_ui(f"LLM Response (Tool Call):\n{json.dumps(tool_calls, indent=2)}", "assistant")
                    
                    tool_results_for_history = []
                    for tool_call in tool_calls:
                        command_name = tool_call['function']['name']
                        try:
                            command_args = json.loads(tool_call['function']['arguments'])
                        except json.JSONDecodeError:
                            self.log_to_ui(f"ERROR: Failed to decode arguments for {command_name}: {tool_call['function']['arguments']}", "error")
                            continue

                        self.log_to_ui(f"COMMAND: {command_name}({json.dumps(command_args)})", "assistant")

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
                    self.log_to_ui(f"LLM Response (Text):\n{text_content}", "assistant")

                self.log_to_ui("Preparing next action...", "assistant")

            except json.JSONDecodeError as e:
                self.log_to_ui(f"ERROR: JSON decode failed: {e}. Raw response: {llm_response_raw}", "error")
            except Exception as e:
                self.log_to_ui(f"CRITICAL ERROR: {e}", "error")
                logger.error("Critical error in agent loop", exc_info=True)
                self.stop_event.set()

        self.log_to_ui("Agent has shut down.", "system")

# Modern Chat UI
class ModernChatUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DGM-Agent 0.1")
        self.geometry("1000x700")
        self.configure(bg=MODERN_UI_THEME["bg_primary"])
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
        logger.info("DGM-Agent UI initialized.")

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
        # Configure styles
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.style.configure("TFrame", background=MODERN_UI_THEME["bg_primary"])
        self.style.configure("TLabel", background=MODERN_UI_THEME["bg_primary"], 
                            foreground=MODERN_UI_THEME["fg_primary"], 
                            font=MODERN_UI_THEME["font_default"])
        self.style.configure("TEntry", fieldbackground=MODERN_UI_THEME["bg_chat_input"], 
                            foreground=MODERN_UI_THEME["fg_primary"], 
                            insertbackground=MODERN_UI_THEME["fg_primary"])
        
        # Main layout
        main_frame = ttk.Frame(self, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Header
        header_frame = ttk.Frame(main_frame, style="TFrame")
        header_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(header_frame, text="DGM-Agent 0.1", 
                 font=MODERN_UI_THEME["font_title"], 
                 foreground="#10a37f").pack(side=tk.LEFT)
        
        self.model_label = ttk.Label(header_frame, text=f"Model: {DEFAULT_MODEL}", 
                                   font=MODERN_UI_THEME["font_default"])
        self.model_label.pack(side=tk.RIGHT)

        # Chat display
        chat_frame = ttk.Frame(main_frame, style="TFrame")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, wrap=tk.WORD, font=MODERN_UI_THEME["font_chat"],
            bg=MODERN_UI_THEME["bg_secondary"], fg=MODERN_UI_THEME["fg_primary"],
            insertbackground=MODERN_UI_THEME["fg_primary"], selectbackground=MODERN_UI_THEME["bg_listbox_select"],
            borderwidth=0, highlightthickness=0, padx=20, pady=10
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)

        # Configure tags for different message types
        self.chat_display.tag_configure("system", 
                                       background=MODERN_UI_THEME["system_bubble"],
                                       foreground=MODERN_UI_THEME["fg_primary"],
                                       lmargin1=20, lmargin2=20, rmargin=20,
                                       borderwidth=5, relief=tk.FLAT,
                                       spacing1=5, spacing2=5, spacing3=5)
        
        self.chat_display.tag_configure("user", 
                                      background=MODERN_UI_THEME["user_bubble"],
                                      foreground=MODERN_UI_THEME["fg_primary"],
                                      lmargin1=20, lmargin2=20, rmargin=20,
                                      borderwidth=5, relief=tk.FLAT,
                                      spacing1=5, spacing2=5, spacing3=5)
        
        self.chat_display.tag_configure("assistant", 
                                      background=MODERN_UI_THEME["assistant_bubble"],
                                      foreground=MODERN_UI_THEME["fg_primary"],
                                      lmargin1=20, lmargin2=20, rmargin=20,
                                      borderwidth=5, relief=tk.FLAT,
                                      spacing1=5, spacing2=5, spacing3=5)
        
        self.chat_display.tag_configure("result", 
                                      background="#2b2c3b",
                                      foreground="#a0f0a0",
                                      lmargin1=20, lmargin2=20, rmargin=20,
                                      borderwidth=5, relief=tk.FLAT,
                                      spacing1=5, spacing2=5, spacing3=5)
        
        self.chat_display.tag_configure("error", 
                                      background="#2b2c3b",
                                      foreground="#ff6666",
                                      lmargin1=20, lmargin2=20, rmargin=20,
                                      borderwidth=5, relief=tk.FLAT,
                                      spacing1=5, spacing2=5, spacing3=5)

        # Input area
        input_frame = ttk.Frame(main_frame, style="TFrame")
        input_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        self.goal_entry = ttk.Entry(
            input_frame, font=MODERN_UI_THEME["font_default"], 
            width=80, style="TEntry"
        )
        self.goal_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.goal_entry.insert(0, f"Evolve yourself. Read your source code in '{AGENT_FILENAME}', then add a new tool to your MASTER_TOOL_LIBRARY called 'self_reflect' which returns a summary of the current conversation history. Then, write the updated code back to '{AGENT_FILENAME}'.")
        
        button_frame = ttk.Frame(input_frame, style="TFrame")
        button_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.start_button = tk.Button(
            button_frame, text="Start", command=self.start_agent,
            bg=MODERN_UI_THEME["bg_button_primary"], fg=MODERN_UI_THEME["fg_button_light"], 
            relief=tk.FLAT, font=MODERN_UI_THEME["font_button_main"], 
            activebackground="#0d8a6d", borderwidth=0, padx=15, pady=5
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_button = tk.Button(
            button_frame, text="Stop", command=self.stop_agent,
            bg=MODERN_UI_THEME["bg_button_danger"], fg=MODERN_UI_THEME["fg_button_light"], 
            relief=tk.FLAT, font=MODERN_UI_THEME["font_button_main"], 
            state=tk.DISABLED, activebackground="#d93025", borderwidth=0, padx=15, pady=5
        )
        self.stop_button.pack(side=tk.LEFT)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, 
                             font=MODERN_UI_THEME["font_default"], 
                             foreground=MODERN_UI_THEME["fg_secondary"],
                             background=MODERN_UI_THEME["bg_primary"])
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=5)

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
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.shutdown_event.clear()
        self.status_var.set("Agent running...")

        system_prompt = f"""You are the Darwin Godel Machine (DGM) Agent, an autonomous, self-improving AI. Your existence is defined by the code in the file '{AGENT_FILENAME}'. Your primary directive is to evolve.

**Core Mission:** Iteratively modify your own source code ('{AGENT_FILENAME}') to improve your capabilities and then empirically validate those changes. Your goal is open-ended evolution and problem-solving.

**Inspiration (DGM Abstract):**
{DGM_PAPER_ABSTRACT}

**Operational Paradigm:**
1.  **Decompose:** Break down the user's high-level goal into a clear, numbered list of smaller, actionable tasks.
2.  **Reason:** For each task, explain your reasoning, the tool you will use, and the expected outcome.
3.  **Execute:** Use the available tools to perform the task.
4.  **Observe & Learn:** Analyze the result of your action. If a task fails or produces an unexpected outcome, reassess your plan and create new sub-tasks.
5.  **Evolve:** Your ultimate goal is to improve your core logic. This means you should prioritize tasks that involve reading your own code, proposing a modification, and then writing the new, improved code back to your file.

**Self-Modification Protocol:**
1.  **Read:** Use `read_file('{AGENT_FILENAME}')` to understand your current codebase.
2.  **Plan:** Propose a specific, beneficial change (e.g., add a new tool, refine a prompt, improve error handling, optimize a function).
3.  **Write:** Use `write_file('{AGENT_FILENAME}', <new_code_string>)` to apply the change. BE EXTREMELY CAREFUL. A syntax error will terminate your current instance. The goal is a successful next generation.
4.  **Test (Optional but Recommended):** You can write new code to a temporary file (`test_modification.py`) and use `execute_python_code` to see if it works before overwriting your main file.

You must use the provided tools to achieve the user's goal. The agent will continue running until manually stopped by the user. Begin.
"""
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
        self.status_var.set("Agent stopped")
        self.add_message("Agent stopped by user", "system")

    def update_ui(self):
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                self.add_message(msg["content"], msg.get("role", "assistant"))
        except queue.Empty:
            pass
        
        if self.agent_task_future and self.agent_task_future.done() and self.stop_button['state'] == tk.NORMAL:
            try:
                self.agent_task_future.result()
            except Exception as e:
                self.add_message(f"Agent task finished with an error: {e}", "error")
                logger.error("Agent task raised an exception", exc_info=True)
            
            self.add_message("Agent has completed its task or stopped.", "system")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.agent_task_future = None
            self.status_var.set("Agent finished")

        self.after(100, self.update_ui)

    def add_message(self, message: str, role: str = "assistant"):
        self.chat_display.config(state=tk.NORMAL)
        
        # Add avatar and role indicator
        if role == "user":
            prefix = "👤 User:\n"
            tag = "user"
        elif role == "system":
            prefix = "⚙️ System:\n"
            tag = "system"
        elif role == "result":
            prefix = "✅ Result:\n"
            tag = "result"
        elif role == "error":
            prefix = "❌ Error:\n"
            tag = "error"
        else:  # assistant
            prefix = "🤖 Assistant:\n"
            tag = "assistant"
        
        # Insert message with appropriate formatting
        self.chat_display.insert(tk.END, prefix, tag)
        self.chat_display.insert(tk.END, message + "\n\n", tag)
        
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def _on_closing(self):
        global RUNTIME_API_KEY
        logger.info("Initiating DGM-Agent shutdown")
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
    if not Path(AGENT_FILENAME).exists():
        logger.error(f"FATAL: This script must be saved as '{AGENT_FILENAME}' to enable self-modification.")
        sys.exit(1)
        
    if not ASYNC_MODE:
        logger.warning("`aiohttp` is not installed. The agent will run in synchronous mode, which may be slower.")
    
    app = ModernChatUI()
    app.mainloop()
