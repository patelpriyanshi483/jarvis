#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Jarvis desktop assistant – refined:
- Wake word: "hey jarvis" (enable with --wake)
- Sanitized TTS output (no emojis, symbols, code fences, ACTION lines, URLs)
- Auto-detects OpenRouter vs DeepSeek keys
- PC control tools (open apps, type, press keys, etc.)
- Web search via DuckDuckGo

Usage:
    python jarvis.py --voice --wake
    python jarvis.py --voice
    python jarvis.py --text
"""

import argparse
import json
import os
import uuid
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import requests
from duckduckgo_search import DDGS
from langdetect import detect
from gtts import gTTS
from playsound import playsound
import tempfile


# Optional voice libs
try:
    import speech_recognition as sr
    import pyttsx3
except Exception:
    sr = None
    pyttsx3 = None

# PC control
try:
    import pyautogui
except Exception:
    pyautogui = None

# .env support
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# ---------------------- CONFIG ----------------------

if load_dotenv:
    load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENROUTER_API_KEY")

# Auto-switch between OpenRouter vs DeepSeek
if API_KEY and API_KEY.startswith("sk-or-"):
    # OpenRouter route
    API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1")
    IS_OPENROUTER = True
else:
    # Native DeepSeek route
    API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    IS_OPENROUTER = False

DESTRUCTIVE_ACTIONS = {"shutdown", "restart", "delete_file"}

WAKE_WORD = "hey jarvis"

SYSTEM_PROMPT = """You are Jarvis, a helpful desktop AI assistant.
You can speak any language the user uses (Hindi, English, etc.).
You have the ability to request the host program to execute actions on the user's PC.

WHEN you want to execute ANY action on the user's PC, you MUST respond with EXACTLY ONE line:
ACTION <action_name> <json_args>

Examples:
ACTION open_app {"name":"notepad"}
ACTION type_text {"text":"Hello, I am Jarvis."}
ACTION hotkey {"keys":["ctrl","s"]}
ACTION search_web {"query":"Latest weather in New Delhi"}
ACTION screenshot {"path":"./screenshot.png"}
ACTION read_clipboard {}

Rules:
1) 'action_name' must be one of: open_app, type_text, press, hotkey, move_mouse, click, scroll, screenshot, search_web, read_clipboard, write_clipboard, system_command.
2) 'json_args' must be valid JSON, no backticks.
3) For multiple key presses with modifiers, use 'hotkey'.
4) Only use actions when you truly need them; otherwise just answer normally.
"""

# -------------------- UTIL: SANITIZE TTS --------------------

EMOJI_PATTERN = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
CODEBLOCK_PATTERN = re.compile(r"```.*?```", flags=re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r"`([^`]*)`")
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
URL_PATTERN = re.compile(r"https?://\S+")
ACTION_LINE_PATTERN = re.compile(r"^ACTION\s+[a-zA-Z_]+\s+\{.*\}\s*$", flags=re.DOTALL | re.MULTILINE)

def sanitize_for_tts(text: str) -> str:
    # Remove ACTION lines altogether
    text = ACTION_LINE_PATTERN.sub("", text)

    # Remove code blocks and inline code
    text = CODEBLOCK_PATTERN.sub("", text)
    text = INLINE_CODE_PATTERN.sub(r"\1", text)

    # Remove markdown links (keep visible text)
    text = MARKDOWN_LINK_PATTERN.sub(r"\1", text)

    # Remove URLs
    text = URL_PATTERN.sub("", text)

    # Remove emojis and high-plane unicode
    text = EMOJI_PATTERN.sub("", text)

    # Remove a bunch of markdown/bullets/symbols
    text = re.sub(r"[*_#>\[\]{}|~^`]", " ", text)

    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# -------------------- DATA STRUCTURES --------------------

@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str

@dataclass
class Conversation:
    messages: List[Message] = field(default_factory=list)

    def add(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))

    def as_openai_payload(self, model: str) -> Dict[str, Any]:
        return {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "stream": False,
            "temperature": 0.7,
        }

# -------------------- LLM CALL --------------------

def call_llm(conv: Conversation) -> str:
    if not API_KEY:
        raise RuntimeError("No API key found. Set DEEPSEEK_API_KEY (native) or OPENROUTER_API_KEY / DEEPSEEK_API_KEY (sk-or- prefix).")

    url = f"{API_BASE}/chat/completions"
    payload = conv.as_openai_payload(MODEL)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # OpenRouter likes these (optional but recommended)
    if IS_OPENROUTER:
        headers["HTTP-Referer"] = "http://localhost"
        headers["X-Title"] = "Jarvis Desktop Assistant"

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code == 401:
        raise RuntimeError(f"401 Unauthorized. Check if your key matches the endpoint. (OpenRouter key => use openrouter.ai)")
    resp.raise_for_status()
    data = resp.json()

    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected LLM response: {data}") from e

# -------------------- ACTION PARSER --------------------

ACTION_REGEX = re.compile(r'^ACTION\s+([a-zA-Z_]+)\s+(\{.*\})\s*$', re.DOTALL)

def try_parse_action(text: str):
    m = ACTION_REGEX.match(text.strip())
    if not m:
        return None, None
    name = m.group(1).strip()
    try:
        args = json.loads(m.group(2))
    except json.JSONDecodeError:
        return None, None
    return name, args

# -------------------- TOOLS --------------------

def tool_search_web(query: str, max_results: int = 5) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
        out = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            link = r.get("href", "")
            out.append(f"{i}. {title}\n{body}\n{link}\n")
        return "\n".join(out) if out else "No results."
    except Exception as e:
        return f"Search failed: {e}"

def tool_open_app(name: str):
    try:
        if sys.platform.startswith("win"):
            mapping = {
                "notepad": "notepad.exe",
                "calculator": "calc.exe",
                "cmd": "cmd.exe",
                "powershell": "powershell.exe",
            }
            exe = mapping.get(name.lower(), name)
            subprocess.Popen(exe, shell=True)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-a", name])
        else:
            subprocess.Popen([name])
        return f"Opened {name}."
    except Exception as e:
        return f"Failed to open {name}: {e}"

def tool_type_text(text: str):
    if not pyautogui:
        return "pyautogui not installed; can't type."
    pyautogui.typewrite(text, interval=0.02)
    return "Typed."

def tool_press(keys: List[str]):
    if not pyautogui:
        return "pyautogui not installed; can't press keys."
    for k in keys:
        pyautogui.press(k)
    return f"Pressed: {keys}"

def tool_hotkey(keys: List[str]):
    if not pyautogui:
        return "pyautogui not installed; can't press hotkey."
    pyautogui.hotkey(*keys)
    return f"Hotkey: {keys}"

def tool_move_mouse(x: int, y: int, duration: float = 0.2):
    if not pyautogui:
        return "pyautogui not installed; can't move mouse."
    pyautogui.moveTo(x, y, duration=duration)
    return f"Moved mouse to ({x},{y})."

def tool_click(button: str = "left", clicks: int = 1):
    if not pyautogui:
        return "pyautogui not installed; can't click."
    pyautogui.click(button=button, clicks=clicks)
    return f"Clicked {button} x{clicks}."

def tool_scroll(amount: int):
    if not pyautogui:
        return "pyautogui not installed; can't scroll."
    pyautogui.scroll(amount)
    return f"Scrolled {amount}."

def tool_screenshot(path: str = "./screenshot.png"):
    if not pyautogui:
        return "pyautogui not installed; can't screenshot."
    img = pyautogui.screenshot()
    img.save(path)
    return f"Screenshot saved to {path}"

def tool_read_clipboard():
    try:
        import pyperclip
    except Exception:
        return "pyperclip not installed."
    return pyperclip.paste()

def tool_write_clipboard(text: str):
    try:
        import pyperclip
    except Exception:
        return "pyperclip not installed."
    pyperclip.copy(text)
    return "Copied to clipboard."

def tool_system_command(cmd: str):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return out
    except subprocess.CalledProcessError as e:
        return f"Command failed with code {e.returncode}:\n{e.output}"
    except Exception as e:
        return f"Command failed: {e}"

TOOLBOX = {
    "search_web": tool_search_web,
    "open_app": tool_open_app,
    "type_text": tool_type_text,
    "press": tool_press,
    "hotkey": tool_hotkey,
    "move_mouse": tool_move_mouse,
    "click": tool_click,
    "scroll": tool_scroll,
    "screenshot": tool_screenshot,
    "read_clipboard": tool_read_clipboard,
    "write_clipboard": tool_write_clipboard,
    "system_command": tool_system_command,
}

# -------------------- VOICE --------------------

class Speaker:
    def say(self, text: str):
        clean = sanitize_for_tts(text)
        if not clean:
            return

        tmp_path = os.path.join(tempfile.gettempdir(), f"jarvis_{uuid.uuid4().hex}.mp3")
        try:
            tts = gTTS(text=clean, lang="en")
            tts.save(tmp_path)
            playsound(tmp_path)
        except Exception as e:
            print(f"[TTS error] {e}")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


class Listener:
    def __init__(self, language_hint: Optional[str] = None):
        self.language_hint = language_hint
        self.r = sr.Recognizer() if sr else None
        self.mic = sr.Microphone() if (sr and Listener.mic_available()) else None

    @staticmethod
    def mic_available() -> bool:
        if sr is None:
            return False
        try:
            sr.Microphone()
            return True
        except Exception:
            return False

    def listen_once(self) -> Optional[str]:
        if not self.r or not self.mic:
            return None
        with self.mic as source:
            print("Listening...")
            self.r.adjust_for_ambient_noise(source, duration=0.3)
            audio = self.r.listen(source)
        try:
            return self.r.recognize_google(audio, language=self.language_hint or "en-IN")
        except Exception as e:
            print(f"[STT error] {e}")
            return None

# -------------------- HELPERS --------------------

def maybe_confirm(action_name: str, args: Dict[str, Any]) -> bool:
    if action_name in DESTRUCTIVE_ACTIONS:
        print(f"Jarvis wants to run a destructive action: {action_name} {args}")
        choice = input("Allow? (y/N) ").strip().lower()
        return choice == "y"
    return True

def execute_action(action_name: str, args: Dict[str, Any]) -> str:
    fn = TOOLBOX.get(action_name)
    if not fn:
        return f"Unknown action '{action_name}'."
    try:
        if action_name in {"press", "hotkey"}:
            return fn(args.get("keys", []))
        if action_name == "move_mouse":
            return fn(int(args.get("x", 0)), int(args.get("y", 0)), float(args.get("duration", 0.2)))
        if action_name == "click":
            return fn(args.get("button", "left"), int(args.get("clicks", 1)))
        if action_name == "scroll":
            return fn(int(args.get("amount", 0)))
        if action_name == "screenshot":
            return fn(args.get("path", "./screenshot.png"))
        if action_name == "write_clipboard":
            return fn(args.get("text", ""))
        if action_name == "system_command":
            print(f"[system_command] {args.get('cmd')}")
            choice = input("This can be dangerous. Really run it? (y/N) ").strip().lower()
            if choice != "y":
                return "Cancelled."
            return fn(args.get("cmd", ""))
        if action_name == "search_web":
            return fn(args.get("query", ""), int(args.get("max_results", 5)))
        if action_name == "open_app":
            return fn(args.get("name", ""))
        if action_name == "type_text":
            return fn(args.get("text", ""))
        if action_name == "read_clipboard":
            return fn()
        # fallback generic
        return fn(**args)
    except Exception as e:
        return f"Tool execution failed: {e}"

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"

def strip_wake_word_if_present(text: str, wake_word: str) -> Optional[str]:
    t = text.strip().lower()
    ww = wake_word.lower()
    if t.startswith(ww):
        # remove the wake word and any separators after
        return text[len(wake_word):].lstrip(" ,.-:").strip()
    return None

# -------------------- MAIN --------------------

def main():
    parser = argparse.ArgumentParser(description="Refined Jarvis (DeepSeek/OpenRouter)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--voice", action="store_true", help="Voice mode (STT + TTS)")
    mode.add_argument("--text", action="store_true", help="Text mode")

    parser.add_argument("--wake", action="store_true", help="Require wake word 'hey jarvis' in voice mode")
    parser.add_argument("--wake-word", default=WAKE_WORD, help="Custom wake word (default: 'hey jarvis')")

    args = parser.parse_args()

    voice_mode = args.voice
    wake_mode = args.wake
    wake_word = args.wake_word

    if voice_mode and (sr is None or pyttsx3 is None):
        print("Voice libs missing. Install: pip install SpeechRecognition pyaudio pyttsx3")
        return

    if pyautogui is None:
        print("WARNING: pyautogui not installed. PC control features will be limited.")
        print("Install with: pip install pyautogui")

    speaker = Speaker() if voice_mode else None
    listener = Listener() if voice_mode else None

    conv = Conversation()
    conv.add("system", SYSTEM_PROMPT)

    print("=== Jarvis ready ===")
    if voice_mode:
        if wake_mode:
            print(f"Voice mode ON. Say '{wake_word}' to talk to Jarvis.")
        else:
            print("Voice mode ON. Speak anytime.")
    else:
        print("Text mode ON. Type 'quit' to exit.")

    while True:
        if voice_mode:
            heard = listener.listen_once()
            if not heard:
                continue

            # Exit phrases
            if heard.strip().lower() in {"quit", "exit", "stop"}:
                print("Goodbye!")
                if speaker:
                    speaker.say("Goodbye!")
                break

            # Wake word handling
            if wake_mode:
                remainder = strip_wake_word_if_present(heard, wake_word)
                if remainder is None or not remainder.strip():
                    # Not addressed to Jarvis; ignore.
                    continue
                user_text = remainder
            else:
                user_text = heard
        else:
            try:
                user_text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if user_text.lower() in {"quit", "exit", "stop"}:
                print("Goodbye!")
                break

        conv.add("user", user_text)

        try:
            assistant_text = call_llm(conv)
        except Exception as e:
            print(f"[LLM error] {e}")
            if voice_mode:
                speaker.say("There was an error talking to the model.")
            continue

        action_name, action_args = try_parse_action(assistant_text)

        if action_name:
            print(f"Jarvis requested action: {action_name} {action_args}")
            if maybe_confirm(action_name, action_args):
                tool_output = execute_action(action_name, action_args)
                print(f"[Tool Output]\n{tool_output}\n")
                conv.add("assistant", assistant_text)  # keep the trace
                conv.add("user", f"Tool result:\n{tool_output}")
                try:
                    final_answer = call_llm(conv)
                except Exception as e:
                    final_answer = f"(Model error after tool) {e}"
                print(f"Jarvis: {final_answer}\n")
                if voice_mode:
                    speaker.say(final_answer)
                conv.add("assistant", final_answer)
            else:
                msg = "Action denied."
                print(msg)
                if voice_mode:
                    speaker.say(msg)
                conv.add("assistant", msg)
        else:
            print(f"Jarvis: {assistant_text}\n")
            if voice_mode:
                speaker.say(assistant_text)
            conv.add("assistant", assistant_text)


if __name__ == "__main__":
    main()
