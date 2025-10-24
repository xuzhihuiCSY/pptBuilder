import os
import sys
import threading
import subprocess
import queue
import signal
import platform

from kivy.config import Config
Config.set("kivy", "exit_on_escape", "1")     # allow Esc to quit

from kivy.core.window import Window
Window.allow_vkeyboard = False
Window.softinput_mode = "pan"

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.modalview import ModalView
from kivy.graphics import Color, RoundedRectangle
from kivy.utils import get_color_from_hex

# -----------------
# Utility & Theming
# -----------------

APP_TITLE = "PPT Agent Console"
BG          = get_color_from_hex("#0f1220")
PANEL       = get_color_from_hex("#171a2b")
ACCENT      = get_color_from_hex("#6cb4ff")
ACCENT_DIM  = get_color_from_hex("#3c78b0")
TEXT        = get_color_from_hex("#e7eaf6")
MUTED       = get_color_from_hex("#aab1c7")
DANGER      = get_color_from_hex("#ff6b6b")
SUCCESS     = get_color_from_hex("#4cd17a")

def script_path_same_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "ppt_agent.py")

# ---------------
# Styled Widgets
# ---------------

class Pill(Label):
    def __init__(self, text, color, **kw):
        super().__init__(text=text, color=TEXT, size_hint=(None, None),
                         height="26dp", padding=(12, 6), **kw)
        self.bind(size=self._redraw, pos=self._redraw, texture_size=self._on_tex)
        self._bg_color = color

    def _on_tex(self, *_):
        self.width = self.texture_size[0] + 24

    def _redraw(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self._bg_color)
            RoundedRectangle(radius=[12], pos=self.pos, size=self.size)

class SolidButton(Button):
    def __init__(self, text, bg=ACCENT, bg_down=ACCENT_DIM, **kw):
        super().__init__(text=text, color=TEXT, size_hint=(None, None),
                         height="36dp", padding=(14, 10), **kw)
        self.background_normal = ""
        self.background_down = ""
        self._bg = bg
        self._bg_down = bg_down
        self.bind(state=self._refresh, size=self._refresh, pos=self._refresh)

    def _refresh(self, *_):
        col = self._bg_down if self.state == "down" else self._bg
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*col)
            RoundedRectangle(radius=[10], pos=self.pos, size=self.size)

class DangerButton(SolidButton):
    def __init__(self, text="Stop", **kw):
        super().__init__(text=text, bg=DANGER, bg_down=get_color_from_hex("#d95959"), **kw)

# ----------------
# Main UI/Behaviour
# ----------------

class ConsoleUI(BoxLayout):
    """
    Console:
      ┌ Header: Title | Status | Start / Stop
      ├ Terminal (read-only)
      └ Footer:  [› command input.....................][Send]
    """
    def __init__(self, **kw):
        super().__init__(orientation="vertical", spacing=10, padding=12, **kw)
        self._decorate_bg()

        # Process state
        self.proc = None
        self.reader_thread = None
        self.output_queue = queue.Queue()
        self._stop_reader = threading.Event()
        self.waiting_response = False

        self.last_output_line = ""

        # Header
        header = BoxLayout(orientation="horizontal", size_hint_y=None, height="46dp", spacing=10)
        title = Label(text=APP_TITLE, color=TEXT, font_size="18sp", bold=True, size_hint_x=1)
        self.status_pill = Pill("stopped", color=get_color_from_hex("#2a2e44"))
        self.start_btn = SolidButton("Start", width="90dp")
        self.stop_btn = DangerButton("Stop", width="90dp", disabled=True)
        header.add_widget(title)
        header.add_widget(self.status_pill)
        header.add_widget(self.start_btn)
        header.add_widget(self.stop_btn)

        # Terminal (read-only output)
        self.terminal = TextInput(
            text="",
            readonly=True,
            cursor_blink=True,
            size_hint_y=1,
            background_color=PANEL,
            foreground_color=TEXT,
            font_size="15sp",
            padding=(10, 10),
        )
        self.terminal.hint_text = "Output will appear here..."
        self.terminal.hint_text_color = (MUTED[0], MUTED[1], MUTED[2], 0.8)

        # Footer command bar
        footer = BoxLayout(orientation="horizontal", size_hint_y=None, height="44dp", spacing=10)
        self.cmd_label = Label(text="›", color=MUTED, size_hint_x=None, width="18dp", font_size="18sp")
        self.cmd_input = TextInput(
            text="",
            multiline=False,
            write_tab=False,
            background_color=get_color_from_hex("#101326"),
            foreground_color=TEXT,
            hint_text="Type a command for ppt_agent.py and press Enter",
            hint_text_color=(MUTED[0], MUTED[1], MUTED[2], 0.7),
            padding=(10, 10),
            disabled=True,
            font_size="15sp",
        )
        self.send_btn = SolidButton("Send", width="96dp", bg=ACCENT, bg_down=ACCENT_DIM, disabled=True)
        footer.add_widget(self.cmd_label)
        footer.add_widget(self.cmd_input)
        footer.add_widget(self.send_btn)

        # Loading modal (blocks input)
        self.loading = ModalView(auto_dismiss=False, background_color=(0, 0, 0, 0.6))
        self.loading.add_widget(Label(text="Working...", color=TEXT, font_size="20sp", bold=True))

        # Build layout
        self.add_widget(header)
        self.add_widget(self.terminal)
        self.add_widget(footer)

        # Bindings
        self.start_btn.bind(on_release=self._start)
        self.stop_btn.bind(on_release=self._stop)
        self.send_btn.bind(on_release=lambda *_: self._send())
        self.cmd_input.bind(on_text_validate=lambda *_: self._send())

        Window.bind(on_key_down=self._on_key_down)
        Clock.schedule_interval(self._drain_output_queue, 0.02)

        Window.size = (740, 550)

    # ---- visuals
    def _decorate_bg(self):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*BG)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[0, 0, 0, 0])
        self.bind(size=self._redraw_bg, pos=self._redraw_bg)

    def _redraw_bg(self, *_):
        self._decorate_bg()

    # ---- loading overlay
    def _start_loading(self):
        self.waiting_response = True
        self.cmd_input.disabled = True
        self.send_btn.disabled = True
        if not self.loading.parent:
            self.loading.open()

    def _stop_loading(self):
        self.waiting_response = False
        if self.loading:
            try:
                self.loading.dismiss()
            except Exception:
                pass
        if self.proc and self.proc.poll() is None:
            self.cmd_input.disabled = False
            self.send_btn.disabled = False

    def _clear_terminal(self):
        self.terminal.text = ""
        self.last_output_line = ""
        try:
            while True:
                self.output_queue.get_nowait()
        except queue.Empty:
            pass
        self._scroll_to_end()

    # ---- process lifecycle
    def _start(self, *_):
        if self.proc and self.proc.poll() is None:
            self._append("[info] process already running\n")
            return

        self._clear_terminal()

        script = script_path_same_dir()
        if not os.path.exists(script):
            self._append(f"[error] ppt_agent.py not found next to this file: {script}\n")
            return

        try:
            self._stop_reader.clear()
            self._stop_loading()
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            env["PYTHONUNBUFFERED"] = "1"

            # Launch ppt_agent.py with --verbose
            cmd = [sys.executable, "-u", "-X", "utf8", script, "--verbose"]

            # Windows: start in new process group so we can send CTRL_BREAK_EVENT
            creationflags = 0
            if platform.system() == "Windows":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=creationflags,
            )

            self.start_btn.disabled = True
            self.stop_btn.disabled = False
            self.cmd_input.disabled = False
            self.send_btn.disabled = False
            self._set_status("running", SUCCESS)

            self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.reader_thread.start()

            self._append(f"[started] {' '.join(cmd)}\n")

        except Exception as e:
            self._append(f"[error] failed to start: {e}\n")
            self.proc = None
            self._set_status("stopped", get_color_from_hex("#2a2e44"))

    def _stop(self, *_):
        if not self.proc:
            return
        self._append("[info] stopping...\n")
        try:
            self._stop_reader.set()

            # --- IMPORTANT: don't touch stdin first; let the process handle signals gracefully ---

            if platform.system() == "Windows":
                # Send CTRL_BREAK_EVENT (requires CREATE_NEW_PROCESS_GROUP at launch)
                try:
                    self.proc.send_signal(signal.CTRL_BREAK_EVENT)
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    # try terminate -> kill
                    try:
                        self.proc.terminate()
                        self.proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
            else:
                # Prefer SIGINT for graceful REPL shutdowns
                try:
                    self.proc.send_signal(signal.SIGINT)
                    self.proc.wait(timeout=2.5)
                except subprocess.TimeoutExpired:
                    try:
                        self.proc.terminate()  # SIGTERM
                        self.proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()      # SIGKILL
        finally:
            self._on_process_exit()

    def _on_process_exit(self):
        self.proc = None
        self.start_btn.disabled = False
        self.stop_btn.disabled = True
        self.cmd_input.disabled = True
        self.send_btn.disabled = True
        self._stop_loading()
        self._set_status("stopped", get_color_from_hex("#2a2e44"))

    def _reader_loop(self):
        """Character-by-character reader for immediate UI updates."""
        try:
            stdout = self.proc.stdout if self.proc else None
            while stdout:
                ch = stdout.read(1)
                if not ch:
                    break
                self.output_queue.put(ch)
        except Exception as e:
            self.output_queue.put(f"[reader] {e}\n")
        finally:
            self.output_queue.put("[process exited]\n")
            Clock.schedule_once(lambda *_: self._on_process_exit())

    # ---- I/O
    def _send(self):
        if not (self.proc and self.proc.stdin):
            self._append("[warn] process is not running\n")
            return
        text = self.cmd_input.text
        if not text.strip():
            return
        try:
            if not text.endswith("\n"):
                text += "\n"
            self._start_loading()
            self.proc.stdin.write(text)
            self.proc.stdin.flush()
            self._append(f"{text}")
            self.cmd_input.text = ""
        except Exception as e:
            self._append(f"[error] failed to send: {e}\n")
            self._stop_loading()

    def _drain_output_queue(self, *_):
        buf = []
        while True:
            try:
                buf.append(self.output_queue.get_nowait())
            except queue.Empty:
                break

        if buf:
            chunk = "".join(buf)
            self._append(chunk)
            if "\n" in chunk:
                self.last_output_line = self.terminal.text.rstrip("\n").split("\n")[-1]
            else:
                self.last_output_line = self.terminal.text.split("\n")[-1]

            if self.waiting_response:
                self._stop_loading()
            self._scroll_to_end()

    # ---- helpers
    def _append(self, s: str):
        self.terminal.text += s

    def _scroll_to_end(self):
        lines = self.terminal.text.split("\n")
        row = max(0, len(lines) - 1)
        col = len(lines[-1]) if lines else 0
        self.terminal.cursor = (col, row)

    def _set_status(self, text, color):
        self.status_pill.text = text
        self.status_pill._bg_color = color
        self.status_pill._redraw()

    def _on_key_down(self, _window, key, scancode, codepoint, modifiers):
        # Ctrl+L -> clear output
        if key == 108 and "ctrl" in modifiers:
            self.terminal.text = ""
            self._scroll_to_end()
            return True
        return False

class AgentConsoleApp(App):
    title = APP_TITLE
    def build(self):
        return ConsoleUI()

if __name__ == "__main__":
    AgentConsoleApp().run()
