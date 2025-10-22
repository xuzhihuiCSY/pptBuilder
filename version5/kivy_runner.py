
import os
import sys
import threading
import subprocess
import queue
from functools import partial

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup

# ----------
# Utilities
# ----------

def default_script_path():
    """
    Try to find ppt_agent.py near this launcher; otherwise return empty string.
    """
    here = os.path.dirname(os.path.abspath(sys.argv[0]))
    guess = os.path.join(here, "ppt_agent.py")
    return guess if os.path.exists(guess) else ""


class TerminalRunner(BoxLayout):
    """
    Two-part UI:
      1) Top row: script path input, Start/Stop, command entry + Send.
      2) Bottom: read-only scrolling text showing the child process stdout/stderr (terminal mirror).
    """

    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", spacing=6, padding=8, **kwargs)

        # ---------- Top controls ----------
        top = BoxLayout(orientation="horizontal", size_hint_y=None, height="40dp", spacing=6)

        self.path_input = TextInput(text=default_script_path(), hint_text="Path to ppt_agent.py", multiline=False)
        self.browse_btn = Button(text="Browse")
        self.start_btn = Button(text="Start")
        self.stop_btn = Button(text="Stop", disabled=True)

        top.add_widget(Label(text="Script:", size_hint_x=None, width="70dp"))
        top.add_widget(self.path_input)
        top.add_widget(self.browse_btn)
        top.add_widget(self.start_btn)
        top.add_widget(self.stop_btn)

        # Command line row
        cmdrow = BoxLayout(orientation="horizontal", size_hint_y=None, height="40dp", spacing=6)
        self.cmd_input = TextInput(hint_text="Type a command for the script (press Enter or click Send)", multiline=False, disabled=True)
        self.send_btn = Button(text="Send", disabled=True)
        cmdrow.add_widget(Label(text="Input:", size_hint_x=None, width="70dp"))
        cmdrow.add_widget(self.cmd_input)
        cmdrow.add_widget(self.send_btn)

        # ---------- Terminal output ----------
        # Removed font_name to avoid missing font on some systems.
        self.terminal = TextInput(readonly=True, text="", size_hint_y=1, cursor_blink=True)
        self.terminal.hint_text = "Terminal output will appear here..."

        # Layout
        self.add_widget(top)
        self.add_widget(cmdrow)
        self.add_widget(self.terminal)

        # Process state
        self.proc = None
        self.reader_thread = None
        self.output_queue = queue.Queue()
        self._stop_reader = threading.Event()

        # Bindings
        self.browse_btn.bind(on_release=self.open_file_dialog)
        self.start_btn.bind(on_release=self.start_process)
        self.stop_btn.bind(on_release=self.stop_process)
        self.send_btn.bind(on_release=self.send_command)
        self.cmd_input.bind(on_text_validate=lambda *_: self.send_command())

        # Periodic pump to drain the queue into the UI
        Clock.schedule_interval(self._drain_output_queue, 0.05)

    # ---------- File chooser ----------
    def open_file_dialog(self, *_):
        chooser = FileChooserIconView(path=os.path.dirname(self.path_input.text) if self.path_input.text else os.getcwd(),
                                      filters=["*.py"])
        box = BoxLayout(orientation="vertical", spacing=6, padding=6)
        box.add_widget(chooser)
        btns = BoxLayout(size_hint_y=None, height="40dp", spacing=6)
        ok = Button(text="Use Selected")
        cancel = Button(text="Cancel")
        btns.add_widget(ok); btns.add_widget(cancel)
        box.add_widget(btns)
        popup = Popup(title="Select ppt_agent.py", content=box, size_hint=(0.9, 0.9))

        def choose_and_close(*_):
            if chooser.selection:
                self.path_input.text = chooser.selection[0]
            popup.dismiss()
        ok.bind(on_release=choose_and_close)
        cancel.bind(on_release=lambda *_: popup.dismiss())
        popup.open()

    # ---------- Process control ----------
    def start_process(self, *_):
        if self.proc and self.proc.poll() is None:
            self._append_text("[info] Process already running.\n")
            return

        script_path = self.path_input.text.strip()
        if not script_path or not os.path.exists(script_path):
            self._append_text(f"[error] Script not found: {script_path}\n")
            return

        try:
            # Reset state
            self.terminal.text = ""
            self._stop_reader.clear()

            # Launch child process: python ppt_agent.py
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            cmd = [sys.executable, '-X', 'utf8', script_path]
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace",
                env=env
            )

            # Enable/disable controls
            self.start_btn.disabled = True
            self.stop_btn.disabled = False
            self.cmd_input.disabled = False
            self.send_btn.disabled = False

            # Start reader thread
            self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.reader_thread.start()

            self._append_text(f"[started] {' '.join(cmd)}\n")

        except Exception as e:
            self._append_text(f"[error] Failed to start: {e}\n")
            self.proc = None

    def stop_process(self, *_):
        if not self.proc:
            return
        try:
            self._append_text("[info] Stopping process...\n")
            self._stop_reader.set()
            if self.proc.stdin:
                try:
                    # Graceful: send exit/quit just in case the CLI honors it
                    self.proc.stdin.write("exit\n")
                    self.proc.stdin.flush()
                except Exception:
                    pass
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()
        finally:
            self.proc = None
            self.start_btn.disabled = False
            self.stop_btn.disabled = True
            self.cmd_input.disabled = True
            self.send_btn.disabled = True

    def send_command(self, *_):
        if not (self.proc and self.proc.stdin):
            self._append_text("[warn] Process is not running.\n")
            return
        text = self.cmd_input.text
        if not text.strip():
            return
        try:
            self.proc.stdin.write(text + ("\n" if not text.endswith("\n") else ""))
            self.proc.stdin.flush()
            self._append_text(f"You> {text}\n")
            self.cmd_input.text = ""
        except Exception as e:
            self._append_text(f"[error] Failed to send: {e}\n")

    # ---------- I/O plumbing ----------
    def _reader_loop(self):
        try:
            for line in self.proc.stdout:
                if self._stop_reader.is_set():
                    break
                self.output_queue.put(line)
        except Exception as e:
            self.output_queue.put(f"[reader] {e}\n")
        finally:
            self.output_queue.put("\n[process exited]\n")
            # Reset buttons from the main thread
            Clock.schedule_once(lambda *_: self._on_process_exit())

    def _on_process_exit(self):
        self.start_btn.disabled = False
        self.stop_btn.disabled = True
        self.cmd_input.disabled = True
        self.send_btn.disabled = True

    def _drain_output_queue(self, *_):
        appended = False
        while True:
            try:
                line = self.output_queue.get_nowait()
            except queue.Empty:
                break
            self._append_text(line)
            appended = True
        if appended:
            # Auto-scroll to the end
            self.terminal.cursor = (len(self.terminal.text), 0)

    def _append_text(self, s: str):
        self.terminal.text += s

class KivyStarterApp(App):
    title = "PPT Agent Starter (Kivy UI)"

    def build(self):
        Window.size = (1100, 700)
        return TerminalRunner()

if __name__ == "__main__":
    KivyStarterApp().run()
