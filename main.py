import sys
import threading
import time
import queue
import numpy as np
import sounddevice as sd
import mlx_whisper
import rumps
import pyperclip
import pyautogui
from pynput import keyboard
import subprocess
import os
import json
import logging

# Setup debug logging
logging.basicConfig(
    filename='/tmp/ghosteagle_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Ghost Eagle starting up...")


# Configuration
CONFIG_FILE = os.path.expanduser("~/.ghosteagle_config.json")
SAMPLE_RATE = 16000
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.audio_queue = queue.Queue()
        self.stream = None
        self.audio_data = []

    def callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        if self.recording:
            self.audio_queue.put(indata.copy())

    def start(self):
        self.recording = True
        self.audio_data = []
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, 
            channels=1, 
            callback=self.callback
        )
        self.stream.start()

    def stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Collect all data from queue
        while not self.audio_queue.empty():
            self.audio_data.append(self.audio_queue.get())
            
        if not self.audio_data:
            return None
            
        return np.concatenate(self.audio_data, axis=0)

class VoiceTranscriberApp(rumps.App):
    def __init__(self):
        super(VoiceTranscriberApp, self).__init__("🎙️")
        self.recorder = AudioRecorder()
        self.is_transcribing = False
        self.is_learning_hotkey = False
        self.learning_keys = set()
        
        # Load configuration
        self.config = self.load_config()
        self.model_path = self.config.get("model", DEFAULT_MODEL)
        self.hotkey_check = self.deserialize_hotkey(self.config.get("hotkey", ["Key.f5"]))
        
        # Pre-generate feedback sounds
        self.start_sound = self.generate_beep(880, 0.1)
        self.stop_sound = self.generate_beep(440, 0.1)
        
        # Menu items
        self.menu = [
            rumps.MenuItem("Model", icon=None, dimensions=(1, 1)),
            rumps.MenuItem("Shortcut", icon=None, dimensions=(1, 1)),
            rumps.separator
        ]
        
        # Hotkey listener initialization
        self.current_keys = set()
        
        # Submenus
        self.setup_model_menu()
        self.setup_shortcut_menu()
        
        self.setup_hotkey_listener()

    def generate_beep(self, frequency, duration=0.1, fs=SAMPLE_RATE):
        t = np.linspace(0, duration, int(fs * duration), False)
        tone = np.sin(2 * np.pi * frequency * t) * 0.1
        # Fade in/out
        envelope = np.concatenate([
            np.linspace(0, 1, int(fs * 0.01)),
            np.ones(int(fs * (duration - 0.02))),
            np.linspace(1, 0, int(fs * 0.01))
        ])
        return (tone * envelope).astype(np.float32)

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def save_config(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f)

    def serialize_hotkey(self, keys):
        serialized = []
        for k in keys:
            if isinstance(k, keyboard.Key):
                serialized.append(f"Key.{k.name}")
            elif isinstance(k, keyboard.KeyCode):
                # Use char if available, otherwise virtual key code
                if k.char:
                    serialized.append(f"Char.{k.char}")
                else:
                    serialized.append(f"Vk.{k.vk}")
        return serialized

    def deserialize_hotkey(self, key_strings):
        keys = set()
        for s in key_strings:
            try:
                if s.startswith("Key."):
                    keys.add(getattr(keyboard.Key, s.split(".", 1)[1]))
                elif s.startswith("Char."):
                    keys.add(keyboard.KeyCode.from_char(s.split(".", 1)[1]))
                elif s.startswith("Vk."):
                    keys.add(keyboard.KeyCode.from_vk(int(s.split(".", 1)[1])))
            except:
                pass # Ignore malformed keys
        return keys

    def setup_model_menu(self):
        models = [
            "mlx-community/whisper-tiny-mlx",
            "mlx-community/whisper-base-mlx",
            "mlx-community/whisper-small-mlx",
            "mlx-community/whisper-large-v3-turbo"
        ]
        img_model_menu = rumps.MenuItem("Select Model")
        for m in models:
            item = rumps.MenuItem(m, callback=self.change_model)
            if m == self.model_path:
                item.state = 1
            img_model_menu.add(item)
        self.menu["Model"].add(img_model_menu)

    def change_model(self, sender):
        for item in self.menu["Model"]["Select Model"].values():
            item.state = 0
        sender.state = 1
        self.model_path = sender.title
        self.config["model"] = self.model_path
        self.save_config()
        print(f"Model switched to: {self.model_path}")

    def setup_shortcut_menu(self):
        # Determine text for current shortcut
        current_serialized = self.config.get("hotkey", ["Key.f5"])
        
        presets = {
            "F5": {keyboard.Key.f5},
            "Cmd+Opt+R": {keyboard.Key.cmd, keyboard.Key.alt, keyboard.KeyCode.from_char('r')},
            "Right Option": {keyboard.Key.alt_r}
        }
        
        shortcut_menu = rumps.MenuItem("Select Shortcut")
        
        # Add presets
        for name, keys in presets.items():
            item = rumps.MenuItem(name, callback=self.change_shortcut_preset)
            if keys == self.hotkey_check:
                item.state = 1
            shortcut_menu.add(item)
            
        shortcut_menu.add(rumps.separator)
        shortcut_menu.add(rumps.MenuItem("Record New Shortcut...", callback=self.start_recording_hotkey))
        
        self.menu["Shortcut"].add(shortcut_menu)

    def change_shortcut_preset(self, sender):
        presets = {
            "F5": {keyboard.Key.f5},
            "Cmd+Opt+R": {keyboard.Key.cmd, keyboard.Key.alt, keyboard.KeyCode.from_char('r')},
            "Right Option": {keyboard.Key.alt_r}
        }
        
        for item in self.menu["Shortcut"]["Select Shortcut"].values():
            item.state = 0
        sender.state = 1
        
        if sender.title in presets:
            self.hotkey_check = presets[sender.title]
            self.config["hotkey"] = self.serialize_hotkey(self.hotkey_check)
            self.save_config()
            print(f"Shortcut changed to: {sender.title}")

    def start_recording_hotkey(self, sender):
        self.is_learning_hotkey = True
        self.learning_keys = set()
        self.title = "⌨️"
        rumps.notification("Ghost Eagle", "Recording Shortcut", "Press your desired key combination now.")

    def setup_hotkey_listener(self):
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        if self.is_learning_hotkey:
            self.learning_keys.add(key)
            return

        if key in self.hotkey_check:
            self.current_keys.add(key)
            
        if self.current_keys == self.hotkey_check and not self.recorder.recording and not self.is_transcribing:
            print("Start recording...")
            try:
                sd.play(self.start_sound, samplerate=SAMPLE_RATE)
            except Exception as e:
                print(f"Error playing sound: {e}")
            self.title = "🔴"
            self.recorder.start()

    def on_release(self, key):
        if self.is_learning_hotkey:
            # When any key is released during learning, we finalize the shortcut
            # using the set of keys that were pressed together.
            if self.learning_keys:
                self.hotkey_check = self.learning_keys.copy()
                self.config["hotkey"] = self.serialize_hotkey(self.hotkey_check)
                self.save_config()
                
                # Reset UI
                self.is_learning_hotkey = False
                self.title = "🎙️"
                rumps.notification("Ghost Eagle", "Shortcut Saved", "New shortcut has been saved.")
                
                # Update menu state (clear others)
                for item in self.menu["Shortcut"]["Select Shortcut"].values():
                    item.state = 0
            return

        if key in self.current_keys:
            self.current_keys.remove(key)
        
        # If we release any key of the hotkey combo and we are recording, stop.
        # This logic mimics "hold to record".
        # If hotkey is single key (F5), releasing F5 stops.
        # If hotkey is Cmd+Opt+R, releasing any of them stops.
        if self.recorder.recording:
            # Check if the combo is broken
            if not self.hotkey_check.issubset(self.current_keys): 
                print("Stop recording...")
                self.title = "🎙️"
                audio = self.recorder.stop()
                try:
                    sd.play(self.stop_sound, samplerate=SAMPLE_RATE)
                except Exception as e:
                    print(f"Error playing sound: {e}")
                    
                if audio is not None:
                    threading.Thread(target=self.transcribe_and_paste, args=(audio,)).start()

    def transcribe_and_paste(self, audio):
        self.is_transcribing = True
        try:
            print("Transcribing...")
            # Normalize audio to float32 range [-1, 1] if needed, sounddevice usually gives float32
            # mlx_whisper expects a string file or np array.
            
            # Simple VAD/Silence check could go here, but user wants ultra-fast.
            
            # Flatten if needed (sounddevice returns [frames, channels])
            audio_flat = audio.flatten()
            
            text = mlx_whisper.transcribe(
                audio_flat, 
                path_or_hf_repo=self.model_path,
                verbose=False
            )["text"]
            
            text = text.strip()
            print(f"Transcribed: {text}")
            
            if text:
                pyperclip.copy(text)
                # Small delay to ensure clipboard is ready
                time.sleep(0.2) 
                
                print("Pasting...")
                # Try pasting using AppleScript which is often more reliable on macOS for this
                try:
                    subprocess.run(
                        ["osascript", "-e", 'tell application "System Events" to keystroke "v" using command down'], 
                        check=True,
                        capture_output=True,
                        text=True
                    )
                except subprocess.CalledProcessError as e:
                    print(f"AppleScript paste failed: {e.stderr}")
                    if "1002" in e.stderr or "permiso" in e.stderr:
                         rumps.notification(
                             "⚠️ Faltan Permisos", 
                             "Error al Pegar", 
                             "Tu Terminal necesita permisos de 'Accesibilidad' y 'Automatización' para pegar texto. Revisa Ajustes del Sistema > Privacidad."
                         )
                    
                    print("Falling back to pyautogui...")
                    pyautogui.hotkey('command', 'v')
                except Exception as as_e:
                    print(f"Unexpected error during paste: {as_e}")
                    pyautogui.hotkey('command', 'v')
                
        except Exception as e:
            print(f"Error during transcription: {e}")
            rumps.notification("Transcriber Error", "Failed to transcribe", str(e))
        finally:
            self.is_transcribing = False

if __name__ == "__main__":
    app = VoiceTranscriberApp()
    app.run()
