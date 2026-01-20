import sys
sys.setrecursionlimit(5000)
from setuptools import setup

APP = ['main.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': False,
    'plist': {
        'LSUIElement': True,
        'CFBundleName': 'GhostEagle',
        'CFBundleDisplayName': 'Ghost Eagle',
        'CFBundleGetInfoString': "Ghost Eagle Voice Transcriber",
        'CFBundleIdentifier': "com.dario.ghosteagle",
        'CFBundleVersion': "0.1.0",
        'CFBundleShortVersionString': "0.1.0",
        'NSMicrophoneUsageDescription': "Necesitamos acceso al micrófono para transcribir tu voz.",
        'NSAppleEventsUsageDescription': "Necesitamos controlar eventos para pegar texto automáticamente.",
        'NSAccessibilityUsageDescription': "Necesitamos accesibilidad para detectar tus atajos de teclado."
    },
    'packages': ['rumps', 'pynput', 'sounddevice', 'numpy', 'pyperclip', 'pyautogui', 'mlx_whisper', 'scipy'],
    'includes': ['mlx.core', 'mlx.nn', 'mlx.utils', 'mlx.optimizers', 'google', 'google.genai'],
    'excludes': ['rubicon'],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
