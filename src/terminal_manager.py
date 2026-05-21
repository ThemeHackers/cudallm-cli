import os
import sys
from rich.box import ROUNDED, ASCII

class TerminalManager:
    @staticmethod
    def get_terminal_name() -> str:
        """
        Returns the friendly name of the active terminal emulator/shell.
        Checks common env variables set by terminal emulators.
        """
        term_program = os.environ.get("TERM_PROGRAM")
        if term_program:
            tp = term_program.lower()
            if "vscode" in tp:
                return "VS Code Terminal"
            if "apple_terminal" in tp:
                return "Apple Terminal"
            if "iterm" in tp:
                return "iTerm2"
            if "ghostty" in tp:
                return "Ghostty"
            if "cursor" in tp:
                return "Cursor Terminal"
            if "windsurf" in tp:
                return "Windsurf Terminal"
            return term_program

        if os.environ.get("WT_SESSION"):
            return "Windows Terminal"
        if os.environ.get("ALACRITTY_WINDOW_ID") or "alacritty" in os.environ.get("TERM", "").lower():
            return "Alacritty"
        if os.environ.get("TMUX"):
            return "tmux Session"
        if os.environ.get("STY"):
            return "screen Session"

        term = os.environ.get("TERM")
        if term:
            return term

        # Windows default checks
        if os.name == "nt":
            if "PSModulePath" in os.environ:
                return "PowerShell Console"
            return "Windows Console Host (cmd.exe)"

        return "Standard Terminal"

    @staticmethod
    def is_modern_terminal() -> bool:
        """
        Returns True if the current terminal supports advanced formatting,
        such as 256 colors, unicode, or modern box-drawing characters.
        """
        if os.environ.get("WT_SESSION"):
            return True
            
        term_program = os.environ.get("TERM_PROGRAM", "").lower()
        if any(x in term_program for x in ["vscode", "apple_terminal", "iterm", "ghostty", "cursor", "windsurf"]):
            return True
            
        if os.environ.get("ALACRITTY_WINDOW_ID") or "alacritty" in os.environ.get("TERM", "").lower():
            return True
            
        if os.environ.get("TMUX") or os.environ.get("STY"):
            return True

        term = os.environ.get("TERM", "").lower()
        if any(x in term for x in ["xterm", "screen", "tmux", "color", "ansi"]):
            return True

        if os.name != "nt":
            return True

        return False

    @staticmethod
    def get_box_style():
        """
        Returns the best border style for Panels/Boxes.
        """
        if TerminalManager.is_modern_terminal():
            return ROUNDED
        return ASCII
