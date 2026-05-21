import os
from unittest.mock import patch
from rich.box import ROUNDED, ASCII
from src.terminal_manager import TerminalManager


class TestTerminalManagerName:
    def test_get_terminal_name_vscode(self):
        with patch.dict(os.environ, {"TERM_PROGRAM": "vscode"}, clear=True):
            assert TerminalManager.get_terminal_name() == "VS Code Terminal"

    def test_get_terminal_name_apple_terminal(self):
        with patch.dict(os.environ, {"TERM_PROGRAM": "Apple_Terminal"}, clear=True):
            assert TerminalManager.get_terminal_name() == "Apple Terminal"

    def test_get_terminal_name_iterm(self):
        with patch.dict(os.environ, {"TERM_PROGRAM": "iTerm.app"}, clear=True):
            assert TerminalManager.get_terminal_name() == "iTerm2"

    def test_get_terminal_name_ghostty(self):
        with patch.dict(os.environ, {"TERM_PROGRAM": "ghostty"}, clear=True):
            assert TerminalManager.get_terminal_name() == "Ghostty"

    def test_get_terminal_name_cursor(self):
        with patch.dict(os.environ, {"TERM_PROGRAM": "cursor"}, clear=True):
            assert TerminalManager.get_terminal_name() == "Cursor Terminal"

    def test_get_terminal_name_windsurf(self):
        with patch.dict(os.environ, {"TERM_PROGRAM": "windsurf"}, clear=True):
            assert TerminalManager.get_terminal_name() == "Windsurf Terminal"

    def test_get_terminal_name_unknown_term_program(self):
        with patch.dict(os.environ, {"TERM_PROGRAM": "SomeOtherTerm"}, clear=True):
            assert TerminalManager.get_terminal_name() == "SomeOtherTerm"

    def test_get_terminal_name_windows_terminal(self):
        with patch.dict(os.environ, {"WT_SESSION": "xyz123"}, clear=True):
            assert TerminalManager.get_terminal_name() == "Windows Terminal"

    def test_get_terminal_name_alacritty_window_id(self):
        with patch.dict(os.environ, {"ALACRITTY_WINDOW_ID": "123"}, clear=True):
            assert TerminalManager.get_terminal_name() == "Alacritty"

    def test_get_terminal_name_alacritty_term(self):
        with patch.dict(os.environ, {"TERM": "alacritty"}, clear=True):
            assert TerminalManager.get_terminal_name() == "Alacritty"

    def test_get_terminal_name_tmux(self):
        with patch.dict(os.environ, {"TMUX": "/tmp/tmux-1000/default,1234,1"}, clear=True):
            assert TerminalManager.get_terminal_name() == "tmux Session"

    def test_get_terminal_name_screen(self):
        with patch.dict(os.environ, {"STY": "1234.pts-0.host"}, clear=True):
            assert TerminalManager.get_terminal_name() == "screen Session"

    def test_get_terminal_name_generic_term(self):
        with patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=True):
            assert TerminalManager.get_terminal_name() == "xterm-256color"

    def test_get_terminal_name_windows_powershell(self):
        with patch.dict(os.environ, {"PSModulePath": "C:\\Powershell"}, clear=True):
            with patch("src.terminal_manager.os.name", "nt"):
                assert TerminalManager.get_terminal_name() == "PowerShell Console"

    def test_get_terminal_name_windows_cmd(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.terminal_manager.os.name", "nt"):
                assert TerminalManager.get_terminal_name() == "Windows Console Host (cmd.exe)"

    def test_get_terminal_name_unix_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.terminal_manager.os.name", "posix"):
                assert TerminalManager.get_terminal_name() == "Standard Terminal"


class TestTerminalManagerModern:
    def test_is_modern_windows_terminal(self):
        with patch.dict(os.environ, {"WT_SESSION": "xyz123"}, clear=True):
            assert TerminalManager.is_modern_terminal() is True

    def test_is_modern_vscode(self):
        with patch.dict(os.environ, {"TERM_PROGRAM": "vscode"}, clear=True):
            assert TerminalManager.is_modern_terminal() is True

    def test_is_modern_alacritty(self):
        with patch.dict(os.environ, {"TERM": "alacritty"}, clear=True):
            assert TerminalManager.is_modern_terminal() is True

    def test_is_modern_tmux(self):
        with patch.dict(os.environ, {"TMUX": "something"}, clear=True):
            assert TerminalManager.is_modern_terminal() is True

    def test_is_modern_generic_xterm(self):
        with patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=True):
            assert TerminalManager.is_modern_terminal() is True

    def test_is_modern_unix_non_nt(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.terminal_manager.os.name", "posix"):
                assert TerminalManager.is_modern_terminal() is True

    def test_is_modern_legacy_windows(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.terminal_manager.os.name", "nt"):
                assert TerminalManager.is_modern_terminal() is False


class TestTerminalManagerBoxStyle:
    def test_get_box_style_modern(self):
        with patch("src.terminal_manager.TerminalManager.is_modern_terminal", return_value=True):
            assert TerminalManager.get_box_style() == ROUNDED

    def test_get_box_style_legacy(self):
        with patch("src.terminal_manager.TerminalManager.is_modern_terminal", return_value=False):
            assert TerminalManager.get_box_style() == ASCII
