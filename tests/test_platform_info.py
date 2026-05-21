"""
Comprehensive tests for src.platform_info.

Every public function is exercised across multiple code-paths by mocking
``os.name``, environment variables, and filesystem operations.
"""

from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

from src.platform_info import (
    detect_platform,
    is_colab,
    is_wsl,
    is_windows,
    is_linux,
    get_home_dir,
    get_config_dir,
    get_config_path,
    get_cache_dir,
    get_legacy_config_path,
    get_exe_extension,
    get_compiled_binary_name,
    get_default_llm_port,
    platform_display_name,
)


class TestIsWindows:
    def test_on_nt(self):
        with patch("src.platform_info.os.name", "nt"):
            assert is_windows() is True

    def test_on_posix(self):
        with patch("src.platform_info.os.name", "posix"):
            assert is_windows() is False


class TestIsLinux:
    def test_on_posix(self):
        with patch("src.platform_info.os.name", "posix"):
            assert is_linux() is True

    def test_on_nt(self):
        with patch("src.platform_info.os.name", "nt"):
            assert is_linux() is False


class TestIsColab:
    def test_colab_gpu_env_var(self):
        with patch.dict("os.environ", {"COLAB_GPU": "1"}, clear=False):
            assert is_colab() is True

    def test_colab_release_tag_env_var(self):
        with patch.dict("os.environ", {"COLAB_RELEASE_TAG": "v1"}, clear=False):
            assert is_colab() is True

    def test_no_env_vars_no_content_dir(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.platform_info.os.path.isdir", return_value=False):
                assert is_colab() is False

    def test_content_dir_with_colab_in_os_release(self):
        """Fallback detection via /etc/os-release containing 'colab'."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.platform_info.os.path.isdir", return_value=True):
                with patch("src.platform_info.os.path.isfile", return_value=True):
                    m = mock_open(read_data="NAME=Google Colab\n")
                    with patch("builtins.open", m):
                        assert is_colab() is True

    def test_content_dir_without_colab_in_os_release(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.platform_info.os.path.isdir", return_value=True):
                with patch("src.platform_info.os.path.isfile", return_value=True):
                    m = mock_open(read_data="NAME=Ubuntu\n")
                    with patch("builtins.open", m):
                        assert is_colab() is False

    def test_content_dir_os_release_read_error(self):
        """If reading /etc/os-release raises, we fall back to False."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.platform_info.os.path.isdir", return_value=True):
                with patch("src.platform_info.os.path.isfile", return_value=True):
                    with patch("builtins.open", side_effect=PermissionError):
                        assert is_colab() is False

    def test_content_dir_exists_but_os_release_missing(self):
        """/content exists but /etc/os-release does not."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.platform_info.os.path.isdir", return_value=True):
                with patch("src.platform_info.os.path.isfile", return_value=False):
                    assert is_colab() is False


class TestIsWSL:
    def test_on_nt_always_false(self):
        with patch("src.platform_info.os.name", "nt"):
            assert is_wsl() is False

    def test_microsoft_in_proc_version(self):
        with patch("src.platform_info.os.name", "posix"):
            m = mock_open(read_data="Linux version 5.10.0 microsoft-standard-WSL2")
            with patch("builtins.open", m):
                assert is_wsl() is True

    def test_wsl_keyword_in_proc_version(self):
        with patch("src.platform_info.os.name", "posix"):
            m = mock_open(read_data="Linux version 5.10.0 WSL2-kernel")
            with patch("builtins.open", m):
                assert is_wsl() is True

    def test_no_wsl_markers(self):
        with patch("src.platform_info.os.name", "posix"):
            m = mock_open(read_data="Linux version 5.15.0 generic")
            with patch("builtins.open", m):
                assert is_wsl() is False

    def test_proc_version_not_found(self):
        with patch("src.platform_info.os.name", "posix"):
            with patch("builtins.open", side_effect=FileNotFoundError):
                assert is_wsl() is False


class TestDetectPlatform:
    def test_returns_colab_when_colab(self):
        with patch("src.platform_info.is_colab", return_value=True):
            assert detect_platform() == "colab"

    def test_returns_windows_on_nt(self):
        with patch("src.platform_info.is_colab", return_value=False):
            with patch("src.platform_info.os.name", "nt"):
                assert detect_platform() == "windows"

    def test_returns_linux_on_posix(self):
        with patch("src.platform_info.is_colab", return_value=False):
            with patch("src.platform_info.os.name", "posix"):
                assert detect_platform() == "linux"


class TestDirectoryHelpers:
    def test_get_home_dir(self):
        fake_home = Path("/fake/home")
        with patch("src.platform_info.Path.home", return_value=fake_home):
            assert get_home_dir() == fake_home

    def test_get_config_dir(self):
        fake_home = Path("/fake/home")
        with patch("src.platform_info.Path.home", return_value=fake_home):
            assert get_config_dir() == fake_home / ".cudallm"

    def test_get_config_dir_name(self):
        result = get_config_dir()
        assert result.name == ".cudallm"

    def test_get_config_path(self):
        fake_home = Path("/fake/home")
        with patch("src.platform_info.Path.home", return_value=fake_home):
            result = get_config_path()
            assert result == fake_home / ".cudallm" / "config.json"
            assert result.name == "config.json"

    def test_get_cache_dir(self):
        fake_home = Path("/fake/home")
        with patch("src.platform_info.Path.home", return_value=fake_home):
            result = get_cache_dir()
            assert result == fake_home / ".cudallm" / "cache"
            assert result.name == "cache"


class TestGetLegacyConfigPath:
    def test_returns_path_when_file_exists(self):
        with patch.object(Path, "is_file", return_value=True):
            result = get_legacy_config_path()
            assert result is not None
            assert result.name == "config.json"
            assert "config" in str(result.parent)

    def test_returns_none_when_file_missing(self):
        with patch.object(Path, "is_file", return_value=False):
            result = get_legacy_config_path()
            assert result is None


class TestExeExtension:
    def test_exe_on_windows(self):
        with patch("src.platform_info.is_windows", return_value=True):
            assert get_exe_extension() == ".exe"

    def test_empty_on_linux(self):
        with patch("src.platform_info.is_windows", return_value=False):
            assert get_exe_extension() == ""


class TestCompiledBinaryName:
    def test_windows_extension(self):
        with patch("src.platform_info.is_windows", return_value=True):
            name = get_compiled_binary_name("abc123")
            assert name == "./temp_cuda_kernel_abc123.exe"

    def test_linux_extension(self):
        with patch("src.platform_info.is_windows", return_value=False):
            name = get_compiled_binary_name("run42")
            assert name == "./temp_cuda_kernel_run42.out"


class TestDefaultLLMPort:
    def test_colab_port(self):
        with patch("src.platform_info.is_colab", return_value=True):
            assert get_default_llm_port() == 1234

    def test_non_colab_port(self):
        with patch("src.platform_info.is_colab", return_value=False):
            assert get_default_llm_port() == 1234


class TestPlatformDisplayName:
    def test_windows_display(self):
        with patch("src.platform_info.detect_platform", return_value="windows"):
            with patch("src.platform_info.is_wsl", return_value=False):
                assert platform_display_name() == "Windows (Native)"

    def test_colab_display(self):
        with patch("src.platform_info.detect_platform", return_value="colab"):
            with patch("src.platform_info.is_wsl", return_value=False):
                assert platform_display_name() == "Google Colab (Linux)"

    def test_linux_display(self):
        with patch("src.platform_info.detect_platform", return_value="linux"):
            with patch("src.platform_info.is_wsl", return_value=False):
                assert platform_display_name() == "Linux"

    def test_wsl_display(self):
        with patch("src.platform_info.detect_platform", return_value="linux"):
            with patch("src.platform_info.is_wsl", return_value=True):
                assert platform_display_name() == "WSL (Windows Subsystem for Linux)"

    def test_unknown_platform_fallback(self):
        with patch("src.platform_info.detect_platform", return_value="freebsd"):
            with patch("src.platform_info.is_wsl", return_value=False):
                assert platform_display_name() == "freebsd"
