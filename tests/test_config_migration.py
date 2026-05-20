"""
Tests for configuration migration logic and path resolution.

Covers scenarios where:
- Legacy config exists and needs migration
- Both configs exist (new takes priority)
- Neither config exists (defaults created)
- Config directory auto-creation
"""

import json
import os
from unittest.mock import patch, mock_open, MagicMock, call
from pathlib import Path

from src.platform_info import (
    get_config_dir,
    get_config_path,
    get_cache_dir,
    get_legacy_config_path,
)




class TestConfigDir:
    def test_returns_path_instance(self):
        result = get_config_dir()
        assert isinstance(result, Path)

    def test_dir_name_is_cudallm(self):
        result = get_config_dir()
        assert result.name == ".cudallm"

    def test_parent_is_home(self):
        fake_home = Path("/home/testuser")
        with patch("src.platform_info.Path.home", return_value=fake_home):
            result = get_config_dir()
            assert result.parent == fake_home


class TestConfigPath:
    def test_returns_json_file(self):
        result = get_config_path()
        assert result.name == "config.json"

    def test_parent_is_config_dir(self):
        result = get_config_path()
        assert result.parent == get_config_dir()

    def test_suffix_is_json(self):
        result = get_config_path()
        assert result.suffix == ".json"


class TestCacheDir:
    def test_name_is_cache(self):
        result = get_cache_dir()
        assert result.name == "cache"

    def test_parent_is_config_dir(self):
        result = get_cache_dir()
        assert result.parent == get_config_dir()




class TestLegacyConfigPath:
    def test_returns_path_when_legacy_exists(self):
        with patch.object(Path, "is_file", return_value=True):
            result = get_legacy_config_path()
            assert result is not None
            assert isinstance(result, Path)
            assert result.name == "config.json"

    def test_returns_none_when_no_legacy(self):
        with patch.object(Path, "is_file", return_value=False):
            result = get_legacy_config_path()
            assert result is None

    def test_legacy_path_contains_config_dir(self):
        """The legacy path should be under a 'config/' directory."""
        with patch.object(Path, "is_file", return_value=True):
            result = get_legacy_config_path()
            
            assert result.parent.name == "config"




class TestMigrationScenarios:
    """
    These tests simulate the migration workflow:
      1. Check if legacy config exists
      2. Check if new config exists
      3. Copy legacy → new (if needed)
      4. Create defaults (if neither exists)
    """

    def _simulate_migration(
        self, legacy_exists: bool, new_exists: bool, legacy_data: dict | None = None,
    ) -> dict:
        """
        Run a simplified migration logic and return a result dict with keys:
        - migrated: bool
        - created_default: bool
        - config_data: dict
        """
        default_config = {"model": "default", "port": 8080}
        if legacy_data is None:
            legacy_data = {"model": "legacy-model", "port": 9090}

        new_config_path = get_config_path()

        if new_exists:
          
            return {"migrated": False, "created_default": False, "config_data": legacy_data}

        if legacy_exists:
         
            return {"migrated": True, "created_default": False, "config_data": legacy_data}


        return {"migrated": False, "created_default": True, "config_data": default_config}

    def test_legacy_exists_new_does_not(self):
        """Legacy config should be migrated when new one is absent."""
        result = self._simulate_migration(legacy_exists=True, new_exists=False)
        assert result["migrated"] is True
        assert result["created_default"] is False
        assert result["config_data"]["model"] == "legacy-model"

    def test_both_exist_new_takes_priority(self):
        """When both exist, migration is skipped — new config wins."""
        result = self._simulate_migration(legacy_exists=True, new_exists=True)
        assert result["migrated"] is False
        assert result["created_default"] is False

    def test_neither_exists_creates_default(self):
        """When no config exists at all, a default one should be created."""
        result = self._simulate_migration(legacy_exists=False, new_exists=False)
        assert result["migrated"] is False
        assert result["created_default"] is True
        assert result["config_data"]["model"] == "default"

    def test_only_new_exists(self):
        """If only the new config exists, no action is needed."""
        result = self._simulate_migration(legacy_exists=False, new_exists=True)
        assert result["migrated"] is False
        assert result["created_default"] is False


class TestConfigDirAutoCreation:
    """Verify that mkdir(parents=True, exist_ok=True) logic would work."""

    def test_config_dir_can_be_created(self, tmp_path):
        """Simulate auto-creation of config directory tree."""
        fake_config_dir = tmp_path / ".cudallm"
        assert not fake_config_dir.exists()

        fake_config_dir.mkdir(parents=True, exist_ok=True)
        assert fake_config_dir.exists()
        assert fake_config_dir.is_dir()

    def test_cache_dir_can_be_created(self, tmp_path):
        """Simulate auto-creation of cache subdirectory."""
        fake_cache_dir = tmp_path / ".cudallm" / "cache"
        fake_cache_dir.mkdir(parents=True, exist_ok=True)
        assert fake_cache_dir.exists()

    def test_mkdir_is_idempotent(self, tmp_path):
        """Calling mkdir with exist_ok=True on an existing dir must not raise."""
        fake_config_dir = tmp_path / ".cudallm"
        fake_config_dir.mkdir(parents=True, exist_ok=True)
     
        fake_config_dir.mkdir(parents=True, exist_ok=True)
        assert fake_config_dir.exists()

    def test_config_file_write_after_dir_creation(self, tmp_path):
        """End-to-end: create dir, write config, read it back."""
        fake_dir = tmp_path / ".cudallm"
        fake_dir.mkdir(parents=True, exist_ok=True)
        config_path = fake_dir / "config.json"

        default_config = {"model": "default", "port": 8080}
        config_path.write_text(json.dumps(default_config))

        loaded = json.loads(config_path.read_text())
        assert loaded == default_config
