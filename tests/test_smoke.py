#!/usr/bin/env python
"""Smoke tests for nano-diffusion project."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestImports(unittest.TestCase):
    """Test that all core modules can be imported."""

    def test_config_import(self):
        """Test that config module imports successfully."""
        # Mock modal to avoid dependency issues
        with patch.dict("sys.modules", {"modal": MagicMock()}):
            import config

            self.assertIsNotNone(config.DIRECTOR_MAP)
            self.assertTrue(len(config.DIRECTOR_MAP) > 0)

    def test_utils_import(self):
        """Test that utils module imports successfully."""
        import utils

        self.assertTrue(hasattr(utils, "CUSTOM_GRADIO_THEME"))

    def test_train_module_imports_exist(self):
        """Test that train module file exists and is importable at syntax level."""
        train_path = Path(__file__).parent.parent / "src" / "train.py"
        self.assertTrue(train_path.exists())

        # Basic syntax check
        with open(train_path) as f:
            content = f.read()
        try:
            compile(content, str(train_path), "exec")
        except SyntaxError as e:
            self.fail(f"Syntax error in train.py: {e}")

        # Check for key class definition in file content
        self.assertIn("class DirectorConfig", content)

    def test_serve_module_imports_exist(self):
        """Test that serve module file exists and is importable at syntax level."""
        serve_path = Path(__file__).parent.parent / "src" / "serve.py"
        self.assertTrue(serve_path.exists())

        # Basic syntax check
        with open(serve_path) as f:
            content = f.read()
        try:
            compile(content, str(serve_path), "exec")
        except SyntaxError as e:
            self.fail(f"Syntax error in serve.py: {e}")

        # Check for key app definition in file content
        self.assertIn("app = modal.App", content)


class TestConfig(unittest.TestCase):
    """Test configuration classes and constants."""

    def setUp(self):
        """Set up test fixtures."""
        with patch.dict("sys.modules", {"modal": MagicMock()}):
            import config

            self.config = config

    def test_director_map_structure(self):
        """Test that DIRECTOR_MAP has expected structure."""
        self.assertIsInstance(self.config.DIRECTOR_MAP, dict)
        for key, value in self.config.DIRECTOR_MAP.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, str)
            self.assertTrue(len(key) > 0)
            self.assertTrue(len(value) > 0)

    def test_evaluation_config_defaults(self):
        """Test EvaluationConfig default values."""
        eval_config = self.config.EvaluationConfig()
        self.assertIsNotNone(eval_config.test_prompts)
        self.assertIsNotNone(eval_config.eval_seeds)
        self.assertTrue(len(eval_config.test_prompts) > 0)
        self.assertTrue(len(eval_config.eval_seeds) > 0)
        self.assertGreater(eval_config.num_samples_per_test, 0)
        self.assertGreater(eval_config.eval_num_inference_steps, 0)

    def test_gradio_config_defaults(self):
        """Test GradioConfig default values."""
        gradio_config = self.config.GradioConfig()
        self.assertIsNotNone(gradio_config.example_prompts)
        self.assertTrue(len(gradio_config.example_prompts) > 0)
        self.assertTrue(len(gradio_config.title) > 0)
        self.assertTrue(len(gradio_config.description) > 0)

    def test_inference_config_values(self):
        """Test InferenceConfig has reasonable defaults."""
        inference_config = self.config.InferenceConfig()
        self.assertGreater(inference_config.num_inference_steps, 0)
        self.assertGreater(inference_config.guidance_scale, 0)
        self.assertGreater(inference_config.height, 0)
        self.assertGreater(inference_config.width, 0)


class TestProjectStructure(unittest.TestCase):
    """Test project structure and files."""

    def setUp(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent
        self.src_dir = self.project_root / "src"

    def test_required_files_exist(self):
        """Test that required project files exist."""
        required_files = [
            "pyproject.toml",
            "README.md",
            "src/__init__.py",
            "src/config.py",
            "src/train.py",
            "src/serve.py",
            "src/utils.py",
        ]

        for file_path in required_files:
            full_path = self.project_root / file_path
            self.assertTrue(full_path.exists(), f"Required file {file_path} not found")

    def test_pyproject_toml_structure(self):
        """Test that pyproject.toml has required structure."""
        pyproject_path = self.project_root / "pyproject.toml"
        content = pyproject_path.read_text()

        # Check for required sections
        self.assertIn("[project]", content)
        self.assertIn("name =", content)
        self.assertIn("dependencies =", content)

    def test_src_modules_are_python_files(self):
        """Test that all src/*.py files are valid Python syntax."""
        python_files = list(self.src_dir.glob("*.py"))
        self.assertGreater(len(python_files), 0)

        for py_file in python_files:
            with open(py_file) as f:
                content = f.read()

            # Basic syntax check - compile should not raise
            try:
                compile(content, str(py_file), "exec")
            except SyntaxError as e:
                self.fail(f"Syntax error in {py_file}: {e}")


class TestDataStructures(unittest.TestCase):
    """Test data structure validity."""

    def setUp(self):
        """Set up test fixtures."""
        with patch.dict("sys.modules", {"modal": MagicMock()}):
            import config

            self.config = config

    def test_lora_info_dataclass(self):
        """Test LoRAInfo dataclass structure."""
        lora_info = self.config.LoRAInfo(
            name="test",
            path="/test/path",
            trigger_phrase="test phrase",
            description="test description",
        )

        self.assertEqual(lora_info.name, "test")
        self.assertEqual(lora_info.path, "/test/path")
        self.assertEqual(lora_info.trigger_phrase, "test phrase")
        self.assertEqual(lora_info.description, "test description")

    def test_director_names_consistency(self):
        """Test that director names are consistent across the project."""
        expected_directors = {"anderson", "fincher", "nolan", "scorsese", "villeneuve"}
        actual_directors = set(self.config.DIRECTOR_MAP.keys())
        self.assertEqual(expected_directors, actual_directors)


if __name__ == "__main__":
    unittest.main()
