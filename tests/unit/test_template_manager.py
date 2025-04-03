# tests/test_template_manager.py
import unittest
import uuid
import tempfile
from pathlib import Path
from symmstate.templates import TemplateManager

class TestTemplateManager(unittest.TestCase):
    def setUp(self):
        # Create unique temporary directory for each test
        self.test_dir = tempfile.TemporaryDirectory()
        self.package_path = Path(self.test_dir.name) / "symmstate_pkg"
        self.package_path.mkdir()
        
        # Create fresh templates directory
        self.templates_dir = self.package_path / "templates"
        self.templates_dir.mkdir(exist_ok=True)

        # Monkeypatch the path finding method
        self.original_find = TemplateManager.find_package_path
        TemplateManager.find_package_path = lambda *args: str(self.package_path)
        
        # Create unique sample file for each test
        self.sample_abi = self.templates_dir / f"test_{uuid.uuid4().hex}.abi"
        self.sample_abi.write_text("""acell 3*1.0
rprim
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
xred
0.0 0.0 0.0
0.5 0.5 0.5
natom 2
""")
        
        # Create fresh manager instance
        self.manager = TemplateManager()

    def tearDown(self):
        # Cleanup temporary directory
        self.test_dir.cleanup()
        # Restore original method
        TemplateManager.find_package_path = self.original_find

    def test_create_new_template(self):
        template_name = f"new_template_{uuid.uuid4().hex}.abi"
        template_path = self.manager.create_template(
            str(self.sample_abi),
            template_name
        )
        
        # Check registry
        self.assertIn(template_name, self.manager.template_registry)
        
        # Verify content replacement
        content = Path(template_path).read_text()
        self.assertIn("rprim\n{rprim}", content)
        self.assertIn("xred\n{xred}", content)
        self.assertIn("acell {acell}", content)

    def test_variable_replacement(self):
        test_content = """ecut 30
    kptrlatt
    1 2 3
    4 5 6
    """
        expected = "ecut {ecut}\nkptrlatt\n{kptrlatt}\n"
        result = self.manager._replace_variables(test_content)
        self.assertEqual(result, expected)

    def test_variable_replacement(self):
        test_content = """ecut 30
    kptrlatt
    1 2 3
    4 5 6
    """
        expected = """ecut {ecut}
    kptrlatt
    {kptrlatt}
    """
        result = self.manager._replace_variables(test_content)
        self.assertEqual(result, expected)

    def test_duplicate_template_creation(self):
        template_name = f"dup_template_{uuid.uuid4().hex}.abi"
        self.manager.create_template(str(self.sample_abi), template_name)
        with self.assertRaises(ValueError):
            self.manager.create_template(str(self.sample_abi), template_name)

    def test_template_exists(self):
        template_name = f"exists_test_{uuid.uuid4().hex}.abi"
        self.assertFalse(self.manager.template_exists(template_name))
        self.manager.create_template(str(self.sample_abi), template_name)
        self.assertTrue(self.manager.template_exists(template_name))

    def test_get_template_path(self):
        template_name = f"path_test_{uuid.uuid4().hex}.abi"
        self.assertIsNone(self.manager.get_template_path(template_name))
        expected = self.manager.create_template(str(self.sample_abi), template_name)
        self.assertEqual(self.manager.get_template_path(template_name), expected)

    def test_variable_replacement(self):
        test_content = """ecut 30
kptrlatt
1 2 3
4 5 6
"""
        expected = """ecut {ecut}
kptrlatt
{kptrlatt}
"""
        result = self.manager._replace_variables(test_content)
        self.assertEqual(result, expected)

    def test_template_registry_persistence(self):
        template_name = f"persistence_test_{uuid.uuid4().hex}.abi"
        self.manager.create_template(str(self.sample_abi), template_name)
        new_manager = TemplateManager()
        self.assertIn(template_name, new_manager.template_registry)