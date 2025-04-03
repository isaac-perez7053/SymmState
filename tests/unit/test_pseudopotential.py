import unittest
from unittest.mock import patch, mock_open, call
import os
from symmstate.pseudopotentials import PseudopotentialManager 

class TestPseudopotentialManager(unittest.TestCase):
    @patch('os.path.isfile')
    @patch('os.listdir')
    def setUp(self, mock_listdir, mock_isfile):
        # Mock listdir and isfile to simulate existing files
        mock_listdir.return_value = ['pseudo1', 'pseudo2']
        mock_isfile.return_value = True
        
        # Initialize the manager with a predefined folder path
        self.manager = PseudopotentialManager('/mocked/folder/path')
        self.manager.pseudo_registry = {'pseudo1': '/mocked/folder/path/pseudo1'}

    @patch('os.path.isfile', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data="data")
    @patch('os.path.join')
    def test_add_pseudopotential(self, mock_join, mock_open, mock_isfile):
        # Mock os.path.join behavior
        mock_join.side_effect = lambda a, b: f'{a}/{b}'

        # Simulate adding a new pseudopotential
        file_path = '/mocked/source/path/new_pseudo'
        self.manager.add_pseudopotential(file_path)

        # Verify the file was copied and dictionary updated
        mock_open.assert_called_with('/mocked/folder/path/new_pseudo', 'wb')
        self.assertIn('new_pseudo', self.manager.pseudo_registry)

    @patch('os.remove')
    def test_delete_pseudopotential(self, mock_remove):
        # Simulate deletion of an existing pseudopotential
        self.manager.delete_pseudopotential('pseudo1')

        # Check that the pseudopotential was removed from both filesystem and dictionary
        mock_remove.assert_called_once_with('/mocked/folder/path/pseudo1')
        self.assertNotIn('pseudo1', self.manager.pseudo_registry)

    def test_get_pseudopotential(self):
        # Retrieve an existing pseudopotential
        path = self.manager.get_pseudopotential('pseudo1')
        self.assertEqual(path, '/mocked/folder/path/pseudo1')

        # Retrieve a non-existing pseudopotential
        path = self.manager.get_pseudopotential('non_existing')
        self.assertIsNone(path)

    def test_get_pseudopotential_real(self):
        self.real_manager = PseudopotentialManager()
        print(self.real_manager.pseudo_registry)


# Run the unit tests
if __name__ == '__main__':
    unittest.main()
