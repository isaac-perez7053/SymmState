import os
import re
from typing import Dict, Optional
from symmstate import SymmStateCore

class TemplateManager(SymmStateCore):
    """Manages creation and tracking of Abinit template files"""
    
    def __init__(self, package_name: str = "symmstate"):
        self.package_name = package_name
        self.template_dir = self._get_template_dir()
        self.template_registry: Dict[str, str] = {}  # {template_name: file_path}
        self._load_existing_templates()

    def _get_template_dir(self) -> str:
        """Get path to templates directory using your existing path finder"""
        # Replace this with your actual path finding logic
        package_path = SymmStateCore.find_package_path()
        return os.path.join(package_path, "templates")

    def _load_existing_templates(self):
        """Load existing templates into registry on initialization"""
        if not os.path.exists(self.template_dir):
            return
            
        for file_name in os.listdir(self.template_dir):
            if file_name.endswith(".abi"):
                self.template_registry[file_name] = os.path.join(self.template_dir, file_name)

    def create_template(self, input_file: str, template_name: str) -> str:
        """
        Create new template with validation.
        Returns path to created template.
        """
        # Validate template name
        if not template_name.endswith(".abi"):
            template_name += ".abi"
            
        if self.template_exists(template_name):
            raise ValueError(f"Template '{template_name}' already exists")

        # Create template content
        with open(input_file, 'r') as f:
            content = f.read()
        template_content = self._replace_variables(content)

        # Ensure templates directory exists
        os.makedirs(self.template_dir, exist_ok=True)
        
        # Create full output path
        output_path = os.path.join(self.template_dir, template_name)
        
        # Write template file
        with open(output_path, 'w') as f:
            f.write(template_content)

        # Add to registry
        self.template_registry[template_name] = output_path
        
        return output_path

    def template_exists(self, template_name: str) -> bool:
        """Check if template exists in registry or filesystem"""
        exists_in_registry = template_name in self.template_registry
        exists_in_fs = os.path.exists(os.path.join(self.template_dir, template_name))
        return exists_in_registry or exists_in_fs

    def get_template_path(self, template_name: str) -> Optional[str]:
        """Get full path for a template by name"""
        return self.template_registry.get(template_name)
    
    def _replace_variables(self, content: str) -> str:
        """Replace ABINIT variables with template placeholders"""
        def is_numeric_line(line: str) -> bool:
            tokens = line.strip().split()
            if not tokens:
                return False
            for token in tokens:
                try:
                    float(token)
                except ValueError:
                    return False
            return True

        lines = content.splitlines()
        output_lines = []
        i = 0
        while i < len(lines):
            # Remove leading/trailing whitespace from the current line.
            line = lines[i].strip()
            if not line:
                output_lines.append("")
                i += 1
                continue

            # If the line contains no spaces, consider it a potential matrix header.
            if " " not in line:
                # If the next line exists and is numeric, treat this as a matrix block.
                if i + 1 < len(lines) and is_numeric_line(lines[i + 1]):
                    output_lines.append(line)
                    output_lines.append(f"{{{line}}}")
                    # Skip all subsequent numeric lines.
                    j = i + 1
                    while j < len(lines) and is_numeric_line(lines[j]):
                        j += 1
                    i = j
                    continue

            # Otherwise, treat the line as a scalar assignment.
            parts = line.split()
            if parts:
                var = parts[0]
                output_lines.append(f"{var} {{{var}}}")
            else:
                output_lines.append(line)
            i += 1

        return "\n".join(output_lines) + "\n"





    def remove_template(self, template_name: str):
        """Remove template from registry and filesystem"""
        if template_name in self.template_registry:
            os.remove(self.template_registry[template_name])
            del self.template_registry[template_name]