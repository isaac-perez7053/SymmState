import re
from typing import Dict, List, Union, Optional
import numpy as np

class AbinitParser:
    """Parser for Abinit input files"""
    
    @staticmethod
    def parse_abinit_file(file_path: str) -> Dict:
        """Parse all parameters from Abinit file"""
        with open(file_path, 'r') as f:
            content = f.read()

        # Determine coordinate type (xcart or xred)
        coord_type = None
        if AbinitParser._parse_matrix(content, 'xcart', float) is not None:
            coord_type = 'xcart'
        elif AbinitParser._parse_matrix(content, 'xred', float) is not None:
            coord_type = 'xred'

        # Parse all variables
        parsed_data = {
            'acell': AbinitParser._parse_array(content, 'acell', float),
            'rprim': AbinitParser._parse_matrix(content, 'rprim', float),
            coord_type: AbinitParser._parse_matrix(content, coord_type, float) if coord_type else None,
            'znucl': AbinitParser._parse_array(content, 'znucl', int),
            'typat': AbinitParser._parse_typat(content),  # Special handling for typat
            'ecut': AbinitParser._parse_scalar(content, 'ecut', int),
            'ecutsm': AbinitParser._parse_scalar(content, 'ecutsm', float),
            'nshiftk': AbinitParser._parse_scalar(content, 'nshiftk', int),
            'nband': AbinitParser._parse_scalar(content, 'nband', int),
            'diemac': AbinitParser._parse_scalar(content, 'diemac', float),
            'toldfe': AbinitParser._parse_scalar(content, 'toldfe', float),
            'tolvrs': AbinitParser._parse_scalar(content, 'tolvrs', float),
            'tolsym': AbinitParser._parse_scalar(content, 'tolsym', float),
            'ixc': AbinitParser._parse_scalar(content, 'ixc', int),
            'kptrlatt': AbinitParser._parse_matrix(content, 'kptrlatt', int),
            'pp_dirpath': AbinitParser._parse_scalar(content, 'pp_dirpath', str),
            'pseudos': AbinitParser._parse_array(content, 'pseudos', str),
            'natom': AbinitParser._parse_scalar(content, 'natom', int),
            'ntypat': AbinitParser._parse_scalar(content, 'ntypat', int),
            'kptopt': AbinitParser._parse_scalar(content, 'kptopt', int),
            'chkprim': AbinitParser._parse_scalar(content, 'chkprim', int),
            'shiftk': AbinitParser._parse_array(content, 'shiftk', float),
            'nstep': AbinitParser._parse_scalar(content, 'nstep', int),
            'useylm': AbinitParser._parse_scalar(content, 'useylm', int),
        }

        # Determine the type of convergence criteria used
        init_methods = [parsed_data['toldfe'], parsed_data['tolvrs'], parsed_data['tolsym']]
        if sum(x is not None for x in init_methods) != 1:
            raise ValueError("Specify exactly one convergence criteria")
        
        conv_criteria = None
        if parsed_data['toldfe'] is not None:
            conv_criteria = 'toldfe'
        elif parsed_data['tolsym'] is not None:
            conv_criteria = 'tolsym'
        elif parsed_data['tolvrs'] is not None:
            conv_criteria = 'tolvrs'
        
        if conv_criteria is None:
            raise ValueError("Please specify a convergence criteria")
        parsed_data['conv_critera'] = conv_criteria

        # Remove None values
        return {k: v for k, v in parsed_data.items() if v is not None}

    @staticmethod
    def _parse_typat(content: str) -> Optional[List[int]]:
        """Handle typat values like '1 2 3*3' -> [1, 2, 3, 3, 3]"""
        match = re.search(r'typat\s+(.*?)\n', content)
        if not match:
            return None
        tokens = match.group(1).split()
        expanded = []
        for token in tokens:
            if '*' in token:
                val, count = token.split('*')
                expanded.extend([int(val)] * int(count))
            else:
                expanded.append(int(token))
        return expanded

    @staticmethod
    def _parse_array(content: str, key: str, dtype: type) -> Union[List, None]:
        match = re.search(fr"{key}\s+(.*?)\n", content)
        if not match:
            return None
        tokens = match.group(1).replace(',', ' ').split()
        return [dtype(x) for x in tokens]

    @staticmethod
    def _parse_matrix(content: str, key: str, dtype: type) -> Union[np.ndarray, None]:
        """Improved matrix parsing"""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if re.fullmatch(rf'\s*{key}\s*', line.strip()):
                matrix = []
                for j in range(i + 1, len(lines)):
                    next_line = lines[j].strip()
                    if not next_line or re.match(r'^\D', next_line):
                        break
                    matrix.append([dtype(x) for x in next_line.split()])
                return np.array(matrix) if matrix else None
        return None

    @staticmethod
    def _parse_scalar(content: str, key: str, dtype: type) -> Union[type, None]:
        match = re.search(fr"{key}\s+([\d\.+-dDeE]+)", content)
        return dtype(match.group(1)) if match else None