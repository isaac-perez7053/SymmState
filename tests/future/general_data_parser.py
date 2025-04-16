from symmstate.slurm import *
from typing import Union, List
import re
from symmstate.abinit import *


def grab_scalar(content, key) -> float:
    """Parse all parameters from Abinit file"""

    match = re.search(rf"{key}\s*:\s*(-?\d+\.\d+E?[+-]?\d*)", content)
    if match:
        # Replace 'd' or 'D' with 'e' for compatibility with Python floats
        value = match.group(1).replace("d", "e").replace("D", "e")
        return float(value)
    return None


def grab_array(content: str, param_name: str, dtype: type) -> Union[List, None]:
    """Parse values for a given parameter name with specified data type.

    Searches for a line starting with `param_name`, then processes tokens that may include
    multiplicative notation like "1.0*3" (i.e. multiply the value) and skips tokens
    that cannot be converted (e.g., string units).

    Parameters:
        content (str): The complete text content from which to search.
        param_name (str): The parameter keyword to search for at the beginning of a line.
        dtype (type): The type function to cast tokens (e.g., float or int).

    Returns:
        List: A list of values of type `dtype` or None if no matching line was found.
    """
    # Adjust regex to allow possible leading whitespace.
    regex_pattern = rf"^\s*{param_name}\s+([^\n]+)"
    match = re.search(regex_pattern, content, re.MULTILINE)
    if not match:
        return None

    # Replace commas with space and split into tokens.
    tokens = match.group(1).replace(",", " ").split()
    result = []
    for token in tokens:
        if "*" in token:
            # Token uses multiplicative shorthand, e.g., "1.0*3" meaning three copies of 1.0.
            parts = token.split("*")
            if len(parts) == 2:
                # Expecting a count and a value.
                count_str, val = parts
                try:
                    count = int(count_str)
                except ValueError:
                    try:
                        # In case the count is given as a float.
                        count = int(float(count_str))
                    except ValueError:
                        continue  # Skip if we cannot parse the count.
                try:
                    number = dtype(val)
                except ValueError:
                    continue  # Skip the value if it can't be converted.
                result.extend([number] * count)
        else:
            try:
                # Try converting the token to the desired type.
                number = dtype(token)
                result.append(number)
            except ValueError:
                # Ignore tokens that cannot be converted, such as a unit specifier.
                continue

    return result


with open("example_file.abo", "r") as f:
    content = f.read()
print(grab_scalar(content, "etotal"))
print(grab_array(content, "acell", float))
