from utils.count_code_lines import count_code_lines
from utils.utils import get_all_files

class ComputeValidCode:
    """Implement the compute valid code component."""
    def __init__(self):
        pass

    def valid_code(self, code_path):
        empty_code_sum = 0
        valid_code_sum = 0

        all_files = get_all_files(code_path, file_type=".py")
        for file in all_files:
            code_lines = count_code_lines(file)
            if code_lines == 0:
                empty_code_sum += 1
            else:
                valid_code_sum += 1
    
        return valid_code_sum
