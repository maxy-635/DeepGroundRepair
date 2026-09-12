import os
from loguru import logger


class MatchSaveCode:
    """Implement the match save code component."""

    def __init__(self, target_folder, response):
        self.response = response
        self.target_folder = target_folder

    def match_code(self):
        """Match code."""
        try:
            completion = self.response
            if "```python" in completion:
                start_line = completion.index("```python")
                completion = completion[start_line:].strip()
                completion = completion.replace("```python", "")
                ending_line = completion.index("```")
                matched = completion[:ending_line].strip()

            else:
                matched = None
                print("match failed with no python code")

        except:
            print("match failed with no python code")
            matched = None

        return matched

    def save_code(self, matched_code, task_name, pyfile_name):
        """Save code."""
        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)
        file_name = os.path.join(self.target_folder, pyfile_name)

        if matched_code == None:
            with open(file_name, "w") as file:
                pass
        else:
            logger.info("Python code extracted successfully")
            with open(file_name, "w") as file:
                file.write(matched_code)
