import yaml
from utils.match_and_save_code import MatchSaveCode


class Chatlog2Code:
    """Extract generated code from chat logs."""

    def __init__(self, log_file_path, save_code_path, code_filename="generated_code.py"):

        self.log_file_path = log_file_path
        self.save_code_path = save_code_path
        self.code_filename = code_filename

    def read_yamldata_to_pycode(self):

        file = open(self.log_file_path, "r", encoding="utf-8")
        file_data = file.read()
        file.close()

        task = yaml.load(file_data, Loader=yaml.FullLoader)
        data = task["response"]

        data_str = "\n".join(data)

        match_saver = MatchSaveCode(
            target_folder=self.save_code_path, response=data_str
        )
        matched_code = match_saver.match_code()

        pyfile_name = self.code_filename
        match_saver.save_code(
            matched_code=matched_code, pyfile_name=pyfile_name
        )

        return matched_code, pyfile_name
