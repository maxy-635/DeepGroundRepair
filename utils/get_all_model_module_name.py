from utils.utils import get_all_files


class GetModelsFiles2Module:
    """Extract imported module names from generated model files."""

    def __init__(self, directory, file_type, project_name):
        self.project_name = project_name
        self.directory = directory
        self.file_type = file_type

    def get_processing_models_files(self):

        model_files_names = get_all_files(self.directory, self.file_type)
        model_module_modified_files = []
        for model_file in model_files_names:
            index = model_file.find(self.project_name)
            if index != -1:
                extracted_string = model_file[
                    index + len(self.project_name) + 1 : -3
                ]
                model_module_modified_file = extracted_string.replace(
                    "/", "."
                )
                model_module_modified_files.append(model_module_modified_file)
            else:
                print("String 'DeepCodeRAG' not found in the path.")

        return model_module_modified_files
