from step_one.document_processor import DocumentProcessor


class APIDocProcessor:
    """Convert one parsed API YAML file into an api_doc-level retrieval document."""

    def yaml2api_doc(self, file_path: str) -> dict:
        """Return api_name and api_doc text for one API YAML file."""
        (
            api_name,
            api_description,
            api_signature,
            api_details,
            api_usage_description,
            api_parameters,
            api_shape_formula,
            api_usage_example,
        ) = DocumentProcessor().doc2str(file_path)

        api_doc = "\n".join(
            [
                api_name,
                api_signature,
                # api_description,
                # api_details,
                api_usage_description,
                api_parameters,
                api_shape_formula,
                # api_usage_example,
            ]
        )

        return {
            "api_name": api_name,
            "api_doc": api_doc,
        }