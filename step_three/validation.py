import subprocess
from loguru import logger
from utils.utils import get_all_files



class Validation:
    """Validate generated programs by executing them in a subprocess."""
    def compile_code(self, pyfile):
        """Execute one Python file and return its validation result."""
        result=subprocess.run(["python", pyfile], capture_output=True, text=True)

        if result.returncode==0:
            compile_status = "compile success"
            compile_error = None
        else:
            compile_status = "compile failed"
            compile_error = result.stderr or result.stdout

        verification = {"pyfile_path": str(pyfile),
                        "compile_status": str(compile_status),
                        "compile_error": str(compile_error)}

        log_msg = "\n".join(
            [f"{[k]}: {v}" for k, v in verification.items()]
        )
        logger.info(f"\n[verification]:\n{log_msg}")

        return verification

    def batch_compile_codes(self, code_path):

        pyfile_paths = get_all_files(code_path, ".py")
        verifications = []

        for pyfile_path in pyfile_paths:
            try:
                verification = self.compile_code(pyfile_path)
                verifications.append(verification)

            except Exception as e:
                logger.error(f"Compile failed: {e}")

        return verifications
