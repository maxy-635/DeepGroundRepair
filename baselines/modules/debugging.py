import os
import re
import sys
import subprocess
from pathlib import Path
from baselines.modules.no_weight_validation import NoWeightRunner

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Validation:
    """Implement the validation component."""

    @staticmethod
    def remove_no_weight_runner_frames(traceback_text):
        """Remove no weight runner frames."""
        if not traceback_text:
            return traceback_text

        lines = str(traceback_text).splitlines()
        cleaned_lines = []
        index = 0

        while index < len(lines):
            if lines[index] != "Traceback (most recent call last):":
                cleaned_lines.append(lines[index])
                index += 1
                continue
            traceback_lines = [lines[index]]
            index += 1
            while index < len(lines) and lines[index] != "Traceback (most recent call last):":
                traceback_lines.append(lines[index])
                index += 1
            cleaned_lines.extend(Validation._remove_leading_runner_frames_from_traceback(traceback_lines))

        return "\n".join(cleaned_lines)

    @staticmethod
    def _remove_leading_runner_frames_from_traceback(traceback_lines):
        """Remove leading runner frames from traceback."""
        if len(traceback_lines) <= 1:
            return traceback_lines

        header = traceback_lines[:1]
        rest = traceback_lines[1:]
        blocks = []
        current_block = []

        for line in rest:
            if line.startswith("  File ") and current_block:
                blocks.append(current_block)
                current_block = [line]
            else:
                current_block.append(line)

        if current_block:
            blocks.append(current_block)

        skip_count = 0
        for block in blocks:
            file_line = block[0] if block else ""
            is_runner_frame = (
                file_line.startswith("  File ")
                and (
                    "no_weight_validation.py" in file_line
                    or "/runpy.py" in file_line
                    or "\\runpy.py" in file_line
                )
            )
            if not is_runner_frame:
                break
            skip_count += 1

        if skip_count == 0 or skip_count >= len(blocks):
            return traceback_lines

        return header + [line for block in blocks[skip_count:] for line in block]

    @staticmethod
    def remove_local_path_info(traceback_text):
        """Remove local path info."""
        if not traceback_text:
            return traceback_text

        home_dir = re.escape(os.path.expanduser("~"))
        project_root = re.escape(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

        cleaned_text = Validation.remove_no_weight_runner_frames(str(traceback_text))

        cleaned_text = re.sub(project_root + r"/\.?/?", "./", cleaned_text)
        env_lib_pattern = (
            home_dir
            + r"/(?:anaconda3|miniconda3|miniforge3|mambaforge|micromamba|\.conda)"
            + r"(?:/envs/[^/]+)?/lib"
        )
        cleaned_text = re.sub(env_lib_pattern, "", cleaned_text)
        cleaned_text = re.sub(home_dir + r"/", "", cleaned_text)
        cleaned_text = cleaned_text.replace("././", "./")

        cleaned_text = "\n".join(
            line for line in cleaned_text.splitlines()
            if not line.strip().startswith("which: no ccache in")
        )

        return cleaned_text

    def compile_code(self, pyfile):

        result = subprocess.run([sys.executable, "-W", "ignore", str(pyfile)], capture_output=True, text=True)

        if result.returncode == 0:
            verification = {
                "compile_status": "compile success",
                "compile_error": None,
            }
        else:
            verification = {
                "compile_status": "compile failed",
                "compile_error": self.remove_local_path_info(result.stderr or result.stdout),
            }

        return verification

    def compile_code_no_weight(self, pyfile):

        runner = Path(__file__).resolve().parent / "no_weight_validation.py"
        env = os.environ.copy()
        env[NoWeightRunner.RUN_TARGET_ENV] = str(pyfile)
        result = subprocess.run(
            [sys.executable, "-W", "ignore", str(runner)], capture_output=True, text=True, env=env,
        )

        if result.returncode == 0:
            verification = {
                "compile_status": "compile success",
                "compile_error": None,
            }
        else:
            verification = {
                "compile_status": "compile failed",
                "compile_error": self.remove_local_path_info(result.stderr or result.stdout),
            }

        return verification
