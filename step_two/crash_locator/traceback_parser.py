import re
from dataclasses import dataclass
from typing import List


@dataclass
class CrashLocation:
    """Implement the crash location component."""

    file_path: str
    line_no: int
    func_name: str
    crash_code: str
    crash_stage: str = "unknown"


class TracebackParser:
    """Implement the traceback parser component."""

    FRAME_PATTERN = re.compile(r'^\s*File "(.+?)", line (\d+), in (.+)$')
    WRAPPER_FRAME_PATH_SUFFIXES = (
        "step_two/tensor_shape_trace/tensor_shape_tracer.py",
    )
    INIT_STAGE_NAME = "__init__"
    FORWARD_LIKE_STAGE_NAMES = {"forward", "call"}

    def is_wrapper_frame(self, file_path: str, crash_code: str):

        normalized_path = file_path.replace("\\", "/")
        is_wrapper_path = any(
            normalized_path.endswith(path_suffix)
            for path_suffix in self.WRAPPER_FRAME_PATH_SUFFIXES
        )
        if not is_wrapper_path:
            return False

        return crash_code.startswith("exec(compile(")

    def parse_user_frames(self, traceback_text: str):
        """Parse user frames."""

        lines = traceback_text.splitlines()
        frames: List[CrashLocation] = []

        for index, line in enumerate(lines):
            match = self.FRAME_PATTERN.match(line)
            if not match:
                continue

            file_path, line_no, func_name = match.groups()
            if "site-packages" in file_path:
                continue

            crash_code = lines[index + 1].strip() if index + 1 < len(lines) else ""
            if self.is_wrapper_frame(file_path, crash_code):
                continue

            frames.append(
                CrashLocation(
                    file_path=file_path,
                    line_no=int(line_no),
                    func_name=func_name.strip(),
                    crash_code=crash_code,
                )
            )

        return frames

    def classify_frame(self, frame: CrashLocation):
        """Classify frame."""

        if frame.func_name == self.INIT_STAGE_NAME:
            return "init"
        if frame.func_name in self.FORWARD_LIKE_STAGE_NAMES:
            return "forward"

        return "unknown"

    def select_most_relevant_frames(self, frames: List[CrashLocation]):
        """Select most relevant frames."""

        if not frames:
            raise ValueError("No user-code frame found in traceback.")

        selected_frame = None
        fallback_frame = None

        for frame in reversed(frames):
            crash_stage = self.classify_frame(frame)
            is_special_stage = crash_stage != "unknown"
            is_non_module_frame = frame.func_name != "<module>"

            if is_special_stage:
                frame.crash_stage = crash_stage
                selected_frame = frame
                break

            if fallback_frame is None and is_non_module_frame:
                fallback_frame = frame

        if selected_frame is None and fallback_frame is not None:
            fallback_frame.crash_stage = self.classify_frame(fallback_frame)
            selected_frame = fallback_frame

        if selected_frame is None:
            selected_frame = frames[-1]
            selected_frame.crash_stage = self.classify_frame(selected_frame)

        return selected_frame

    def main(self, traceback_text: str):
        """Run the main workflow."""

        frames = self.parse_user_frames(traceback_text)
        most_relevant_frame = self.select_most_relevant_frames(frames)

        return most_relevant_frame
