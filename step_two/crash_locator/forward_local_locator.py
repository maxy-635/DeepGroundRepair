from step_two.crash_locator.traceback_parser import TracebackParser
from step_two.crash_locator.called_module_extractor import SelfModuleCallExtractor


class ForwardLocalCallLocator:
    """Implement the forward local call locator component."""

    def __init__(self):
        self.traceback_parser = TracebackParser()
        self.call_extractor = SelfModuleCallExtractor()

    def locate(self, runtime_error_info):
        traceback_text = runtime_error_info.get("traceback")
        crash_location = self.traceback_parser.main(traceback_text)

        if crash_location.crash_stage != "forward":
            return None
        if not self.call_extractor.is_local_forward_call(crash_location.crash_code):
            return None

        return {
            "crash_location": {
                "crash_stage": getattr(crash_location, "crash_stage", "unknown"),
                "line_no": getattr(crash_location, "line_no", None),
                "crash_code": getattr(crash_location, "crash_code", ""),
            },
            "forward_definition": {
                "line_no": getattr(crash_location, "line_no", None),
                "original_call": getattr(crash_location, "crash_code", ""),
            },
            "init_definition": {},
            "root_cause_type": "Forward_Local_Crash",
        }
