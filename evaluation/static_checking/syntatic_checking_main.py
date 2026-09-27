import os
import json
import argparse
from collections import OrderedDict
import pandas as pd
import pylint_plugin
from pylint.lint import Run
from pathlib import Path
from loguru import logger
from static_checking_utils.get_task_name import get_task_name
from utils.utils import get_all_files
from utils.count_code_lines import count_code_lines



class StaticAnalysis:
    """Implement the static analysis component."""

    def __init__(self, checking_rule, save_flag):
        self.checking_rule = checking_rule
        self.save_flag = save_flag

    def define_rules(self):
        rules = {
            # The current checking unit is a subdirectory, and there is no need to check for duplicate code
            'default': ("--disable=duplicate-code,",), 
        }

        return rules

    def _count_messages(self, message_type, message_total_counts):
        if message_type == 'convention':
            message_total_counts['convention'] += 1
        elif message_type == 'warning':
            message_total_counts['warning'] += 1
        elif message_type == 'error':
            message_total_counts['error'] += 1
        elif message_type == 'fatal':
            message_total_counts['fatal'] += 1
        elif message_type == 'refactor':
            message_total_counts['refactoring'] += 1

        return message_total_counts

    def _compute_score(self, code_lines, message_total_counts):
        convention_count = message_total_counts.get('convention')
        warning_count = message_total_counts.get('warning')
        error_count = message_total_counts.get('error')
        fatal_count = message_total_counts.get('fatal')
        refactor_count = message_total_counts.get('refactoring')

        return max(
            0,
            0 if fatal_count else 10.0 - (
                (float(5 * error_count + warning_count + refactor_count + convention_count) / code_lines) * 10
            )
        )

    def _report(self, json_file, module_dict_list):
        import json

        score_list, message_counts_list = [], []

        if not os.path.exists(json_file) or os.path.getsize(json_file) == 0:
            return {}

        with open(json_file, 'r', encoding='utf-8') as file:
            data_list = json.load(file)

        for module_dict in module_dict_list:
            code_lines = module_dict['total_code_lines']
            if code_lines == 0:
                continue

            message_total_counts = {
                'convention': 0,
                'warning': 0,
                'error': 0,
                'fatal': 0,
                'refactoring': 0
            }

            for data in data_list:
                if module_dict['module_name'] == data.get('module'):
                    symbol = data.get('symbol')
                    if symbol != 'false-positive-tensorflow-import':
                        message_type = data.get('type')
                        message_total_counts = self._count_messages(message_type, message_total_counts)

            score = self._compute_score(code_lines, message_total_counts)
            score_list.append(score)
            message_counts_list.append(message_total_counts)

        total_counts = {'convention': 0, 'warning': 0, 'error': 0, 'fatal': 0, 'refactoring': 0}
        for count in message_counts_list:
            for key, value in count.items():
                total_counts[key] += value

        total_counts['valid_pyfiles'] = len(score_list)
        total_counts['average_score'] = sum(score_list) / len(score_list) if score_list else None

        return total_counts

    def run(self, code_files, save_path):
        rules = self.define_rules()
        msg_template = "{path}: [{line},{column}]: [{msg_id}]: {msg}" 
        pylint_args = [
            "--load-plugins", "pylint_plugin",
            "--msg-template", msg_template,
            code_files, 
            *rules[self.checking_rule],
            "--reports=n"]

        module_dict_list = []
        if os.path.isdir(code_files):
            pylint_args.append("--recursive=y")
            all_files = get_all_files(directory=code_files, file_type='.py')

            for file in all_files:
                module_name = file.split('/')[-1].split('.')[0] 
                total_code_lines = count_code_lines(file)
                codeline_dict = {
                    "module_name": module_name, 
                    "total_code_lines": total_code_lines
                }
                module_dict_list.append(codeline_dict)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        filename = os.path.basename(os.path.normpath(save_path))

        if self.save_flag["save_json_flag"]:
            output_file_path_json = os.path.join(save_path, f'{filename}.json')
            pylint_args.append(f"--output-format=json:{output_file_path_json},colorized")

        # step1: run pylint
        logger.info("-----Running pylint-----")
        Run(pylint_args, exit=False)
        logger.info("-----Pylint finished, Begin Counting----")
        # Extract information from the saved json file and count the number of errors
        message_counts_socre = self._report(output_file_path_json, module_dict_list)

        return message_counts_socre

    def _normalize_config(self, config_value, default_values):
        if config_value is None:
            return default_values
        if isinstance(config_value, str):
            return [config_value]
        return config_value

    def main(self, response_root, report_root, step_id, model_ids=None, experiment_ids=None, dlls=None):
        """
        main function
        """
        model_ids = self._normalize_config(model_ids, get_task_name(path=response_root))
        for model_id in model_ids:
            step_one_path = os.path.join(response_root, model_id, step_id)
            if not os.path.isdir(step_one_path):
                continue

            dll_names = self._normalize_config(dlls, get_task_name(path=step_one_path))
            for dll in dll_names:
                dll = dll.lower()
                dll_path = os.path.join(step_one_path, dll)
                if not os.path.isdir(dll_path):
                    continue

                # Published response trees are flattened at model/step/framework;
                # the former experiment/batch directory is metadata only.
                task_names = get_task_name(path=dll_path)
                logger.info(f"Static analysis begin: {model_id}/{dll}")

                pd_cols = []
                for task in task_names:
                    task_code_files = os.path.join(dll_path, task)
                    save_path = os.path.join(report_root, model_id, step_id, dll, task)
                    count_result = self.run(
                        code_files=task_code_files,
                        save_path=save_path
                    )
                    count_result = OrderedDict([("Benchmark", task)] + list(count_result.items()))
                    pd_cols.append(count_result)

                df_data = pd.DataFrame(pd_cols)
                total = df_data.sum(axis=0, numeric_only=True)
                total['Benchmark'] = 'Total'
                df_data = pd.concat([df_data, total.to_frame().T], ignore_index=True)

                logger.info(f"Static analysis finished: {model_id}/{dll}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_id", type=str, required=True, help="step ids")
    parser.add_argument("--model_ids", type=str, required=True, help="JSON list of model ids")
    parser.add_argument("--dlls", type=str, required=True, help="JSON list of dll names")
    parser.add_argument("--experiment_id", type=str, required=True, help="experiment batch id")
    args = parser.parse_args()

    step_id = args.step_id
    model_id = json.loads(args.model_ids)
    dll = json.loads(args.dlls)
    experiment_id = args.experiment_id

    checking_rule = 'default'
    save_flag = {"save_json_flag": True}

    my_pylinter = StaticAnalysis(checking_rule=checking_rule, save_flag=save_flag)
    my_pylinter.main(
        response_root=Path("./response"),
        report_root=Path("evaluation/static_checking/report"),
        step_id=step_id,
        model_ids=model_id,
        experiment_ids=experiment_id,
        dlls=dll
    )
