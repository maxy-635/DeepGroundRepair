import json
import os
import argparse
from collections import OrderedDict
from loguru import logger
from static_checking_utils.get_task_name import get_task_name


def get_all_json_files(directory):
    json_files = []
    for current_directory, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(current_directory, file))

    return sorted(json_files)


def save_json(source_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as json_file:
        json.dump(source_dict, json_file, indent=2, ensure_ascii=False)


class MessagesStatistics:

    def __init__(self, report_root, save_root, step_id, model_ids=None, experiment_ids=None, dlls=None):
        self.report_root = report_root
        self.save_root = save_root
        self.step_id = step_id
        self.model_ids = model_ids
        self.experiment_ids = experiment_ids
        self.dlls = dlls

    def _normalize_config(self, config_value, default_values):
        if config_value is None:
            return default_values
        if isinstance(config_value, str):
            return [config_value]
        return config_value

    def _sort_message_types(self, message_counts):
        type_order = ['convention', 'warning', 'error', 'refactor', 'fatal']
        type_weight = {type_: index for index, type_ in enumerate(type_order)}
        sorted_types = sorted(
            message_counts.items(),
            key=lambda item: type_weight.get(item[0], len(type_order))
        )
        return OrderedDict(sorted_types)

    def _format_symbol_counts(self, message_counts):
        for type_ in message_counts:
            total_count = sum(message_counts[type_].values())
            sorted_symbols = sorted(
                message_counts[type_].items(),
                key=lambda item: item[1],
                reverse=True
            )

            message_counts[type_] = OrderedDict()
            for symbol, symbol_count in sorted_symbols:
                percentage = (symbol_count / total_count) * 100 if total_count > 0 else 0
                message_counts[type_][symbol] = f"{symbol_count}({percentage:.0f}%)"

        return message_counts

    def _sort_summary(self, summary):
        for type_ in summary:
            sorted_symbols = sorted(
                summary[type_].items(),
                key=lambda item: item[1],
                reverse=True
            )
            summary[type_] = OrderedDict(sorted_symbols)

        return summary

    def _update_summary(self, summary, message_counts):
        for type_ in message_counts:
            if type_ not in summary:
                summary[type_] = {}

            for symbol, value in message_counts[type_].items():
                count = int(value.split('(')[0])
                if symbol not in summary[type_]:
                    summary[type_][symbol] = 0
                summary[type_][symbol] += count

        return summary

    def _count_messages_in_jsons(self, report_json_files):
        message_counts = {}

        for report_json_file in report_json_files:
            if os.path.getsize(report_json_file) == 0:
                continue

            with open(report_json_file, "r", encoding="utf-8") as file:
                data_list = json.load(file)

            for data in data_list:
                type_ = data.get('type')
                symbol = data.get('symbol')

                if not type_ or not symbol:
                    continue

                if type_ not in message_counts:
                    message_counts[type_] = {}
                if symbol not in message_counts[type_]:
                    message_counts[type_][symbol] = 0
                message_counts[type_][symbol] += 1

        if 'fatal' not in message_counts:
            message_counts['fatal'] = {}

        message_counts = self._sort_message_types(message_counts)
        message_counts = self._format_symbol_counts(message_counts)

        return message_counts

    def main(self):
        """Run the main workflow."""
        model_ids = self._normalize_config(self.model_ids, get_task_name(path=self.report_root))
        all_message_counts = {}

        for model_id in model_ids:
            step_id_path = os.path.join(self.report_root, model_id, self.step_id)
            if not os.path.isdir(step_id_path):
                continue

            all_message_counts[model_id] = {'summary': {}}

            dll_names = self._normalize_config(self.dlls, get_task_name(path=step_id_path))
            for dll in dll_names:
                dll = dll.lower()
                dll_path = os.path.join(step_id_path, dll)
                if not os.path.isdir(dll_path):
                    continue

                all_message_counts[model_id][dll] = {}

                report_json_files = get_all_json_files(directory=dll_path)
                message_counts = self._count_messages_in_jsons(report_json_files)
                experiment_id = (self.experiment_ids[0] if isinstance(self.experiment_ids, list)
                                 else self.experiment_ids) or "flattened"
                all_message_counts[model_id][dll][experiment_id] = message_counts
                all_message_counts[model_id]['summary'] = self._update_summary(
                    all_message_counts[model_id]['summary'], message_counts
                )

            all_message_counts[model_id]['summary'] = self._sort_summary(all_message_counts[model_id]['summary'])

            save_path = os.path.join(
                self.save_root,
                model_id,
                self.step_id,
                f'{model_id}_messages_statistics.json'
            )
            save_json(all_message_counts[model_id], save_path)

            logger.info(f"Messages statistics finished: {model_id}")

        return all_message_counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_id", type=str, required=True, help="step ids")
    parser.add_argument("--model_ids", type=str, required=True, help="List of model ids")
    parser.add_argument("--dlls", type=str, required=True, help="List of dll names")
    parser.add_argument("--experiment_id", type=str, required=True, help="experiment id")
    args = parser.parse_args()

    step_id = args.step_id
    model_ids = json.loads(args.model_ids)
    dlls = json.loads(args.dlls)
    experiment_id = args.experiment_id

    report_root = "evaluation/static_checking/report"
    save_root = "evaluation/static_checking/report"

    messages_statistics = MessagesStatistics(
        report_root=report_root,
        save_root=save_root,
        step_id=step_id,
        model_ids=model_ids,
        dlls=dlls,
        experiment_ids=experiment_id
    )
    messages_statistics.main()
