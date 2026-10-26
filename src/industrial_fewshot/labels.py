"""Class lists shared by single-request MLLM detection and saved-run evaluation."""


def target_label_names(target: dict) -> list[str]:
    labels = target.get('labels')
    if labels is None:
        labels = target.get('config', {}).get('labels')
    if labels is None:
        return [target['label']]
    return [row['name'] if isinstance(row, dict) else row for row in labels]


def all_label_names(targets: list[dict]) -> list[str]:
    return list(dict.fromkeys(name for target in targets for name in target_label_names(target)))
