import llm_annotator.utils as utils
import re

from typing import Dict

@utils.component("build_system_prompt")
def build_system_prompt(system_prompt_path: str = "data/prompts/system_prompt.txt") -> str:
    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()
    return "system_prompt", system_prompt

# TO-DO: Change to flexible example structure
@utils.component("build_examples")
def build_examples(feature_dict: Dict, feature: str) -> str:
    template = ""
    """Create the example prompt template."""
    for i in range(1, 4):
        example_text = f"example{i}"
        student_text = feature_dict[feature][example_text]
        if not (student_text is None) and student_text != "":
            template += f"Student text: {student_text}\nOutput: 1\n"
    for i in range(1, 4):
        example_text = f"nonexample{i}"
        student_text = feature_dict[feature][example_text]
        if not (student_text is None) and student_text != "":
            template += f"Student text: {student_text}\nOutput: 0\n"
    return "examples", template


def generate_feature_def(feature_dict: Dict):
    template = ""
    for feature, meta in feature_dict.items():
        template += (
            f"{feature}: {meta.get('definition', feature)}. {meta['format']}\n\n"
            f"Examples: \n"
        )
    return template


# TO-DO: Implement batching(built-in and batching-API)
@utils.component("build_user_prompt")
def build_annotation_prompt(feature_dict: Dict,
                            annotation_prompt_path: str = "",
                            if_context: bool = False,
                            examples: str = "",
                            ) -> str:
    """Create the annotation prompt template."""
    definition = generate_feature_def(feature_dict=feature_dict)
    if annotation_prompt_path == "":
        template = "Task Overview: In this task, you will classify student utterances from a mathematics classroom transcript."\
                   "You will be given a list of dialogue utterances. Each utterance is formatted as <uttid>:<utterance text>."\
                   "Your goal is to categorize each utterance based on the following label. You must output in JSON format the annotations for all <uttid>. Output only the json annotations and no additional text.\n\n"

        template += definition
        template += examples

        template += (
            f"\n\nFormat the output as JSON with the <uttid> as keys.\n\n"
            "Utterances:\n {dialogue}\n\n"
        )
        if if_context:
            template += "Dialogue Context: {summary}\n"
        # TO-DO: Implement formats that support multiple scales
        # template += "{annotation_format}"
        return "prompt_template", template
    else:
        # TO-DO: Implement this read-in file
        with open(annotation_prompt_path, "r") as f:
            template = f.read()
        template = replace_template_variables(template=template, definition=definition, examples=examples)
        return "prompt_template", template


def replace_template_variables(template: str, definition: str, examples: str):
    template.format(definition=definition)
    template.format(examples=examples)
    return template


def extract_template_variables(template_text: str):
    pattern = r'{{([^{}]+)}}'
    matches = re.findall(pattern, template_text)
    return list(set(matches))
