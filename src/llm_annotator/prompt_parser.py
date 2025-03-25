import llm_annotator.utils as utils
import re

from typing import Dict


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


# TO-DO: Implement batching(built-in and batching-API)
@utils.component("build_annotation_prompt")
def build_annotation_prompt(feature_dict: Dict,
                            annotation_prompt_path: str = "",
                            if_context: bool = False,
                            examples: str = "",
                            ) -> str:
    """Create the annotation prompt template."""
    if annotation_prompt_path == "":
        template = "Task Overview: In this task, you will classify student utterances from a mathematics classroom transcript."\
                   "You will be given a list of dialogue utterances. Each utterance is formatted as <uttid>:<utterance text>."\
                   "Your goal is to categorize each utterance based on the following label. You must output in JSON format the annotations for all <uttid>.\n\n"

        """
        for feature, meta in feature_dict.items():
            template += (
                f"{feature}: {meta.get('definition', feature)}. {meta['format']}\n\n"
                f"Examples: \n"
            )
        """

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
        #template = PromptBuilder.replace_template_variables(template, features, list(features.keys())[0])
        return None


def extract_template_variables(template_text: str):
    pattern = r'{{([^{}]+)}}'
    matches = re.findall(pattern, template_text)
    return list(set(matches))
