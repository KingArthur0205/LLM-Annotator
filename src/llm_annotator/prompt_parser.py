import llm_annotator.utils as utils

from typing import Dict


# TO-DO: Change to flexible example structure
@utils.component("build_example")
def build_examples(feature_dict: Dict, feature: str) -> str:
    template = ""
    """Create the example prompt template."""
    for i in range(1, 4):
        example_text = f"example{i}"
        student_text = feature_dict[feature][example_text]
        if not (student_text is None) and student_text != "":
            template += f"Student text: {student_text}\n{feature}: 1\n"
    for i in range(1, 4):
        example_text = f"nonexample{i}"
        student_text = feature_dict[feature][example_text]
        if not (student_text is None) and student_text != "":
            template += f"Student text: {student_text}\n{feature}: 0\n"
    return template

