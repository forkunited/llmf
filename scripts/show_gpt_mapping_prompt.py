"""
Displays a mapping prompt template represented by files in a given directory

Usage:

python -m scripts.show_gpt_mapping_prompt
  --prompt_dir_path [PATH TO DIRECTORY CONTAINING PROMPT]
"""

from argparse import ArgumentParser

from llmf.completions import ChatGPTTextCompletionsClient
from llmf.mappings import ChatGPTTextCompletionsMapping


def parse_args() -> str:
    """
    Parse arguments of script
    """
    parser = ArgumentParser(
        description="Script for printing out a string reprsentation for a given prompt"
    )

    parser.add_argument(
        '--prompt_dir_path',
        type=str,
        help="Path to directory containing files representing the mapping prompt",
        required=True
    )

    args = parser.parse_args()

    return args.prompt_dir_path

prompt_dir_path = parse_args()

gpt_mapping = ChatGPTTextCompletionsMapping.load_from_directory(
    gpt_client=ChatGPTTextCompletionsClient.load("gpt-3.5-turbo"),
    dir_path=prompt_dir_path
)

print(gpt_mapping.prompt_template)
