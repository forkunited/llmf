"""
Runs a mapping prompt on some data

Usage:

python -m scripts.run_gpt_mapping_prompt
  --prompt_dir_path [PATH TO DIRECTORY CONTAINING PROMPT]
  --input_file_path [PATH TO FILE CONTAINING INPUT DATA]
  --output_file_path [PATH TO FILE TO CONTAIN OUTPUT DATA]
"""

from argparse import ArgumentParser
from typing import Tuple

from llmf.completions import ChatGPTTextCompletionsClient
from llmf.corpora import SchematizedTextCorpus
from llmf.mappings import ChatGPTTextCompletionsMapping


def parse_args() -> Tuple[str, str, str]:
    """
    Parse arguments of script
    """
    parser = ArgumentParser(
        description="Script for running a prompt on some data"
    )

    parser.add_argument(
        '--prompt_dir_path',
        type=str,
        help="Path to directory containing files representing the mapping prompt",
        required=True
    )

    parser.add_argument(
        '--input_file_path',
        type=str,
        help="Path to TSV or YAML file containing input data",
        required=True
    )

    parser.add_argument(
        '--output_file_path',
        type=str,
        help="Path to TSV or YAML file containing output data",
        required=True
    )

    args = parser.parse_args()

    return args.prompt_dir_path, args.input_file_path, args.output_file_path

prompt_dir_path, input_file_path, output_file_path = parse_args()

ChatGPTTextCompletionsMapping.load_from_directory(
    gpt_client=ChatGPTTextCompletionsClient.load("gpt-4-0125-preview"),
    dir_path=prompt_dir_path
).map_corpus(
    SchematizedTextCorpus.load_from_file(input_file_path),
    include_inputs_in_output=True, include_debug_fields=True
).save(output_file_path)
