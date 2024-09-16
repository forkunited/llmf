"""
Text completions mapping based on OpenAI's ChatGPT API
"""

import time
from typing import Dict, List, Tuple, Union

from llmf.completions import ChatGPTTextCompletionsClient, ChatGPTPrompt, ChatGPTRole
from llmf.corpora import SchematizedTextCorpusExample

from llmf.util import logging

from llmf.mappings.completions.base import (
    TextCompletionsMapping, TextCompletionsMappingDefinition, TextCompletionsMappingOutput
)

_LOGGING_SOURCE = "ChatGPTTextCompletionsMapping"
_LOGGING_COMPLETION_KEY = "Completion"
_LOGGING_COMPLETION_ERROR_KEY ="Error"

class ChatGPTTextCompletionsMapping(TextCompletionsMapping):
    """ Text mapping based on a text completions client """

    def __init__(
        self,
        definition: TextCompletionsMappingDefinition,
        gpt_client: ChatGPTTextCompletionsClient,
        prompt_prefix: ChatGPTPrompt
    ):
        """ Initialize a mapping """
        super().__init__(definition)
        self._gpt_client = gpt_client
        self._prompt_prefix = prompt_prefix

    @property
    def gpt_client(self) -> ChatGPTTextCompletionsClient:
        """ GPT client for this mapping """
        return self._gpt_client

    @property
    def prompt_prefix(self) -> ChatGPTPrompt:
        """ Prefix of the GPT prompt used by this mapping """
        return self._prompt_prefix

    def map(
        self,
        input: Union[SchematizedTextCorpusExample, Dict[str, str]],
        parse_error_default=None
    ) -> TextCompletionsMappingOutput:
        """
        Generate output dictionary for thi given input
        """
        user_input_prompt_message = self.definition.input_template.fill(input)
        gpt_prompt = self.prompt_prefix + (ChatGPTRole.USER, user_input_prompt_message)
        
        completion_start_time = time.time()
        completion = self.gpt_client.run(gpt_prompt)
        completion_end_time = time.time()

        logging.info(
            _LOGGING_SOURCE,
            _LOGGING_COMPLETION_KEY,
            {
                "GPT Model": self.gpt_client.parameters.model,
                "Prompt Character Count": gpt_prompt.character_length,
                "Prompt Definition Directory": self.definition.source_dir_path,
                "Prompt Guidelines": f"{str(self._prompt_prefix[0][1][:64])}...",
                "Prompt Input Message": user_input_prompt_message,
                "Completion": completion,
                "Completion Time (seconds)": completion_end_time - completion_start_time
            }
        )

        try:
            return TextCompletionsMappingOutput(
                raw=completion,
                parsed=self.definition.output_template.parse(completion),
                success=True,
                error_message=None
            )
        except ValueError as err:
            logging.info(
                _LOGGING_SOURCE,
                _LOGGING_COMPLETION_ERROR_KEY,
                {
                    "GPT Model": self.gpt_client.parameters.model,
                    "Prompt Definition Directory": self.definition.source_dir_path,
                    "Prompt Guidelines": f"{str(self._prompt_prefix[0][1][:64])}...",
                    "Prompt Input Message": user_input_prompt_message,
                    "Completion": completion,
                    "Error Type": "Parse",
                    "Error Default Value": str(parse_error_default),
                    "Error Message": err.args[0]
                }
            )

            if parse_error_default is None:
                raise err

            return TextCompletionsMappingOutput(
                raw=completion,
                parsed={ key: parse_error_default for key in self.definition.output_template.keys },
                success=False,
                error_message=err.args[0]
            )

    @property
    def prompt_template(self) -> ChatGPTPrompt:
        """
        Template for the prompt, with the final message containing input placeholders
        like {this}
        """
        return self.prompt_prefix + (ChatGPTRole.USER, self.definition.input_template.raw)

    @classmethod
    def _load_gpt_prompt_prefix(
        cls, definition: TextCompletionsMappingDefinition
    ) -> ChatGPTPrompt:
        """ GPT prompt template based on the given text completion mapping definion """
        messages: List[Tuple[ChatGPTRole, str]] = [(ChatGPTRole.SYSTEM, definition.guidelines)]
        for example in definition.examples:
            messages.extend([
                (ChatGPTRole.USER, definition.input_template.fill(example.inputs)),
                (ChatGPTRole.ASSISTANT, definition.output_template.fill(example.outputs))
            ])
        return ChatGPTPrompt(messages=messages)

    @classmethod
    def load_from_directory(
        cls, gpt_client: ChatGPTTextCompletionsClient, dir_path: str
    ) -> "ChatGPTTextCompletionsMapping":
        """
        Load a mapping from a directory containing guidelines.txt, input_template.txt,
        output_template.txt, and examples.tsv or examples.yaml.
        """
        mapping_definition = TextCompletionsMappingDefinition.load_from_directory(dir_path)
        return ChatGPTTextCompletionsMapping(
            definition=mapping_definition,
            prompt_prefix=cls._load_gpt_prompt_prefix(mapping_definition),
            gpt_client=gpt_client
        )


        