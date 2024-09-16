"""
Text completions based mapping
"""

import re

from abc import ABC, abstractmethod
from attrs import define
from os import path
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar, Union

from llmf.corpora import SchematizedTextCorpus, SchematizedTextCorpusExample
from llmf.mappings.example import TextMappingExample

T = TypeVar('T')

@define(frozen=True)
class TextCompletionsMappingTemplatePart:
    """
    Part of a text completions template.  Each part is
    either a key like {this}, or just plaintext.
    """
    raw: str
    key: Optional[str]

@define(frozen=True)
class TextCompletionsMappingTemplate:
    """
    Template for input or output from a text completions mapping.
    Template keys have a form like {this}.  Note that the template
    is guaranteed to have at least one part, and alternate between
    keys and non-keys.
    """
    raw: str
    parts: Tuple[TextCompletionsMappingTemplatePart,...]
    keys: Tuple[str,...]

    def fill(self, values: Dict[str, str]) -> str:
        """ Fills this template with values for the given keys """
        filled_template = self.raw
        for key in self.keys:
            if key not in values:
                raise ValueError(f"Key {key} missing from dictionary for template '{self.raw}'")
            filled_template = filled_template.replace("{" + key + "}", values[key])
        return filled_template

    def parse(self, text: str) -> Dict[str, str]:
        """
        Parse text into mapping from keys to their values within the text
        such that they match the template
        """
        parsed_keys_to_values: Dict[str, str] = {}
        cur_text = text.strip()
        cur_key_part_index = 0
        if self.parts[0].key is None:
            if not text.startswith(self.parts[0].raw):
                raise ValueError(f"Expected '{self.parts[0].raw}' at start of text '{text}'.")
            cur_text = cur_text[len(self.parts[0].raw):]
            cur_key_part_index = 1
    
        while cur_key_part_index < len(self.parts):
            if cur_key_part_index == len(self.parts) - 1:
                parsed_keys_to_values[self.parts[cur_key_part_index].key] = cur_text
            else:
                if self.parts[cur_key_part_index + 1].raw not in cur_text:
                    raise ValueError(f"Expected '{self.parts[cur_key_part_index+1].raw}' in text '{cur_text}'.")
                next_non_key_char_index = cur_text.index(self.parts[cur_key_part_index + 1].raw)
                parsed_keys_to_values[self.parts[cur_key_part_index].key] = cur_text[:next_non_key_char_index]
                cur_text = cur_text[next_non_key_char_index + len(self.parts[cur_key_part_index + 1].raw):]

            cur_key_part_index += 2

        return parsed_keys_to_values

    @classmethod
    def _parse_template_keys(cls, raw: str) -> Tuple[str,...]:
        """
        Extracts a set of keys like {this} from the given template string, returning
        them in the order they occur in the string, but with each only occurring at most
        once
        """
        template_keys = re.findall(r'\{([^}]*)\}', raw)
        template_key_set = set()
        unique_keys = []
        for key in template_keys:
            if key not in template_key_set:
                template_key_set.add(key)
                unique_keys.append(key)
        return tuple(unique_keys)

    @classmethod
    def _parse_template_into_parts(cls, raw: str) -> Tuple[TextCompletionsMappingTemplatePart,...]:
        """
        Parses a given raw template string into parts that are either
        plaintext or keys like {this}
        """
        template_parts: List[TextCompletionsMappingTemplatePart] = []

        # Regular expression to match keys and non-key text
        pattern = re.compile(r'(\{[^}]*\})|([^{}]+)')
        
        # Iterate over matches
        for pattern_match in pattern.finditer(raw):
            key_match, text_match = pattern_match.groups()
            if key_match:
                # Strip the curly braces from the key and append
                template_parts.append(TextCompletionsMappingTemplatePart(key_match, key_match.strip('{}')))
            elif text_match:
                # Append the text as is
                template_parts.append(TextCompletionsMappingTemplatePart(text_match, None))

            if len(template_parts) > 1 and (template_parts[-1].key is None) == (template_parts[-2].key is None):
                raise ValueError("Keys and non-keys must alternate within a template.")

        if len(template_parts) == 0:
            raise ValueError("Template cannot be empty.")

        return tuple(template_parts)

    @classmethod
    def load(cls, raw: str) -> "TextCompletionsMappingTemplate":
        """ Loads a raw text template """
        raw = raw.strip()
        return cls(
            raw=raw, parts=cls._parse_template_into_parts(raw),
            keys=cls._parse_template_keys(raw)
        )

@define(frozen=True)
class TextCompletionsMappingDefinition:
    """
    Guidelines, input/output templates, and example parameters that define
    a text completions mapping
    """

    guidelines: str
    input_template: TextCompletionsMappingTemplate
    output_template: TextCompletionsMappingTemplate
    examples: Tuple[TextMappingExample,...]

    source_dir_path: Optional[str] = None

    @classmethod
    def _load_examples_from_file(cls, file_path: str, input_keys: List[str], output_keys: List[str]) -> Tuple[TextMappingExample,...]:
        """ Load input/output examples from YAML or TSV """
        examples: List[TextMappingExample] = []
        for corpus_example in SchematizedTextCorpus.load_from_file(file_path):
            examples.append(TextMappingExample(
                inputs={
                    key: corpus_example[key] for key in input_keys
                },
                outputs={
                    key: corpus_example[key] for key in output_keys
                }
            ))
        return tuple(examples)

    @classmethod
    def load_from_directory(cls, dir_path: str) -> "TextCompletionsMapping":
        """
        Load a mapping from a directory containing guidelines.txt, input_template.txt,
        output_template.txt, and examples.tsv or examples.yaml.
        """
        with open(path.join(dir_path, "guidelines.txt"), "r") as fp:
            guidelines = fp.read().strip()

        with open(path.join(dir_path, "input_template.txt")) as fp:
            input_template = TextCompletionsMappingTemplate.load(fp.read())

        with open(path.join(dir_path, "output_template.txt")) as fp:
            output_template = TextCompletionsMappingTemplate.load(fp.read())

        if len(set(input_template.keys) & set(output_template.keys)) > 1:
            raise ValueError("Template key occurs in both input and output templates")

        if path.exists(path.join(dir_path, "examples.yaml")):
            examples = cls._load_examples_from_file(path.join(dir_path, "examples.yaml"), input_template.keys, output_template.keys)
        elif path.exists(path.join(dir_path, "examples.tsv")):
            examples = cls._load_examples_from_file(path.join(dir_path, "examples.tsv"), input_template.keys, output_template.keys)
        else:
            raise FileNotFoundError(
                f"TextCompletionsMapping requires examples.yaml or examples.tsv in given directory '{dir_path}'"
            )

        return cls(
            guidelines, input_template, output_template, examples,
            source_dir_path=dir_path
        )

@define(frozen=True)
class TextCompletionsMappingOutput:
    """ Output from a text completions mapping """

    raw: str
    parsed: Dict[str, str]
    success: bool
    error_message: Optional[str]


class TextCompletionsMapping(ABC):
    """ Text mapping based on a text completions client """

    def __init__(
        self, definition: TextCompletionsMappingDefinition
    ):
        """ Initialize a text mapping """
        self._definition = definition

    @property
    def definition(self) -> TextCompletionsMappingDefinition:
        """ Definition for this mapping """
        return self._definition

    def map_corpus(
        self,
        corpus: SchematizedTextCorpus,
        parse_error_default=None,
        include_inputs_in_output=True,
        include_debug_fields=False,
        error_message_field="Error Message",
        success_field="Completion Success",
        raw_completion_field="Completion"
    ) -> SchematizedTextCorpus:
        """ Map a corpus of text inputs to a new corpus of text outputs """
        extra_fields = { error_message_field, success_field, raw_completion_field }
        if include_debug_fields and (
            len(extra_fields & set(self.definition.input_template.keys)) > 0
            or len(extra_fields & set(self.definition.output_template.keys)) > 0
        ):
            raise ValueError(
                "Cannot map corpus because debug field included in input/output fields."
            )

        output_corpus = SchematizedTextCorpus(
            fields=(
                list(self.definition.output_template.keys)
                + [success_field, error_message_field, raw_completion_field]
            ) if include_debug_fields else list(self.definition.output_template.keys),
            examples=[
                (
                    SchematizedTextCorpusExample(output.parsed)
                    + SchematizedTextCorpusExample({
                        success_field: str(output.success),
                        error_message_field: str(output.error_message),
                        raw_completion_field: output.raw
                    })
                ) if include_debug_fields
                else SchematizedTextCorpusExample(output.parsed)
                for output in self.map_batch(corpus, parse_error_default=parse_error_default)
            ]
        )

        return corpus.join(output_corpus) if include_inputs_in_output else output_corpus

    def map_batch(self, inputs: Iterable[SchematizedTextCorpusExample], parse_error_default=None) -> List[TextCompletionsMappingOutput]:
        """
        Generate output dictionaries for the given collection of inputs
        """
        return [self.map(input, parse_error_default=parse_error_default) for input in inputs]

    @abstractmethod
    def map(
        self,
        input: Union[SchematizedTextCorpusExample, Dict[str, str]],
        parse_error_default=None
    ) -> TextCompletionsMappingOutput:
        """
        Generate output dictionary for thi given input example
        """
