"""
Representations for corpora of schematized text examples.  Each
example consists of several named fields containing text string values.
"""

import csv
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import yaml

class SchematizedTextCorpusExample:
    """
    Schematized text corpus example consistsing of several
    named fields containing text strings
    """

    def __init__(self, field_values: Dict[str, str]):
        """ Initialize a corpus example """
        self._field_values = field_values

    def keys(self) -> Iterable[str]:
        """ Field keys for this example """
        return self._field_values.keys()

    def get(self, field: str, default: Optional[Any] = None) -> Optional[str]:
        """ Get the value for a given field"""
        if field not in self._field_values:
            return default
        return self._field_values[field]

    def __iter__(self) -> Iterator[str]:
        """ Iterate over fields of the example """
        return iter(self._field_values)

    def __len__(self) -> int:
        """ Number of fields in this example """
        return len(self._field_values)

    def __getitem__(self, field: str) -> str:
        """ Get the value for a given field """
        return self._field_values[field]

    def __add__(self, other: "SchematizedTextCorpusExample") -> "SchematizedTextCorpusExample":
        """
        Concatenate two examples under the assumption
        that they have distinct fields
        """
        return SchematizedTextCorpusExample(
            field_values=dict(self._field_values, **other._field_values)
        )

class SchematizedTextCorpus:
    """
    Corpus of examples where each example consists of several
    named fields containing strings
    """

    def __init__(self, fields: Iterable[str], examples: Iterable[SchematizedTextCorpusExample]):
        """ Initialize corpus """
        self._fields = tuple(fields)
        self._examples = tuple(examples)

    @property
    def fields(self) -> Tuple[str, ...]:
        """ Fields defining examples in this corpus """
        return self._fields

    def __iter__(self) -> Iterator[SchematizedTextCorpusExample]:
        """ Iterate over examples in this corpus """
        return iter(self._examples)

    def __len__(self) -> int:
        """ Number of examples in this corpus """
        return len(self._examples)

    def __getitem__(self, index: Union[int, slice]) -> str:
        """ Get the example at a given index """
        return self._examples[index]

    def join(self, other: "SchematizedTextCorpus") -> "SchematizedTextCorpus":
        """
        Joins to corpora, currently under the assumption that each
        contains the same number of rows and no fields in common.
        """
        if len(set(self.fields) & set(other.fields)) > 0:
            raise ValueError("Cannot join corpora that have fields in common.")
        if len(self) != len(other):
            raise ValueError("Joined corpora must contain the same number of rows.")

        return SchematizedTextCorpus(
            fields=self.fields + other.fields,
            examples=[
                example + other[index]
                for index, example in enumerate(self)
            ]
        )

    def _save_to_yaml(self, yaml_path: str):
        """ Save corpus to YAML """
        with open(yaml_path, 'w') as fp:
            # Use the safe_dump method to write the list of dictionaries to the file
            for example in self._examples:
                example_yaml = "\n".join(
                    f"  {field}: '{example[field]}'"
                    if "\n" not in example[field]
                    else f"  {field}: |2-\n    " + "\n    ".join(example[field].split("\n"))
                    for field in self.fields
                )
                fp.write(f"-\n{example_yaml}\n")

    def _save_to_tsv(self, tsv_path: str):
        """ Save corpus to TSV """
        with open(tsv_path, mode='w') as fp:
            # Create a DictWriter object specifying the delimiter as a tab
            writer = csv.DictWriter(fp, fieldnames=self.fields, delimiter='\t')
            writer.writeheader()
            writer.writerows(self._examples)

    def save(self, file_path: str) -> None:
        """ Save corpus to YAML or TSV file """
        if file_path.lower().endswith(".tsv"):
            self._save_to_tsv(file_path)
        elif file_path.lower().endswith(".yaml"):
            self._save_to_yaml(file_path)
        else:
            raise ValueError(f"File path must end in either .tsv or .yaml file extension.")

    @classmethod
    def _load_from_yaml(cls, yaml_path: str) -> "SchematizedTextCorpus":
        """ Load corpus from YAML """
        examples: List[SchematizedTextCorpusExample] = []
        with open(yaml_path, 'r') as file:
            yaml_rows = yaml.safe_load(file)
            fields = [field for field in yaml_rows[0]]
            for row in yaml_rows:
                examples.append(SchematizedTextCorpusExample(
                    field_values={
                        field: row[field] for field in fields
                    }
                ))
        return SchematizedTextCorpus(
            fields=fields,
            examples=examples
        )

    @classmethod
    def _load_from_tsv(cls, tsv_path: str) -> "SchematizedTextCorpus":
        """ Load corpus from TSV """
        examples: List[SchematizedTextCorpusExample] = []
        with open(tsv_path, 'r') as file:
            tsv_reader = csv.DictReader(file, delimiter='\t')
            fields = list(tsv_reader.fieldnames)
            for row in tsv_reader:
                examples.append(SchematizedTextCorpusExample(
                    field_values={
                        field: row[field] for field in fields
                    },
                ))
        return SchematizedTextCorpus(
            fields=fields,
            examples=examples
        )

    @classmethod
    def load_from_file(cls, file_path: str) -> "SchematizedTextCorpus":
        """ Loads corpus from either TSV or YAML file """
        if file_path.lower().endswith(".tsv"):
            return cls._load_from_tsv(file_path)
        elif file_path.lower().endswith(".yaml"):
            return cls._load_from_yaml(file_path)
        else:
            raise ValueError(f"File path must end in either .tsv or .yaml file extension.")
