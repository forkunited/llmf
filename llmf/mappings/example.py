"""
Representations for mapping examples
"""

from typing import Dict

class TextMappingExample:
    """ Example input/outputs for a text mapping """

    def __init__(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str]
    ):
        """ Initialize example """
        self._inputs = inputs
        self._outputs = outputs

    @property
    def inputs(self) -> Dict[str, str]:
        """ Input fields of this example """
        return dict(self._inputs)

    @property
    def outputs(self) -> Dict[str, str]:
        """ Output fields of this example """
        return dict(self._outputs)

