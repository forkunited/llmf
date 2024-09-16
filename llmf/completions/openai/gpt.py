"""
OpenAI text completions (GPT) client
"""

from openai import OpenAI
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

from llmf.util import logging

from llmf.completions.base import TextCompletionsClient
from llmf.completions.openai.parameters import ChatGPTParameters, ChatGPTRole

_LOGGING_SOURCE = "ChatGPTTextCompletionsClient"
_LOGGING_COMPLETION_KEY = "Completion"

class ChatGPTPrompt:
    """ Prompt to ChatGPT """

    def __init__(self, messages: List[Tuple[ChatGPTRole, str]]):
        """ Initialize a prompt """
        self._messages = tuple(messages)

    @property
    def messages(self) -> Tuple[Tuple[ChatGPTRole, str]]:
        """ Messages of this prompt """
        return self._messages

    @property
    def character_length(self) -> int:
        """ Number of characters across all prompt messages """
        return sum(len(message[1]) for message in self._messages)

    def __getitem__(
        self, key: Union[int, slice]
    ) -> Union[Tuple[ChatGPTRole, str], "ChatGPTPrompt"]:
        """ Retrieve messages at the given key """
        if isinstance(key, int):
            return self._messages[key]
        else:
            return ChatGPTPrompt(self._messages[key])

    def __eq__(self, other: "ChatGPTPrompt") -> bool:
        """ Equality check for GPT prompts """
        return self._messages == other._messages

    def __hash__(self) -> int:
        """ Hash of this prompt """
        return hash(self._messages)

    def __len__(self) -> int:
        """ Number of messages in the prompt """
        return len(self._messages)

    def __iter__(self) -> Iterator[Tuple[ChatGPTRole, str]]:
        """ Iterator over messages in the prompt """
        return iter(self._messages)

    def __add__(
        self,
        other: Union[Tuple[ChatGPTRole, str], Iterable[Tuple[ChatGPTRole, str]]]
    ) -> "ChatGPTPrompt":
        """ Add messages to produce a new prompt """
        return ChatGPTPrompt(
            messages=list(self.messages) + (
                [other]
                if isinstance(other, tuple) and \
                    len(other) == 2 and isinstance(other[0], ChatGPTRole) \
                    else list(other)
            )
        )

    def __str__(self) -> str:
        """ String representation of this prompt """
        return "\n\n".join(
            f"{message[0].value.upper()}:\n{message[1]}" for message in self.messages
        )

    def to_api_format(self) -> List[Dict[str, str]]:
        """ Convert to format fed into GPT API """
        return [
            {"role": message[0].value, "content": message[1]}
            for message in self.messages
        ]

class ChatGPTTextCompletionsClient(TextCompletionsClient[ChatGPTPrompt]):
    """ Client for generating text completions from ChatGPT """

    def __init__(
        self,
        client: OpenAI,
        parameters: ChatGPTParameters
    ):
        """ Initialize client """
        self._client = client
        self._parameters = parameters

    @property
    def client(self) -> OpenAI:
        """ OpenAI client """
        return self._client

    @property
    def parameters(self) -> ChatGPTParameters:
        """ Client parameters """
        return self._parameters

    def run(self, prompt: ChatGPTPrompt) -> str:
        """
        Generate a completion for the given text
        """
        completion = self.client.chat.completions.create(
            model=self.parameters.model,
            messages=prompt.to_api_format(),
            frequency_penalty=self.parameters.frequency_penalty,
            logit_bias=self.parameters.logit_bias,
            logprobs=self.parameters.logprobs,
            top_logprobs=self.parameters.top_logprobs,
            max_tokens=self.parameters.max_tokens,
            n=self.parameters.n,
            presence_penalty=self.parameters.presence_penalty,
            response_format=self.parameters.response_format,
            seed=self.parameters.seed,
            stop=self.parameters.stop,
            stream=self.parameters.stream,
            temperature=self.parameters.temperature,
            top_p=self.parameters.top_p
        )

        # For now, just log in the mapping utility that wraps this completions thing
        # logging.info(
        #    _LOGGING_SOURCE,
        #    _LOGGING_COMPLETION_KEY,
        #    {
        #        "GPT Model": self.parameters.model,
        #        "Prompt Character Count": prompt.character_length,
        #        "Prompt Guidelines": f"{str(prompt[0][1][:64])}...",
        #        "Final Prompt Message": prompt[-1][1],
        #        "Completion": completion.choices[0].message.content,#
        #
        #    }
        #)
        
        return completion.choices[0].message.content

    @classmethod
    def load(
        cls,
        model: str,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[dict] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> "ChatGPTTextCompletionsClient":
        """
        Parameters:
            model: ID of the model to use. See the model endpoint compatibility
                table for details on which models work with the Chat API.
 
            frequency_penalty: Defaults to 0. Number between -2.0 and 2.0. Positive
                values penalize new tokens based on their existing frequency in the
                text so far, decreasing the model's likelihood to repeat the same
                line verbatim.

            logit_bias: Defaults to null
                Modify the likelihood of specified tokens appearing in the completion.
                Accepts a JSON object that maps tokens (specified by their token ID in
                the tokenizer) to an associated bias value from -100 to 100.
                Mathematically, the bias is added to the logits generated by the model
                prior to sampling. The exact effect will vary per model, but values
                between -1 and 1 should decrease or increase likelihood of selection;
                values like -100 or 100 should result in a ban or exclusive selection
                of the relevant token.

            logprobs: Defaults to false
                Whether to return log probabilities of the output tokens or not.
                If true, returns the log probabilities of each output token
                returned in the content of message. This option is currently
                not available on the gpt-4-vision-preview model.

            top_logprobs: An integer between 0 and 5 specifying the number of most likely
                tokens to return at each token position, each with an associated
                log probability. logprobs must be set to true if this parameter is used.

            max_tokens: The maximum number of tokens that can be generated in the chat completion.
                The total length of input tokens and generated tokens is limited by the
                model's context length. Example Python code for counting tokens.

            n: Defaults to 1
                How many chat completion choices to generate for each input
                message. Note that you will be charged based on the number of
                generated tokens across all of the choices. Keep n as 1 to
                minimize costs.

            presence_penalty: Defaults to 0
                Number between -2.0 and 2.0. Positive values penalize new tokens
                based on whether they appear in the text so far, increasing the
                model's likelihood to talk about new topics.

            response_format: An object specifying the format that the model must output.
                Compatible with GPT-4 Turbo and all GPT-3.5 Turbo models
                newer than gpt-3.5-turbo-1106. Setting to { "type": "json_object" }
                enables JSON mode, which guarantees the message the model generates
                is valid JSON.

                Important: when using JSON mode, you must also instruct the model
                to produce JSON yourself via a system or user message. Without this,
                the model may generate an unending stream of whitespace until the
                generation reaches the token limit, resulting in a long-running
                and seemingly "stuck" request. Also note that the message content
                may be partially cut off if finish_reason="length", which indicates
                the generation exceeded max_tokens or the conversation exceeded
                the max context length.

            seed: This feature is in Beta. If specified, our system will make a best
                effort to sample deterministically, such that repeated requests
                with the same seed and parameters should return the same result.
                Determinism is not guaranteed, and you should refer to the
                system_fingerprint response parameter to monitor changes in the backend.

            stop: Defaults to null
                Up to 4 sequences where the API will stop generating further tokens.

            stream: Defaults to false
                If set, partial message deltas will be sent, like in ChatGPT. Tokens
                will be sent as data-only server-sent events as they become available,
                with the stream terminated by a data: [DONE] message.

            temperature: Defaults to 1
                What sampling temperature to use, between 0 and 2. Higher values
                like 0.8 will make the output more random, while lower values like
                0.2 will make it more focused and deterministic.
                We generally recommend altering this or top_p but not both.

            top_p: Defaults to 1
                An alternative to sampling with temperature, called nucleus sampling,
                where the model considers the results of the tokens with top_p probability
                mass. So 0.1 means only the tokens comprising the top 10% probability
                mass are considered. We generally recommend altering this or temperature
                but not both.
        
        Returns:
            ChatGPT client with the given parameters
        """
        return cls(
            client = OpenAI(),
            parameters = ChatGPTParameters(
                model=model,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                max_tokens=max_tokens,
                n=n,
                presence_penalty=presence_penalty,
                response_format=response_format,
                seed=seed,
                stop=stop,
                stream=stream,
                temperature=temperature,
                top_p=top_p
          )
        )
