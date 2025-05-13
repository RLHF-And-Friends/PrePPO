import typing as tp

from abc import ABC, abstractmethod

import logging

import numpy as np

from vllm import LLM, SamplingParams


class PairwiseArbiter(ABC):
    """
    Interface for all pairwise arbiters.
    """

    def judge(
        self,
        prompts: tp.List[str],
        completions: tp.List[tp.Tuple[str, str]],
        shuffle_order: bool = True
    ) -> tp.List:
        """
        Args:
            prompts:
                list of prompts.
            completions:
                list of tuple pairs with completions for given prompts.
            shuffle_order:
                whether to shuffle completions before passing them to the
                underlying model.
        Returns:
            list of [-1/0/1] where -1 denotes invalid result, 0 denotes win of
            0th completion and 1 denotes win of 1st completion in a tuple for
            each tuple in completions list.
        """
                # Shuffle the order of the completions to avoid positional bias
        # -----------------------------------------------------------------------------------------
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completions = [
                (pair[1], pair[0]) if flip else pair
                for flip, pair in zip(flip_mask, completions)
            ]

        # Get ranks for completions
        # -----------------------------------------------------------------------------------------

        ranks = self.get_ranks(prompts, completions)

        # Flip back the ranks to the original order if needed
        # -----------------------------------------------------------------------------------------
        if shuffle_order:
            ranks = [ranks[i] if not flip else 1 - ranks[i] for i, flip in enumerate(flip_mask)]

        return ranks
    
    @abstractmethod
    def get_ranks(
        self,
        prompts: tp.List[str],
        completions: tp.List[tp.Tuple[str, str]]
    ) -> tp.List[int]:
        """
        Should be implemented for each arbiter.
        """


class LocalVLLMArbiter(PairwiseArbiter):
    def __init__(
        self,
        system_prompt: str,
        model: str = "Qwen/Qwen3-32B",
        sampling_params: tp.Optional[SamplingParams] = None,
        **model_kwargs,
    ) -> None:
        self._system_prompt = system_prompt
        self._llm = LLM(model, **model_kwargs)
        self._sampling_params = sampling_params

    def get_ranks(self, prompts, completions):
        judge_messages = [
            [{
                "role": "user",
                "content": self._system_prompt.format(
                    text=prompt, response0=completion_a, response1=completion_b
                )
            }]
            for prompt, (completion_a, completion_b) in zip(prompts, completions)
        ]

        judge_responses = self._llm.chat(
            messages=judge_messages,
            sampling_params=self._sampling_params
        )

        potential_ranks = []
        for response in judge_responses:
            print(response.outputs[0].text)
            potential_ranks.append(response.outputs[0].text[-1])

        # potential_ranks = [response.outputs[0].text[-1] for response in judge_responses]
        
        ranks = []
        for potential_rank in potential_ranks:
            if potential_rank in ["0", "1"]:
                rank = int(potential_rank)
            else:
                # logging.warning(
                #     f"Invalid response from the judge model: '{potential_rank}'. Returning -1."
                # )
                rank = -1
            ranks.append(rank)
            
        return ranks
