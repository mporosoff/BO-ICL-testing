from typing import Dict, List, Any, Union
from pydantic import Field
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
import numpy as np
from functools import partial
from typing import *
from .llm_model import (
    get_llm,
    DiscreteDist,
    GaussDist,
    scale_distribution,
)
from .aqfxns import (
    probability_of_improvement,
    expected_improvement,
    log_expected_improvement,
    upper_confidence_bound,
    greedy,
)
from .pool import Pool
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.example_selectors import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)

import warnings


class QuantileTransformer:
    def __init__(self, values, n_quantiles):
        self.n_quantiles = n_quantiles
        self.quantiles = np.linspace(0, 1, n_quantiles + 1)
        self.values_quantiles = np.quantile(values, self.quantiles)

    def to_quantiles(self, values):
        quantile_scores = np.digitize(values, self.values_quantiles[1:-1])
        return quantile_scores

    def to_values(self, quantile_scores):
        values_from_scores = np.interp(
            quantile_scores, range(self.n_quantiles + 1), self.values_quantiles
        )
        return values_from_scores


class LabelSimilarityExampleSelector(SemanticSimilarityExampleSelector):
    examples: List[Dict] = Field(default_factory=list)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        # need to select examples with the most similar y from input_variables
        y = input_variables["y"]
        self.examples.sort(key=lambda ex: abs(float(y) - float(ex["y"])))
        return self.examples[: self.k]

    def add_example(self, example: Dict[str, str]) -> str:
        self.examples.append(example)

    @classmethod
    def from_examples(
        cls,
        examples: List[Dict],
        embeddings: Embeddings,
        vectorstore_cls: type[VectorStore],
        k: int = 5,
        input_keys: Union[List[str], None] = None,
        *,
        example_keys: Union[List[str], None] = None,
        vectorstore_kwargs: Union[Dict, None] = None,
        **vectorstore_cls_kwargs: Any,
    ):
        new_class = super().from_examples(
            examples,
            embeddings,
            vectorstore_cls,
            k,
            input_keys,
            example_keys=example_keys,
            vectorstore_kwargs=vectorstore_kwargs,
            **vectorstore_cls_kwargs,
        )
        new_class.examples = examples
        return new_class

    def __str__(self) -> str:
        return (
            f"LabelSimilarityExampleSelector(examples={len(self.examples)}, k={self.k})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class AskTellFewShot:
    def __init__(
        self,
        prompt_template: PromptTemplate = None,
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        model: str = "gpt-4o",
        inverse_model: Optional[str] = None,
        temperature: Optional[float] = None,
        inverse_temperature: Optional[float] = None,
        x_formatter: Callable[[str], str] = lambda x: x,
        y_formatter: Callable[[float], str] = lambda y: f"{y:0.2f}",
        y_name: str = "output",
        x_name: str = "input",
        selector_k: Optional[int] = 5,
        k: int = 5,
        use_quantiles: bool = False,
        n_quantiles: int = 100,
        verbose: bool = False,
        cos_sim: bool = True,
        use_logprobs: bool = False,
        embedding_model: str = "text-embedding-3-large",
        objective_bounds: Optional[Tuple[float, float]] = None,
        maximize: bool = True,
        reference_scale: Optional[float] = None,
        min_samples: int = 2,
    ) -> None:
        """Initialize Ask-Tell optimizer.

        You can pass formatters that will make your data more compatible with the model. Note that
        y as output form the model must be a float(can be parsed with ``float(y_str)``)

        Args:
            prompt_template: Prompt template that should take x and y (for few shot templates)
            suffix: Matching suffix for first part of prompt template - for actual completion.
            prefix: Prefix to add before all examples (e.g., some context for the model).
            model: OpenAI base model to use for training and inference.
            temperature: Temperature to use for prediction inference. If None, will use model default.
            inverse_temperature: Temperature to use for inverse-design inference. If None, uses temperature.
            x_formatter: Function to format x for prompting.
            y_formatter: Function to format y for prompting.
            y_name: Name of y variable in prompt template (e.g., density, value of function, etc.)
            x_name: Name of x variable in prompt template (e.g., input, x, etc.). Only appears in inverse prompt
            selector_k: What k to use when switching to selection mode. If None, will use all examples
            k: Number of examples to use for each prediction.
            verbose: Whether to print out debug information.
        """
        self._selector_k = selector_k
        if selector_k is not None and selector_k < 1:
            raise ValueError(
                "selector_k must be positive, or None for all observed examples"
            )
        self.embedding_model = embedding_model
        self.objective_bounds = objective_bounds
        self.maximize = maximize
        self.reference_scale = reference_scale
        self.min_samples = min_samples
        self._observed_x = set()
        self.last_prediction_records = []
        self._ready = False
        self._ys = []
        self.format_x = x_formatter
        self.format_y = y_formatter
        self._y_name = y_name
        self._x_name = x_name
        self._prompt_template = prompt_template
        self._prefix = prefix
        self._suffix = suffix
        self._model = model
        self._inverse_model = inverse_model or model
        self._example_count = 0
        self._temperature = temperature
        self._inverse_temperature = (
            temperature if inverse_temperature is None else inverse_temperature
        )
        self._k = k
        self.use_quantiles = use_quantiles
        self.n_quantiles = n_quantiles
        self._calibration_factor = None
        self._verbose = verbose
        self.tokens_used = 0
        self.cos_sim = cos_sim
        self.use_logprobs = use_logprobs
        self.llm = None
        self.inv_llm = None

    def _setup_llm(self, model: str, temperature: Optional[float] = None):
        raise NotImplementedError

    def _setup_inv_llm(self, model: str, temperature: Optional[float] = None):
        raise NotImplementedError

    def _setup_prompt(
        self,
        example: Dict,
        prompt_template: Optional[PromptTemplate] = None,
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> FewShotPromptTemplate:
        raise NotImplementedError

    def _setup_inverse_prompt(self, example: Dict):
        raise NotImplementedError

    def _tell(self, x: str, y: float, alt_ys: Optional[List[float]] = None) -> Dict:
        raise NotImplementedError

    def _predict(self, queries: List[str]) -> List[DiscreteDist]:
        raise NotImplementedError

    def _inv_predict(self, queries: List[str]) -> List[DiscreteDist]:
        raise NotImplementedError

    def _ask(
        self, possible_x: List[str], best: float, aq_fxn: Callable, k: int
    ) -> Tuple[List[str], List[float], List[float]]:
        raise NotImplementedError

    def _tell(self, x: str, y: float, alt_ys: Optional[List[float]] = None) -> Dict:
        raise NotImplementedError

    def set_calibration_factor(self, calibration_factor):
        if calibration_factor is not None and (
            not np.isfinite(calibration_factor) or calibration_factor < 0
        ):
            raise ValueError("Uncertainty multiplier must be finite and nonnegative")
        self._calibration_factor = calibration_factor

    def inv_predict(self, y: float, system_message: Optional[str] = "") -> str:
        """A rough inverse model"""
        if not self._ready:
            raise ValueError(
                "Must tell at least one example before inverse predicting."
            )

        query = self.inv_prompt.format(
            y=self.format_y(y), y_name=self._y_name, x_name=self._x_name
        )
        if self.inv_llm is None:
            self.inv_llm = self._setup_inv_llm(
                self._inverse_model, self._inverse_temperature
            )
        x, tokens = self._inv_predict(query, system_message=system_message)
        self.tokens_used += tokens

        return x[0]

    def predict(
        self, x: str, system_message: Optional[str] = ""
    ) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
        """Predict the probability distribution and values for a given x.

        Args:
            x: The x value(s) to predict.
        Returns:
            The probability distribution and values for the given x.
        """
        if not isinstance(x, list):
            x = [x]
        if not self._ready:
            # special zero-shot
            self.prompt = self._setup_prompt(
                None, self._prompt_template, self._suffix, self._prefix
            )
            self.inv_prompt = self._setup_inverse_prompt(None)
            self._ready = True

        if self.llm is None:
            self.llm = self._setup_llm(self._model, self._temperature)

        if self._selector_k is not None:
            self.prompt.example_selector.k = min(self._example_count, self._selector_k)

        queries = [
            self.prompt.format(
                x=self.format_x(x_i),
                y_name=self._y_name,
            )
            for x_i in x
        ]
        results, tokens = self._predict(queries, system_message=system_message)
        self.tokens_used += tokens

        # GaussDist comes back with no std when all LLM samples agreed on one
        # value. Default it to 0 so the calibration multiply below is well
        # defined; this is an honest "the LLM agreed with itself across
        # samples" reading instead of the prior substitution which used
        # np.std(self._ys) and produced a growing band tied to label spread
        # rather than to prediction confidence.
        for i, result in enumerate(results):
            if isinstance(result, GaussDist) and result.std() is None:
                results[i].set_std(0.0)

        if self._calibration_factor is not None:
            results = [
                scale_distribution(
                    result, self._calibration_factor, self.objective_bounds
                )
                for result in results
            ]

        # compute mean and standard deviation
        if len(x) == 1:
            return results[0]
        return results

    def tell(self, x: str, y: float, alt_ys: Optional[List[float]] = None) -> None:
        """Tell the optimizer about a new example."""
        example_dict, inv_example = self._tell(x, y, alt_ys)
        self._observed_x.add(self.format_x(x))
        # we want to have example
        # to initialize prompts, so send it
        if not self._ready:
            self.prompt = self._setup_prompt(
                example_dict, self._prompt_template, self._suffix, self._prefix
            )
            self.inv_prompt = self._setup_inverse_prompt(inv_example)
            self._ready = True
        else:
            # in else, so we don't add twice
            if self._selector_k is not None:
                self.prompt.example_selector.add_example(example_dict)
                self.inv_prompt.example_selector.add_example(inv_example)
            else:
                self.prompt.examples.append(example_dict)
                self.inv_prompt.examples.append(inv_example)
        self._example_count += 1

    def ask(
        self,
        possible_x: Union[Pool, List[str]],
        aq_fxn: str = "expected_improvement",
        k: int = 1,
        inv_filter: int = 16,
        aug_random_filter: int = 0,
        lambda_mult: float = 0.5,
        _lambda: float = 0.5,
        system_message: Optional[str] = "",
        inv_system_message: Optional[str] = "",
    ) -> Tuple[List[str], List[float], List[float]]:
        """Ask the optimizer for the next x to try.

            Args:
            possible_x: List of possible x values to choose from.
            aq_fxn: Acquisition function to use.
            k: Number of x values to return.
            inv_filter: Reduce pool size to this number with inverse model. If 0, not used
            aug_random_filter: Add this man y random examples to the pool to increase diversity after reducing pool with inverse model
            _lambda: Lambda value to use for UCB
            lambda_mult: control MMR diversity ,0-1 lower = more diverse
        Return:
            The selected x values, their acquisition function values, and the predicted y modes.
            Sorted by acquisition function value (descending)
        """
        possible_x = Pool(
            [x for x in possible_x if self.format_x(x) not in self._observed_x],
            self.format_x,
            embedding_model=self.embedding_model,
        )
        if len(possible_x) == 0:
            return [], [], []

        # if we have less than 2 examples, just return random
        if len(self._observed_x) < 2:
            init_pnt = possible_x.sample(min(k, len(possible_x)))
            return (
                init_pnt,
                [0] * len(init_pnt),
                [0] * len(init_pnt),
            )

        if aq_fxn == "probability_of_improvement":
            aq_fxn = partial(probability_of_improvement, maximize=self.maximize)
        elif aq_fxn == "expected_improvement":
            aq_fxn = partial(expected_improvement, maximize=self.maximize)
        elif aq_fxn == "log_expected_improvement":
            aq_fxn = log_expected_improvement
        elif aq_fxn == "upper_confidence_bound":
            aq_fxn = partial(upper_confidence_bound, _lambda=_lambda)
        elif aq_fxn == "greedy":
            aq_fxn = greedy
        elif aq_fxn == "random":
            return (
                possible_x.sample(k),
                [0] * k,
                [0] * k,
            )
        else:
            raise ValueError(f"Unknown acquisition function: {aq_fxn}")

        if len(self._ys) == 0:
            best = 0
        else:
            best = np.max(self._ys) if self.maximize else np.min(self._ys)

        if inv_filter + aug_random_filter < len(possible_x):
            possible_x_l = []
            if inv_filter:
                from .llm_engine import resolve_inverse_target

                self.last_target = resolve_inverse_target(
                    best,
                    maximize=self.maximize,
                    bounds=self.objective_bounds,
                    reference_scale=self.reference_scale,
                )
                approx_x = self.inv_predict(
                    self.last_target["resolved_target"],
                    system_message=inv_system_message,
                )
                possible_x_l.extend(
                    possible_x.approx_sample(
                        approx_x, inv_filter, lambda_mult=lambda_mult
                    )
                )

            if aug_random_filter:
                possible_x_l.extend(possible_x.sample(aug_random_filter))
        else:
            possible_x_l = list(possible_x)

        results = self._ask(
            possible_x_l, best, aq_fxn, k, system_message=system_message
        )
        return results


class AskTellFewShotTopk(AskTellFewShot):
    def _setup_llm(self, model: str, temperature: Optional[float] = None):
        # nucleus sampling seems to get more diversity
        return get_llm(
            n=self._k,
            temperature=0.7 if temperature is None else temperature,
            model_name=model,
            # stop=["\n", "###", "#", "##"],
            # logit_bias={
            #     "198": -100,  # new line,
            #     "628": -100,  # double new line,
            #     "50256": -100,  # endoftext
            # },
            max_tokens=256,
            use_logprobs=self.use_logprobs,
        )

    def _setup_inv_llm(self, model: str, temperature: Optional[float] = None):
        return get_llm(
            n=1,
            model_name=model,
            # stop=[
            #     self.prompt.suffix.split()[0],
            #     self.inv_prompt.suffix.split()[0],
            #     "\n",
            # ],
            max_tokens=576,
            temperature=0.7 if temperature is None else temperature,
        )

    def _setup_prompt(
        self,
        example: Dict,
        prompt_template: Optional[PromptTemplate] = None,
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> FewShotPromptTemplate:
        if prefix is None:
            prefix = (
                "The following are correctly answered questions. "
                "Each answer is numeric and ends with ###\n"
            )
        if prompt_template is None:
            prompt_template = PromptTemplate(
                input_variables=["x", "y", "y_name"],
                template="Q: Given {x}, what is {y_name}?\nA: {y}###\n\n",
            )
            if suffix is not None:
                raise ValueError(
                    "Cannot provide suffix if using default prompt template."
                )
            suffix = "Q: Given {x}. What is {y_name}?\nA: "
        elif suffix is None:
            raise ValueError("Must provide suffix if using custom prompt template.")
        # test out prompt
        if example is not None:
            prompt_template.format(**example)
            examples = [example]
        # TODO: make fake example text
        else:
            examples = []
        example_selector = None
        if self._selector_k is not None:
            if len(examples) == 0:
                raise ValueError("Cannot do zero-shot with selector")
            sim_selector = (
                SemanticSimilarityExampleSelector
                if self.cos_sim
                else MaxMarginalRelevanceExampleSelector
            )
            example_selector = sim_selector.from_examples(
                [example],
                OpenAIEmbeddings(model=self.embedding_model),
                FAISS,
                k=self._selector_k,
                input_keys=["x"],
            )
        return FewShotPromptTemplate(
            examples=examples if example_selector is None else None,
            example_prompt=prompt_template,
            example_selector=example_selector,
            suffix=suffix,
            prefix=prefix,
            input_variables=["x", "y_name"],
        )

    def _setup_inverse_prompt(self, example: Dict):
        prompt_template = PromptTemplate(
            input_variables=["x", "y", "y_name", "x_name"],
            template="If {y_name} is {y}, then {x_name} is @@@\n{x}###",
        )
        if example is not None:
            prompt_template.format(**example)
            examples = [example]
        else:
            examples = []
        example_selector = None
        if self._selector_k is not None:
            if len(examples) == 0:
                raise ValueError("Cannot do zero-shot with selector")

            sim_selector = (
                SemanticSimilarityExampleSelector
                if self.cos_sim
                else MaxMarginalRelevanceExampleSelector
            )  # LabelSimilarityExampleSelector
            example_selector = sim_selector.from_examples(
                [example],
                OpenAIEmbeddings(model=self.embedding_model),
                FAISS,
                k=self._selector_k,
            )
        return FewShotPromptTemplate(
            examples=examples if example_selector is None else None,
            example_prompt=prompt_template,
            example_selector=example_selector,
            suffix="If {y_name} is {y}, then {x_name} is @@@",
            input_variables=["y", "y_name", "x_name"],
        )

    def _predict(self, queries: List[str], system_message: str) -> List[DiscreteDist]:
        if not system_message:
            warnings.warn(
                "No system message provided for prediction. Using default. \nNot clearly specifying the task for the LLM usually decreases its performance considerably."
            )

        results, tokens = self.llm.predict(queries, system_message=system_message)
        return results, tokens

    def _inv_predict(
        self, queries: List[str], system_message: str
    ) -> List[DiscreteDist]:
        if not system_message:
            warnings.warn(
                "No system message provided for inverse prediction. Using default. \nNot clearly specifying the task for the LLM usually decreases its performance considerably."
            )

        x, tokens = self.inv_llm.predict(
            queries, inv_pred=True, system_message=system_message
        )

        return x, tokens

    def _tell(self, x: str, y: float, alt_ys: Optional[List[float]] = None) -> Dict:
        """Tell the optimizer about a new example."""

        if self.use_quantiles:
            self.qt = QuantileTransformer(
                values=self._ys + [y], n_quantiles=self.n_quantiles
            )
            y = self.qt.to_quantiles(y)

        if alt_ys is not None:
            raise ValueError("Alt ys not supported for topk.")
        example_dict = dict(
            x=self.format_x(x),
            y=self.format_y(y),
            y_name=self._y_name,
        )
        self._ys.append(y)
        inv_dict = dict(
            x=self.format_x(x),
            y=self.format_y(y),
            y_name=self._y_name,
            x_name=self._x_name,
        )
        return example_dict, inv_dict

    def _ask(
        self,
        possible_x: List[str],
        best: float,
        aq_fxn: Callable,
        k: int,
        system_message: str,
    ) -> Tuple[List[str], List[float], List[float]]:
        results = self.predict(possible_x, system_message=system_message)
        if not isinstance(results, list):
            results = [results]
        if len(results) != len(possible_x):
            raise ValueError("Prediction count does not match candidate count")
        records = []
        for rank, (candidate, dist) in enumerate(zip(possible_x, results)):
            samples = dist.raw_samples()
            valid = len(samples) >= self.min_samples and np.all(np.isfinite(samples))
            if self.objective_bounds is not None:
                lower, upper = self.objective_bounds
                valid = valid and all(
                    (lower is None or lower <= value)
                    and (upper is None or value <= upper)
                    for value in samples
                )
            records.append(
                dict(
                    candidate=candidate,
                    rank=rank,
                    distribution=dist,
                    status="scored" if valid else "insufficient_or_invalid_samples",
                    acquisition=float(aq_fxn(dist, best)) if valid else None,
                )
            )
        self.last_prediction_records = records
        selected = sorted(
            (r for r in records if r["status"] == "scored"),
            key=lambda r: (-r["acquisition"], r["rank"]),
        )[:k]
        return (
            [r["candidate"] for r in selected],
            [r["acquisition"] for r in selected],
            [r["distribution"].mean() for r in selected],
        )
