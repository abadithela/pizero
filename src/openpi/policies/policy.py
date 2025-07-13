from collections.abc import Sequence
import logging
import pathlib
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._emb_prefix = nnx_utils.module_jit(model.embed_prefix)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        batch_size = obs.pop("batch_size")
        if "seed" in obs:
            seed = obs.pop("seed")
            self._rng = jax.random.PRNGKey(seed)
            logging.info("New trial, policy rng set to: %s", self._rng)
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if batch_size is not None:
            # Make inputs["tokenized_prompt_mask"] and inputs["tokenized_prompt"] and inputs['image_mask'] from (d,) to (b, d)
            inputs["image_mask"] = jax.tree.map(lambda x: jnp.repeat(x[np.newaxis, ...], batch_size, axis=0), inputs["image_mask"])
            inputs["tokenized_prompt"] = jax.tree.map(lambda x: jnp.repeat(x[np.newaxis, ...], batch_size, axis=0), inputs["tokenized_prompt"])
            inputs["tokenized_prompt_mask"] = jax.tree.map(lambda x: jnp.repeat(x[np.newaxis, ...], batch_size, axis=0), inputs["tokenized_prompt_mask"])
        else:
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs),
        }

        # Unbatch and convert to np.ndarray.
        if batch_size is not None:
            outputs = jax.tree.map(lambda x: np.asarray(x), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        return self._output_transform(outputs)

    def get_final_inputs(self, obs: dict, get_tokens=True) -> dict:
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        tokens = None
        if get_tokens:
            tokens, _, _ = self._emb_prefix(_model.Observation.from_dict(inputs))
        return tokens

    def get_visual_tokens(self, obs: dict) -> dict:
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        inputs.tokenized_prompt = None
        tokens, _, _ = self._emb_prefix(_model.Observation.from_dict(inputs))
        return tokens

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
