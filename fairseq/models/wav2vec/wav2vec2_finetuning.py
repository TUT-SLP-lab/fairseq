import contextlib
import copy
import logging
import math
import re
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import II, MISSING, open_dict
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.wav2vec.wav2vec2_asr import Wav2Vec2AsrConfig,Wav2VecEncoder
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.tasks import FairseqTask

logger = logging.getLogger(__name__)

# refferd Wav2VecEncoder class in wav2vec2_asr.py
@register_model("w2v2_selftune", dataclass=Wav2Vec2AsrConfig)
class Wav2Vec2SelfTune(BaseFairseqModel):
    @classmethod
    def build_model(cls, cfg: Wav2Vec2AsrConfig, task: FairseqTask):
        """Build a new model instance."""
        return cls.build_w2v_model(cfg)

    @classmethod
    def build_w2v_model(cls,cfg: Wav2Vec2AsrConfig):
        # w2v読み込み
        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            w2v_args.criterion = None
            w2v_args.lr_scheduler = None
            cfg.w2v_args = w2v_args

            logger.info(w2v_args)

        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        model_normalized = w2v_args.task.get(
            "normalize", w2v_args.model.get("normalize", False)
        )
        assert cfg.normalize == model_normalized, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        if hasattr(cfg, "checkpoint_activations") and cfg.checkpoint_activations:
            with open_dict(w2v_args):
                w2v_args.model.checkpoint_activations = cfg.checkpoint_activations

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        # model.remove_pretraining_modules()

        if state is not None and not cfg.no_pretrained_weights:
            cls.load_model_weights(state, model, cfg)

        return model

    @classmethod
    def load_model_weights(cls,state, model, cfg):
        # 正味ここまでしなくても動く気がするけど前例踏襲．
        if cfg.ddp_backend == "fully_sharded":
            from fairseq.distributed import FullyShardedDataParallel

            for name, module in model.named_modules():
                if "encoder.layers" in name and len(name.split(".")) == 3:
                    # Only for layers, we do a special handling and load the weights one by one
                    # We dont load all weights together as that wont be memory efficient and may
                    # cause oom
                    new_dict = {
                        k.replace(name + ".", ""): v
                        for (k, v) in state["model"].items()
                        if name + "." in k
                    }
                    assert isinstance(module, FullyShardedDataParallel)
                    with module.summon_full_params():
                        module.load_state_dict(new_dict, strict=True)
                    module._reset_lazy_init()

            # Once layers are loaded, filter them out and load everything else.
            r = re.compile("encoder.layers.\d.")
            filtered_list = list(filter(r.match, state["model"].keys()))

            new_big_dict = {
                k: v for (k, v) in state["model"].items() if k not in filtered_list
            }

            model.load_state_dict(new_big_dict, strict=False)
        else:
            if "_ema" in state["model"]:
                del state["model"]["_ema"]
            model.load_state_dict(state["model"], strict=True)
