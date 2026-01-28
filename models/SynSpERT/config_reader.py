"""
Configuration reader for SynSpERT models.

This module maps a `model_type` string to concrete model/config/tokenizer
classes and provides a utility to read a model configuration from disk.

Inputs:
- `args` typically is an `argparse.Namespace` with attributes `model_type`
  and `config_path`.

Outputs:
- Returns a model `config` object loaded via `from_pretrained()`.
"""
from transformers import (BertConfig, BertTokenizer)
#from syn_models.syntax_bert import (SyntaxBertConfig, SyntaxBertModel)
from spert.models import SynSpERTConfig
from spert.models import SynSpERT
from typing import Any


MODEL_CLASSES = {
    'syn_spert': (SynSpERTConfig, SynSpERT, BertTokenizer) 
}


def read_config_file(args: Any) -> Any:
    """Read and return a model configuration object.

    Args:
        args: Argument namespace with attributes `model_type` and `config_path`.

    Returns:
        A model configuration object (library-specific type).
    """
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]   
    config = config_class.from_pretrained(args.config_path)
#                                          finetuning_task=args.task_name)
    return config
