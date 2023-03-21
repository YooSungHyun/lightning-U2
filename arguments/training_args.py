from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingArguments:
    """Help string for this group of command-line arguments"""

    pl_data_dirs: Optional[List[str]] = field(default=None)
    cache_main_dir: str = "./cache"
    vocab_path: str = "../config/vocab.json"
    num_shards: int = 1
    num_proc: int = None
    per_device_train_batch_size: int = 1
    train_batch_drop_last: bool = False
    per_device_eval_batch_size: int = 1
    eval_batch_drop_last: bool = False
    output_dir: str = "../../models"
    model_config: str = "../config/config.json"
    learning_rate: float = 0.001
    warmup_ratio: float = 0.2
    final_div_factor: int = 1e4  # (max_lr/div_factor)*final_div_factor is final lr
    weight_decay: float = 0.0001
    val_on_cpu: bool = False
    seed: int = None
    local_rank: int = None
    div_factor: int = 25  # initial_lr = max_lr/div_factor
    deepspeed_config: str = "ds_config/zero2.json"
    group_by_length: bool = False
    input_name: str = "input_values"
    label_name: str = "labels"
    length_column_name: str = "input_values"
    encoder_type: str = "conformer"
