from dataclasses import dataclass, field


@dataclass
class TrainingArguments:
    """Help string for this group of command-line arguments"""

    hf_data_dirs: list[str] = field(default_factory=list)
    pl_data_dir: str = "../datasets"
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
    label_name: str = "syllable_input_ids"
    encoder_type: str = "conformer"
