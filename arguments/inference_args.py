from dataclasses import dataclass, field


@dataclass
class InferenceArguments:
    """Help string for this group of command-line arguments"""

    pl_data_dir: list[str] = field(default_factory=list)
    vocab_path: str = "../config/vocab.json"
    model_path: str = "../model/model.ckpt"
    model_config: str = "../config/config.json"
    beam_size: int = 10
    decoding_chunk_size: int = 16
    num_decoding_left_chunks: int = 1
    simulate_streaming: bool = True
    reverse_weight: float = 0.5
    val_on_cpu: bool = False
