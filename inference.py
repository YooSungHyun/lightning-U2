import pytorch_lightning as pl
from models.u2.lightningmodule import BiU2
import torch
from simple_parsing import ArgumentParser
import librosa
import numpy as np
from torchaudio.transforms import MelSpectrogram
from utils.comfy import dataclass_to_namespace
from utils.config_loader import load_config
from arguments.inference_args import InferenceArguments

WINDOWS = {
    "hamming": torch.hamming_window,
    "hann": torch.hann_window,
    "blackman": torch.blackman_window,
    "bartlett": torch.bartlett_window,
}


def main(hparams):
    model_config = load_config(hparams.model_config)
    lightning_module = BiU2.load_from_checkpoint(hparams.model_path, args=hparams)
    lightning_module.eval()
    sampling_rate = model_config.data.audio.log_mel_conf.sample_rate
    window_size_sec = model_config.data.audio.log_mel_conf.window_size_sec
    window_stride_sec = model_config.data.audio.log_mel_conf.window_stride_sec
    n_mels = model_config.data.audio.log_mel_conf.n_mels
    raw_audio, sr = librosa.load(hparams.wav_file, sr=sampling_rate)

    if model_config.data.audio.normalize:
        raw_audio = np.array((raw_audio - raw_audio.mean()) / np.sqrt(raw_audio.var() + 1e-7))
    # window_size는 통상 사람이 변화를 느끼는 한계인 25ms을 기본으로 합니다 (0.025)
    # 16000 * 0.025 = 400
    win_length = int(np.ceil(sampling_rate * window_size_sec))
    # n_fft는 학습에 쓰이기 위한 max_length로 보면 됩니다. 해당 길이만큼 pad가 붙고, 모델과 관련이 있다는 점에서
    # 2의 n승 혹은 512, 1024 등의 값을 주로 쓰는데, win_length보다 크거나 같으면 되므로 저는 같게 설정.
    n_fft = win_length
    # 얼마만큼 밀면서 자를것이냐, (얼마만큼 겹치게 할 것이냐) 1부터 숫자에서 win_length가 10 hop_length를 2로 하면, 1~10 -> 3~13 과 같은 형태로 밀림.
    hop_length = int(sampling_rate * window_stride_sec)

    raw_audio = torch.FloatTensor(raw_audio)

    # log_mel spec (channel(mono(1), 2~3 etc), n_mels, time)
    mel_spect = MelSpectrogram(
        sample_rate=sampling_rate, win_length=win_length, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )(raw_audio)
    log_melspect = torch.log1p(mel_spect)
    test_target = log_melspect.T
    input_audios = torch.stack([test_target], dim=0)
    # test_target = datasets[1]
    # input_audios = torch.stack([test_target["input_values"]], dim=0)
    input_lengths = torch.IntTensor([s.size(0) for s in input_audios])
    with torch.no_grad():
        if hparams.mode == "attention":
            hyps, _ = lightning_module.model.recognize(
                input_audios,
                input_lengths,
                beam_size=hparams.beam_size,
                decoding_chunk_size=hparams.decoding_chunk_size,
                num_decoding_left_chunks=hparams.num_decoding_left_chunks,
                simulate_streaming=hparams.simulate_streaming,
            )
            hyps = [hyp.tolist() for hyp in hyps]
        elif hparams.mode == "ctc_greedy_search":
            hyps, _ = lightning_module.model.ctc_greedy_search(
                input_audios,
                input_lengths,
                decoding_chunk_size=hparams.decoding_chunk_size,
                num_decoding_left_chunks=hparams.num_decoding_left_chunks,
                simulate_streaming=hparams.simulate_streaming,
            )
        # ctc_prefix_beam_search and attention_rescoring only return one
        # result in List[int], change it to List[List[int]] for compatible
        # with other batch decoding mode
        elif hparams.mode == "ctc_prefix_beam_search":
            assert input_audios.size(0) == 1
            hyp, _ = lightning_module.model.ctc_prefix_beam_search(
                input_audios,
                input_lengths,
                hparams.beam_size,
                decoding_chunk_size=hparams.decoding_chunk_size,
                num_decoding_left_chunks=hparams.num_decoding_left_chunks,
                simulate_streaming=hparams.simulate_streaming,
            )
            hyps = [hyp]
        elif hparams.mode == "attention_rescoring":
            assert input_audios.size(0) == 1
            hyp, _ = lightning_module.model.attention_rescoring(
                input_audios,
                input_lengths,
                hparams.beam_size,
                decoding_chunk_size=hparams.decoding_chunk_size,
                num_decoding_left_chunks=hparams.num_decoding_left_chunks,
                ctc_weight=0.3,
                simulate_streaming=hparams.simulate_streaming,
                reverse_weight=hparams.reverse_weight,
            )
            hyps = [hyp]
    print(lightning_module.tokenizer.batch_decode(hyps))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_arguments(InferenceArguments, dest="inference_args")
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "inference_args")
    main(args)
