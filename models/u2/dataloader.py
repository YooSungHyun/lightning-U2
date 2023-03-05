import torch
from torch.nn.utils.rnn import pad_sequence


class AudioDataLoader(torch.utils.data.DataLoader):
    def __init__(self, pad_token_id: int = 0, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.pad_token_id = pad_token_id

    def _collate_fn(self, batch):
        # batch : input_values: log_melspect, ["grapheme_labels"]["input_ids"]: tokenized labels
        # input_values shape: (seq, mel_cnt)
        input_audios = [s["input_values"] for s in batch]
        audio_lengths = torch.IntTensor([s["input_values"].size(0) for s in batch])
        targets = [torch.as_tensor(s["syllable_input_ids"], dtype=torch.int32) for s in batch]
        target_lengths = torch.IntTensor([len(s["syllable_input_ids"]) for s in batch])

        input_audios = pad_sequence(input_audios, batch_first=True, padding_value=self.pad_token_id)
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_token_id)

        return input_audios, audio_lengths, targets, target_lengths
