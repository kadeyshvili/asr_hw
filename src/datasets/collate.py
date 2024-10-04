import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result_batch = {}
    audio = []
    spectrogram = []
    text = []
    text_encoded = []
    audio_path = []
    spectrogram_length = []
    text_encoded_length = []
    for item in dataset_items:
        audio.append(item['audio'].squeeze(0))
        spectrogram_length.append(item['spectrogram'].shape[2])
        text_encoded_length.append(item['text_encoded'].shape[1])
        spectrogram.append(item['spectrogram'].squeeze(0).permute(1, 0))
        text.append(item['text'])
        text_encoded.append(item['text_encoded'].squeeze(0))
        audio_path.append(item['audio_path'])

    result_batch['audio'] = pad_sequence(audio, batch_first = True)
    result_batch['spectrogram'] = pad_sequence(spectrogram, batch_first = True).permute(0, 2, 1)
    result_batch['text'] = text
    result_batch['text_encoded'] = pad_sequence(text_encoded, batch_first = True)
    result_batch['audio_path'] = audio_path
    result_batch['spectrogram_length'] = torch.tensor(spectrogram_length)
    result_batch['text_encoded_length'] = torch.tensor(text_encoded_length)
    return result_batch