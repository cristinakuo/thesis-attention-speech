import torchaudio
from torchaudio.datasets.librispeech import LIBRISPEECH

from models import ASR

MAX_SAMPLES = 2
feature_dim = 40

root_path = '/home/cristin/tesis/data/'
dataset = LIBRISPEECH(root=root_path, url='dev-clean')

data_feats = []
# Get features
for i in range(MAX_SAMPLES):
    sample = dataset[i][0] # waveform
    print("prev shape: ", sample.shape)
    data_feat = torchaudio.compliance.kaldi.fbank(sample, num_mel_bins=feature_dim)
    data_feat = data_feat.unsqueeze(0)
    data_feats.append(data_feat)
    # Shape of feature is [Batch x SeqLen x FeatureDim]
    print("Hello, just extracted a feature!", data_feat.shape)
    print(data_feat)


print("Donee")
print(data_feats)


out_dim = 5 # TODO: use vocab size
model = ASR(feature_dim, out_dim)

# Go through the model
for sample in data_feats:
    out, _ = model(sample)


