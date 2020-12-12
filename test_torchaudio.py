import torchaudio

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from torchaudio.datasets.librispeech import LIBRISPEECH

root_path = '/home/cristin/tesis/data/'
dataset = LIBRISPEECH(root=root_path, url='dev-clean')

# Each item of the dataset is a tuple of len 6
# It contains: the waveform, the sample rate, the utterance (transcription),
#   the speaker ID, the chapter ID and the utterance ID.
print(dataset[0]) 
print(len(dataset[0]))
wf_original = dataset[0][0]
wf = dataset[0][0].numpy()[0]
print("the waveform is: ", wf)
print("audio len: ", len(wf))

plt.plot(wf)
plt.show()

fbank = torchaudio.compliance.kaldi.fbank(wf_original)

print("The fbank feat is: ", fbank)
print(fbank.shape)
plt.figure()
plt.imshow(fbank.t().numpy(), cmap='gray')
plt.show()

