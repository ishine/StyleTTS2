from phonemizer import phonemize
import librosa
import soundfile as sf
from tqdm import tqdm
import os
from itertools import islice

from g2pw import G2PWConverter
conv = G2PWConverter(style='pinyin', enable_non_tradional_chinese=True)

root_dir = "/home/epentibi/Documents/aishell"

do_resample = True

wavs = []
transcriptions = []
speakers = []

with open(root_dir + "/transcript/aishell_transcript_v0.8.txt") as file:
    lines = file.readlines()
    for line in tqdm(lines):

        try:
            #print(line.rstrip())
            args = line.split(' ')
            code = args[0]
            speaker = code[7: 11]
            transcript = ''.join(args[1:-1])

            subdir = "train"
            if not os.path.exists(os.path.join(root_dir, f"wav_un/S{speaker}/train")):
                if not os.path.exists(os.path.join(root_dir, f"wav_un/S{speaker}/test")):
                    subdir = "dev"
                else:
                    subdir = "test"

            original_file = f"wav_un/S{speaker}/{subdir}/S{speaker}/{code}.wav"
            resampled_file = f"wav_24/S{speaker}/{subdir}/S{speaker}/{code}.wav"

            duration = librosa.get_duration(filename=os.path.join(root_dir, original_file))

            if duration < 1.2 or len(transcript) < 4:
                continue

            if do_resample and not librosa.get_samplerate(os.path.join(root_dir, resampled_file)) == 24000:
                y, s = librosa.load(os.path.join(root_dir, original_file), sr=24000)
                sf.write(os.path.join(root_dir, resampled_file), y, 24000)

            transcriptions.append(transcript)
            wavs.append(resampled_file)
            speakers.append(speaker)

        except Exception as e:
            print(e)

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

phonemized_lines = []
phonemized_transcriptions = []

batches = list(divide_chunks(transcriptions, 1000))

for batch in tqdm(batches):
    #phonemized = phonemize(batch, language="cmn", backend='espeak')  # Phonemize all text in one go to avoid triggering the memory protections error
    phonemized = conv(batch)
    #print("phonemized", len(phonemized))
    phonemized_transcriptions.extend(phonemized)

generate_dict = True
char_dictionary = { }
if generate_dict:
    print("Generating dictionary")
    for phonemized_transcription in tqdm(phonemized_transcriptions):
        #chars = phonemized_transcription.split(" ")
        for char in phonemized_transcription:
            if len(char) > 0 and char not in char_dictionary:
                char_dictionary[char] = len(char_dictionary)

    with open('dictionary.txt', "w+") as f:
        for char in char_dictionary:
            f.write(f'"{char}",{char_dictionary[char]}\n')


print(len(phonemized_transcriptions))

#phonemize(transcript, language="cmn", backend='espeak')

for i in tqdm(range(len(wavs))):  # Build the expected train_list
    phonemized_lines.append(f'{wavs[i]}|{phonemized_transcriptions[i]}|{int(speakers[i])}\n')

train_lines = phonemized_lines[:int(len(phonemized_lines) * 0.9)]
val_lines = phonemized_lines[int(len(phonemized_lines) * 0.9):]

ood_lines = phonemized_lines[:int(len(phonemized_lines) * 0.05)]
with open('ood_text.txt', 'w+') as f: # Path for train_list.txt in the training data folder
    for line in ood_lines:
        f.write(line.split('|')[1])

with open('train_list.txt', 'w+') as f: # Path for train_list.txt in the training data folder
    for line in train_lines:
        f.write(line)

with open('val_list.txt', 'w+') as f:  # Path for val_list.txt in the training data folder
    for line in val_lines:
        f.write(line)