import re
import string
import os
import random

import soundfile as sf
from datasets import DatasetDict, concatenate_datasets, load_dataset
from tqdm import tqdm
import dragonmapper.hanzi
import pandas as pd
import phonemizer
import pinyin as chinese_to_pinyin
from hanziconv import HanziConv
from phonemizer import phonemize
from pinyin_to_ipa import pinyin_to_ipa
from pydub import AudioSegment
from pypinyin import Style, lazy_pinyin, pinyin

from utils import fix_number, split_by_language


def remove_punctuation(text):
    # Define English punctuation and Chinese punctuation
    english_punctuation = string.punctuation
    chinese_punctuation = '。，！？、；：“”‘’（）《》〈〉【】〔〕'

    # Combine English and Chinese punctuation into one string
    all_punctuation = english_punctuation + chinese_punctuation

    # Create translation table for removing punctuation
    translation_table = str.maketrans('', '', all_punctuation)

    # Remove punctuation from the text
    return text.translate(translation_table)

def traditional_to_simplified(traditional_text):
	"""Convert Traditional Chinese to Simplified Chinese."""
	simplified_text = HanziConv.toSimplified(traditional_text)
	# simplified_text = remove_punctuation(simplified_text)
	return simplified_text

def text_to_phonemes(text, global_phonemizer):
	"""Convert Chinese text to phonemes using phonemizer."""
	
	# Rule-based
	text = text.replace(".", "點")

	segments = split_by_language(text)
	phonemes = []

	for segment in segments:
		if re.match(r'[a-zA-Z]', segment): 
			phonemes += global_phonemizer.phonemize([segment])
		elif re.match(r'[\u4e00-\u9fff]', segment):
			phonemes += lazy_pinyin(segment, style=Style.TONE3, neutral_tone_with_five=True, errors=fix_number)
		elif re.match(r'\d', segment):
			phonemes += fix_number(segment)
		else: 
			phonemes += segment

	phonemes = " ".join(phonemes)
	return phonemes



# Load the dataset
dataset_starrail = load_dataset("simon3000/starrail-voice")['train']
dataset_genshin = load_dataset("simon3000/genshin-voice")['train']
# dataset_cv_1 = load_dataset("mozilla-foundation/common_voice_16_0", "zh-TW", trust_remote_code=True)
dataset_gen_ai = load_dataset("voidful/gen_ai_2024")['train']
dataset_ml = load_dataset("Evan-Lin/snr-ml2021-hungyi-corpus")['test']

print("dataset_gen_ai length", len(dataset_gen_ai))
dataset_gen_ai = dataset_gen_ai.filter(lambda example: len(example['audio']['array']) / example['audio']['sampling_rate'] >= 1.5)
print("dataset_gen_ai length", len(dataset_gen_ai))
# dataset_gen_ai = dataset_gen_ai.filter(lambda example: len(example['text']) <= 60)
# print("dataset_gen_ai length", len(dataset_gen_ai))

print("dataset_ml length", len(dataset_ml))
dataset_ml = dataset_ml.filter(lambda example: len(example['audio']['array']) / example['audio']['sampling_rate'] >= 1.5)
print("dataset_ml length", len(dataset_ml))

# dataset_cv = concatenate_datasets([dataset_cv_1['train'], dataset_cv_1['validation']])
# dataset_cv = concatenate_datasets([dataset_cv, dataset_cv_1['test']])
# dataset_cv = concatenate_datasets([dataset_cv, dataset_cv_1['other']])
# dataset_cv = concatenate_datasets([dataset_cv, dataset_cv_1['invalidated']])

print("dataset_starrail length", len(dataset_starrail))	
dataset_starrail = dataset_starrail.filter(lambda example: example['language'] == "Chinese(PRC)")
print("dataset_starrail length", len(dataset_starrail))

dataset_starrail = dataset_starrail.filter(lambda example: len(example['audio']['array']) / example['audio']['sampling_rate'] >= 1.5)
print("dataset_starrail length", len(dataset_starrail))
dataset_starrail = dataset_starrail.filter(lambda example: "{" not in example['transcription'])
print("dataset_starrail length", len(dataset_starrail))
dataset_starrail = dataset_starrail.filter(lambda example: "}" not in example['transcription'])
print("dataset_starrail length", len(dataset_starrail))
dataset_starrail = dataset_starrail.filter(lambda example: example['speaker'] != "" and example['speaker'] is not None)
print("dataset_starrail length", len(dataset_starrail))
dataset_starrail = dataset_starrail.filter(lambda example: "<" not in example['transcription'])
print("dataset_starrail length", len(dataset_starrail))
dataset_starrail = dataset_starrail.filter(lambda example: ">" not in example['transcription'])
print("dataset_starrail length", len(dataset_starrail))
dataset_starrail = dataset_starrail.filter(lambda example: len(example['transcription']) >= 5)
print("dataset_starrail length", len(dataset_starrail))

print("dataset_genshin length", len(dataset_genshin))	
dataset_genshin = dataset_genshin.filter(lambda example: example['language'] == "Chinese")
print("dataset_genshin length", len(dataset_genshin))
# dataset_genshin = dataset_genshin.filter(lambda example: len(example['transcription']) >= 12)
dataset_genshin = dataset_genshin.filter(lambda example: len(example['audio']['array']) / example['audio']['sampling_rate'] >= 1.5)
print("dataset_genshin length", len(dataset_genshin))
dataset_genshin = dataset_genshin.filter(lambda example: "{" not in example['transcription'])
print("dataset_genshin length", len(dataset_genshin))
dataset_genshin = dataset_genshin.filter(lambda example: "}" not in example['transcription'])
print("dataset_genshin length", len(dataset_genshin))
dataset_genshin = dataset_genshin.filter(lambda example: example['speaker'] != "" and example['speaker'] is not None)
print("dataset_genshin length", len(dataset_genshin))
dataset_genshin = dataset_genshin.filter(lambda example: "<" not in example['transcription'])
print("dataset_genshin length", len(dataset_genshin))
dataset_genshin = dataset_genshin.filter(lambda example: ">" not in example['transcription'])
print("dataset_genshin length", len(dataset_genshin))
dataset_genshin = dataset_genshin.filter(lambda example: len(example['transcription']) >= 5)
print("dataset_genshin length", len(dataset_genshin))

# dataset_cv = dataset_cv.filter(lambda example: len(example['audio']['array']) / example['audio']['sampling_rate'] >= 1.5)
# print("dataset_cv length", len(dataset_cv))

common_voice_train_tsvs = ["/workspace/backup/cv-corpus-18.0-2024-06-14/zh-TW/train.tsv",
              "/workspace/backup/cv-corpus-18.0-2024-06-14/zh-TW/dev.tsv",
			  "/workspace/backup/cv-corpus-18.0-2024-06-14/zh-TW/invalidated.tsv",
              "/workspace/backup/cv-corpus-18.0-2024-06-14/zh-TW/other.tsv",
              "/workspace/backup/cv-corpus-18.0-2024-06-14/zh-TW/validated.tsv"]  
common_voice_test_tsvs = ["/workspace/backup/cv-corpus-18.0-2024-06-14/zh-TW/test.tsv"] 
input_tsvs = common_voice_train_tsvs + common_voice_test_tsvs

unique_speakers = set()
unique_speakers.update(dataset_starrail['speaker'])
unique_speakers.update(dataset_genshin['speaker'])
# unique_speakers.update(set(dataset_cv['client_id']))
unique_speakers.update({"hy"})

for input_tsv in input_tsvs:
	df = pd.read_csv(input_tsv, sep='\t')
	unique_client = df['client_id'].unique()
	unique_speakers.update(unique_client)

unique_speakers = list(unique_speakers)

# Split the dataset into train and test sets (90% train, 10% test)
train_test_split = dataset_starrail.train_test_split(test_size=0.05)
dataset_starrail = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

train_test_split = dataset_genshin.train_test_split(test_size=0.05)
dataset_genshin = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

# train_test_split = dataset_cv.train_test_split(test_size=0.1)
# dataset_cv = DatasetDict({
#     'train': train_test_split['train'],
#     'test': train_test_split['test']
# })


train_test_split = dataset_gen_ai.train_test_split(test_size=0.05)
dataset_gen_ai = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})


train_test_split = dataset_ml.train_test_split(test_size=0.05)
dataset_ml = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

# Define directory to save audio files
output_dir = "/workspace/StyleTTS2/Data"
os.makedirs(output_dir, exist_ok=True)
os.makedirs("/workspace/backup/starrail", exist_ok=True)
os.makedirs("/workspace/backup/genshin", exist_ok=True)
os.makedirs("/workspace/backup/common-voice", exist_ok=True)
os.makedirs("/workspace/backup/gen_ai", exist_ok=True)
os.makedirs("/workspace/backup/ml", exist_ok=True)
# Prepare the metadata file
train_metadata_path = os.path.join(output_dir, "train_all_pinyin.txt")
test_metadata_path = os.path.join(output_dir, "test_all_pinyin.txt")
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True)
def save_audio_and_metadata(metadata, dataset_split, dir_path, transcription_column, speaker_column):
	#
	for example in tqdm(dataset_split):
		# Save audio file
		audio = example['audio']['array']
		sampling_rate = example['audio']['sampling_rate']
		audio_file_path = os.path.join(f"/workspace/backup/{dir_path}", example['audio']['path'])
		if not os.path.exists(audio_file_path):
			sf.write(audio_file_path, audio, sampling_rate)
		# try:
		# Write metadata line
		transcription = example[transcription_column]
		transcription = traditional_to_simplified(transcription)
		phonemes = text_to_phonemes(transcription, global_phonemizer)

		if speaker_column == "hy":
			speaker = "hy"
		else: 
			speaker = example[speaker_column]
		speaker_id = unique_speakers.index(speaker) + 3000

		if len(phonemes) > 500:
			print("skip", phonemes)
			continue

		# f.write(f"{audio_file_path}|{transcription}|{speaker_id}\n")
		metadata.append(f"{audio_file_path}|{phonemes}|{speaker_id}\n")
		# except: 
		# 	print(example['transcription'])


metadata_train = []
metadata_test = []

def process_common_voice(meta_data, input_tsvs):

	for input_tsv in input_tsvs:
		df = pd.read_csv(input_tsv, sep='\t')

		for _, row in df.iterrows():
			filename = row['path']
			sentence = row['sentence']

			# if "other" in input_tsv or "validated" in input_tsv:
			# 	if len(sentence) < 15:
			# 		continue

			client_id = unique_speakers.index(row['client_id'])
			
			simplified_sentence = traditional_to_simplified(sentence)
			phonemes = text_to_phonemes(simplified_sentence, global_phonemizer)
			
			meta_data.append(f"/workspace/backup/cv-corpus-18.0-2024-06-14/zh-TW/clips/{filename}|{phonemes}|{client_id}\n")

process_common_voice(metadata_train, common_voice_train_tsvs)
process_common_voice(metadata_test, common_voice_test_tsvs)


save_audio_and_metadata(metadata_train, dataset_starrail['train'], "starrail", "transcription", "speaker")
save_audio_and_metadata(metadata_test, dataset_starrail['test'], "starrail", "transcription", "speaker")

save_audio_and_metadata(metadata_train, dataset_genshin['train'], "genshin", "transcription", "speaker")
save_audio_and_metadata(metadata_test, dataset_genshin['test'], "genshin", "transcription", "speaker")

# save_audio_and_metadata(metadata_train, dataset_cv['train'], "common_voice", "sentence", "client_id")
# save_audio_and_metadata(metadata_test, dataset_cv['test'], "common_voice", "sentence", "client_id")

save_audio_and_metadata(metadata_train, dataset_gen_ai['train'], "gen_ai", "text", "hy")
save_audio_and_metadata(metadata_test, dataset_gen_ai['test'], "gen_ai", "text", "hy")

save_audio_and_metadata(metadata_train, dataset_ml['train'], "ml", "transcription", "hy")
save_audio_and_metadata(metadata_test, dataset_ml['test'], "ml", "transcription", "hy")



random.seed(531)
random.shuffle(metadata_train)
random.shuffle(metadata_test)

with open(train_metadata_path, 'w', encoding='utf-8') as f:
	for line in metadata_train:
		f.write(line)

with open(test_metadata_path, 'w', encoding='utf-8') as f:
	for line in metadata_test:
		f.write(line)
