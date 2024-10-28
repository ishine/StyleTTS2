import pandas as pd
from hanziconv import HanziConv
from phonemizer import phonemize
import phonemizer
from pinyin_to_ipa import pinyin_to_ipa
import pinyin as chinese_to_pinyin
import string
import dragonmapper.hanzi
from pypinyin import pinyin, lazy_pinyin, Style
from pydub import AudioSegment

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
	# simplified_text = simplified_text.replace("，", ",").replace("、", ",").replace("。", ".").replace("？", "?").replace("！", "!").replace("：", ":").replace("；", ";")
	return simplified_text

def text_to_phonemes(text, global_phonemizer=None, language='cmn'):
	"""Convert Chinese text to phonemes using phonemizer."""
	# phonemes = phonemize(
	#     text,
	#     language=language,
	#     backend='espeak',
	#     strip=True,  # Removes extra spaces
	#     preserve_punctuation=True,
	#     with_stress=True,  # Keep stress marks if available
	#     language_switch='remove-flags'  # Handle multilingual text
	# )
	# phonemes =  ''.join(global_phonemizer.phonemize([text]))
	# phonemes = phonemes.strip()
	# pinyin = ''.join(global_phonemizer.phonemize([text]))
	# pinyin = chinese_to_pinyin.get(text, format="numerical", delimiter="#")
	# pinyin = pinyin.split("#")
	# phonemes = " ".join(["".join(list(pinyin_to_ipa(p)[0])) for p in pinyin])
	# phonemes = dragonmapper.hanzi.to_ipa(text)

	# phonemes = dragonmapper.transcriptions.pinyin_to_ipa(s)
	phonemes = lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
	phonemes = " ".join(phonemes)
	return phonemes


# Common_voice in not enough longer
# def process_tsv(input_tsvs, output_file):
# 	"""Process the TSV file and generate the output file in the specified format."""

# 	# global_phonemizer = phonemizer.backend.EspeakBackend(language='cmn', preserve_punctuation=True, with_stress=True, words_mismatch='ignore')
# 	with open(output_file, 'w', encoding='utf-8') as f_out:

# 		unique_clients = set()
# 		for input_tsv in input_tsvs:
# 			df = pd.read_csv(input_tsv, sep='\t')
# 			unique_client = df['client_id'].unique()
# 			unique_clients.update(unique_client)
			
# 		unique_clients = list(unique_clients)
# 		client_id_map = {client: idx + 3000 for idx, client in enumerate(unique_clients)}

# 		for input_tsv in input_tsvs:
# 			df = pd.read_csv(input_tsv, sep='\t')
			
# 			# Create a mapping from client_id (original) to a unique integer starting from 3000

# 			for _, row in df.iterrows():
# 				# Extract the necessary columns
# 				filename = row['path']
# 				sentence = row['sentence']
# 				filename = f"/workspace/backup/cv-corpus-18.0-2024-06-14/zh-TW/clips/{filename}"

# 				# Too slow
# 				# audio = AudioSegment.from_file(filename)
# 				# duration_in_seconds = len(audio) / 1000.0
# 				# print(duration_in_seconds)
# 				# if duration_in_seconds < 8:
# 				# 	continue
# 				if len(sentence) < 15:
# 					continue

# 				client_id = client_id_map[row['client_id']]
				
# 				# Convert Traditional Chinese to Simplified Chinese (if needed)
# 				simplified_sentence = traditional_to_simplified(sentence)
# 				# try:
# 				# Convert the sentence to phonemes
# 				phonemes = text_to_phonemes(simplified_sentence, global_phonemizer=None)
				
# 				# Write to output file in the format: filename.wav|phoneme|speaker
# 				print(f"{input_tsv}|{filename}|{phonemes}|{client_id}\n")
# 				f_out.write(f"{filename}|{phonemes}|{client_id}\n")
# 				# except:
# 				# 	continue

# # Example usage
# input_tsvs = ["/workspace/backup/cv-corpus-18.0-2024-06-14/zh-TW/train.tsv",
#               "/workspace/backup/cv-corpus-18.0-2024-06-14/zh-TW/dev.tsv",
# 			  "/workspace/backup/cv-corpus-18.0-2024-06-14/zh-TW/invalidated.tsv",
#               "/workspace/backup/cv-corpus-18.0-2024-06-14/zh-TW/other.tsv",
#               "/workspace/backup/cv-corpus-18.0-2024-06-14/zh-TW/validated.tsv"]  # Replace with your input TSV file path
# output_file = 'train_list_8s.txt'  # Output file path
# process_tsv(input_tsvs, output_file)


# # input_tsvs = ["/workspace/backup/cv-corpus-18.0-2024-06-14/zh-TW/train.tsv",
#             #   "/workspace/backup/cv-corpus-18.0-2024-06-14/zh-TW/dev.tsv"]  # Replace with your input TSV file path
# # output_file = 'train_list.txt'  # Output file path

# input_tsvs = ["/workspace/backup/cv-corpus-18.0-2024-06-14/zh-TW/test.tsv"]  # Replace with your input TSV file path
# output_file = 'val_list_8s.txt'  # Output file path
# process_tsv(input_tsvs, output_file)

from datasets import load_dataset, DatasetDict
import os
import soundfile as sf
from tqdm import tqdm

# Load the dataset
dataset = load_dataset("simon3000/starrail-voice")

print(dataset["train"][0])
print(dataset["train"][1])
print(dataset["train"][2])
print("dataset length", len(dataset["train"]))	
dataset = dataset.filter(lambda example: example['language'] == "Chinese(PRC)")
print("dataset length", len(dataset["train"]))
# dataset = dataset.filter(lambda example: len(example['transcription']) >= 12)
dataset = dataset.filter(lambda example: len(example['audio']['array']) / example['audio']['sampling_rate'] >= 1.5)
print("dataset length", len(dataset["train"]))
dataset = dataset.filter(lambda example: "{" not in example['transcription'])
print("dataset length", len(dataset["train"]))
dataset = dataset.filter(lambda example: "}" not in example['transcription'])
print("dataset length", len(dataset["train"]))
dataset = dataset.filter(lambda example: example['speaker'] != "" and example['speaker'] is not None)
print("dataset length", len(dataset["train"]))

print(dataset["train"][0])
print(dataset["train"][1])
print(dataset["train"][2])

unique_speakers = set(dataset['train']['speaker'])
unique_speakers = list(unique_speakers)

# Split the dataset into train and test sets (90% train, 10% test)
train_test_split = dataset['train'].train_test_split(test_size=0.1)
dataset = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

# Define directory to save audio files
output_dir = "/workspace/StyleTTS2/Data"
os.makedirs(output_dir, exist_ok=True)
os.makedirs("/workspace/backup/starrail", exist_ok=True)

# Prepare the metadata file
train_metadata_path = os.path.join(output_dir, "train_starrail_pinyin.txt")
test_metadata_path = os.path.join(output_dir, "test_starrail_pinyin.txt")

def save_audio_and_metadata(dataset_split, metadata_path):
	with open(metadata_path, 'w', encoding='utf-8') as f:
		for example in tqdm(dataset_split):
			# Save audio file
			audio = example['audio']['array']
			sampling_rate = example['audio']['sampling_rate']
			audio_file_path = os.path.join("/workspace/backup/starrail", example['audio']['path'])
			if not os.path.exists(audio_file_path):
				sf.write(audio_file_path, audio, sampling_rate)
			try:
				# Write metadata line
				transcription = example['transcription']
				transcription = text_to_phonemes(transcription)
		
				speaker = example['speaker']
				speaker_id = unique_speakers.index(speaker) + 3000

				f.write(f"{audio_file_path}|{transcription}|{speaker_id}\n")
			except: 
				print(example['transcription'])


save_audio_and_metadata(dataset['train'], train_metadata_path)
save_audio_and_metadata(dataset['test'], test_metadata_path)

