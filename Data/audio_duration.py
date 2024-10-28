import os
from pydub import AudioSegment
import argparse
from collections import defaultdict
def get_total_audio_length(file_path):
	total_length = 0.0
	user_length = defaultdict(float)

	# Open the text file and read lines
	with open(file_path, 'r') as file:
		audio_paths = file.readlines()

	# Iterate through each audio file path
	for audio_path in audio_paths:

		audio_path = audio_path.strip()  # Remove any surrounding whitespace/newlines
		data = audio_path.split("|")
		audio_path = data[0]
		if os.path.exists(audio_path):  # Check if the audio file exists
			audio = AudioSegment.from_file(audio_path)
			total_length += len(audio) / 1000.0  # Convert length from milliseconds to seconds

		user_length[data[2]] += len(audio) / 1000.0

	return total_length, user_length

def main():
	# Example usage:
	parser = argparse.ArgumentParser(description="Calculate the total length of audio files listed in a text file.")
	parser.add_argument('--file', type=str, help='Path to the text file containing audio file paths')
	args = parser.parse_args()
	total_length, user_length = get_total_audio_length(args.file)
	print(user_length)
	print(f"Total length of all audio files: {total_length:.2f} seconds")
	print(f"Total length of all audio files: {total_length / 60 / 60:.2f} hrs")
	print(f"Total speaker: {len(user_length)}")
     
if __name__ == '__main__':
	main()	
