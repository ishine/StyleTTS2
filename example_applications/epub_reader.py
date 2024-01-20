# requires ebooklib bs4 nltk tqdm
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import nltk
import numpy as np
from example_applications.core import ExampleApplicationsCore 

class EpubReader:
    def __init__(self, config_path, model_path, ref_audio_path):
        print("loading model")
        self.core = ExampleApplicationsCore()
        self.core.load_model(config_path, model_path)
        print("done loading model")
        self.set_style(ref_audio_path)

    def set_style(self, ref_audio_path):
        print(f"calculating style for {ref_audio_path}")
        self.style = self.core.style_from_path(ref_audio_path)
        print(f"done calculating style for {ref_audio_path}")

    def read_sentences(self, sentences,
        alpha=0.1, beta=0.9, t=0.7, diffusion_steps=5, embedding_scale=1):
        s_prev = None
        wavs = []

        for sentence in tqdm(sentences, desc="Inferring audio."):
            print(f"Inferring sentence {sentence}")
            wav, s_prev = self.core.LFinference(
                sentence, s_prev, self.style,
                alpha=alpha, beta=beta,
                t=t, diffusion_steps=diffusion_steps,
                embedding_scale=embedding_scale)
            wavs.append(wav)
        return np.concatenate(wavs)

    def read_passage(self, p, **kwargs):
        sentences = nltk.sent_tokenize(p)
        return self.read_sentences(sentences, **kwargs)

    def read_epub(self, path, **kwargs):
        assert(os.path.exists(path))
        book = epub.read_epub(path)
        chapters = []
        chapter_audios = []

        for i,chapter in enumerate(book.get_items_of_type(ebooklib.ITEM_DOCUMENT)):
            soup = BeautifulSoup(chapter.content, 'html.parser')
            text = soup.body.get_text(separator=' ').strip().replace('\n','.')
            if len(text) > 0:
                chapter_audios.append(self.read_passage(text, **kwargs))
        return chapter_audios
        
#er = EpubReader("Configs/config.yml", "Models/Omni1/checkpoint.pth", "ref_data/celestia.wav")
#chapters = er.read_epub(r"C:\Users\vul\Downloads\days-of-wasp-and-spider.epub")