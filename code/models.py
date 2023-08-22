import torch
import whisper
from pydub import AudioSegment
from transformers import T5Tokenizer, T5ForConditionalGeneration
from elevenlabs import set_api_key, generate, play, save
from google.cloud import translate_v2 as translate
import io

class TextModel():
    def __init__(self):
        pass

    def infer(self, x):
        pass

class TranscribeModel(TextModel):
    def __init__(self):
        self.model = whisper.load_model("base")

    def infer(self, audio_file):
        audio = whisper.load_audio(audio_file)
        """
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        # detect the spoken language
        _, probs = self.model.detect_language(mel)
        # print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(self.model, mel, options)

        # print the recognized text
        # print(result.text)
        """
        # import pdb; pdb.set_trace()

        result = self.model.transcribe(audio)
        segments = result['segments']
        output = []
        for s in segments:
            output.append({'start': s['start'], 'text': s['text']})
        return segments

        
class T5TranslateModel(TextModel):
    def __init__(self, size='t5-small', from_language='English', to_language='Spanish'):
        self.tokenizer = T5Tokenizer.from_pretrained(size)
        self.model = T5ForConditionalGeneration.from_pretrained(size)
        self.from_language = from_language
        self.to_language = to_language

    def infer(self, text):
        text = 'translate {} to {}: '.format(self.from_language, self.to_language) + text 
        inputs = self.tokenizer(text, return_tensors='pt').input_ids
        outputs = self.model.generate(inputs, max_length=int(inputs.shape[1]*2), num_beams=3, early_stopping=True)
        translation = self.tokenizer.decode(outputs[0])

        return translation

class GTranslateModel(TextModel):
    def __init__(self, from_language='en', to_language='es'):
        self.client = translate.Client()
        self.from_language = from_language
        self.to_language = to_language

    def infer(self, text):
        translation = self.client.translate(text, self.to_language)
        return translation['translatedText']
    
class TTSModel(TextModel):
    def __init__(self, language='Spanish'):
        set_api_key("a16bb4afb7ed20626f8df63f853659f6")
        self.language = language
        
    def infer(self, text_segments, file_path):
        # Arguments:
        # text_segments: dictionary containing 'translation' and 'start' parameters
        # file_path: what file should this audio be written to?
        # Function: combines text from text_segments together, and outputs result to file
        # import pdb; pdb.set_trace()

        # find last sound so we know how long to make the audio buffer
        last_sound = 0
        for segment in text_segments:
            last_sound = max(last_sound, segment['end'])

        combined_audio = AudioSegment.silent(duration=int(last_sound*1000))
        for segment in text_segments:
            translation = segment['translation']
            start = segment['start']
            audio = generate(
                text=translation,
                voice="Arnold",
                model='eleven_multilingual_v1'
            )
            audio = io.BytesIO(audio)
            new_audio = AudioSegment.from_mp3(audio)
            combined_audio = combined_audio.overlay(new_audio, position=int(start*1000))

        combined_audio.export(file_path, format='mp3')
        
        return combined_audio

