import torch
import whisper
from pydub import AudioSegment
from transformers import T5Tokenizer, T5ForConditionalGeneration
from elevenlabs import set_api_key, generate, play, save
from pyannote.audio import Pipeline
from google.cloud import translate_v2 as translate
import io

# base model with functions that all models should incorporate
class TextModel():
    def __init__(self):
        pass

    def infer(self, x):
        pass

# transcribe audio data using
# 1) Whisper to extract text and timestamps for their locations
# 2) Pyannote to extract speakers corresponding to text
class TranscribeModel(TextModel):
    def __init__(self):
        self.transcribe_model = whisper.load_model("base")
        self.diarization_model = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token='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')
        
    def infer(self, audio_file):
        audio = whisper.load_audio(audio_file)
        result = self.transcribe_model.transcribe(audio)
        segments = result['segments']
        diarization = self.diarization_model(audio_file)
        diarization_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_time = str(turn)[2:14]
            start_time = int(start_time[9:]) + 1000*int(start_time[6:8]) + 6000*int(start_time[3:5]) + 36000*int(start_time[:2])
            segment_info = {'start': start_time, 'speaker': speaker}
            diarization_segments.append(segment_info)

        # dumb way to label speakers to segments. More efficient is using binary search since
        # diarization_segments is already sorted by time
        for segment in segments:
            for dia_seg in diarization_segments:
                if abs(segment['start']*1000 - dia_seg['start']) < 1500:
                    segment['speaker'] = dia_seg['speaker']
            if 'speaker' not in segment:
                segment['speaker'] = 'SPEAKER_00'
                
        return segments


# model for translating English to other language using T5 transformer via HuggingFace
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

# model for translating between two languages using Google Translate Cloud API
class GTranslateModel(TextModel):
    def __init__(self, from_language='en', to_language='es'):
        self.client = translate.Client()
        self.from_language = from_language
        self.to_language = to_language

    def infer(self, text):
        translation = self.client.translate(text, self.to_language)
        return translation['translatedText']

# model for generating speech from text using ElevenLabs
class TTSModel(TextModel):
    def __init__(self, language='Spanish'):
        set_api_key("a16bb4afb7ed20626f8df63f853659f6")
        self.language = language
        self.speakers = {0: 'Arnold',
                         1: 'Antoni',
                         2: 'Rachel',
                         3: 'Charlotte',
                         4: 'Clyde',
                         5: 'Daniel'}
        
    def infer(self, text_segments, file_path):
        # Arguments:
        # text_segments: dictionary containing 'translation' and 'start' parameters
        # file_path: what file should this audio be written to?
        # Function: combines text from text_segments together, and outputs result to file
        # import pdb; pdb.set_trace()

        # find last sound so we know how long to make the audio buffer
        last_sound = 0
        speaker_to_voice = {}
        voice_id = 0
        for segment in text_segments:
            last_sound = max(last_sound, segment['end'])
            speaker = segment['speaker']
            if speaker not in speaker_to_voice:
                voice_id = voice_id % 6
                speaker_to_voice[speaker] = self.speakers[voice_id]
                voice_id += 1

        combined_audio = AudioSegment.silent(duration=int(last_sound*1000))
        for segment in text_segments:
            translation = segment['translation']
            start = segment['start']
            speaker = segment['speaker']
            audio = generate(
                text=translation,
                voice=speaker_to_voice[speaker],
                model='eleven_multilingual_v1'
            )
            audio = io.BytesIO(audio)
            new_audio = AudioSegment.from_mp3(audio)
            combined_audio = combined_audio.overlay(new_audio, position=int(start*1000))

        combined_audio.export(file_path, format='mp3')
        
        return combined_audio

