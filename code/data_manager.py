import os
import whisper
import time

class Video():
    def __init__(self, video_url, download_path='../data', download_name='video.mp4'):
        self.video_url = video_url
        self.download_path = download_path
        self.video_file = download_name
        self.audio_file = None
        self.muted_video_file = None
        self.text = None
        self.translated_audio_file = None
        self.translated_video_file = None
        
    def download(self):
        os.system('yt-dlp -f mp4 -P {} -o {} {}'.format(self.download_path, self.video_file, self.video_url))
        # wait for file to finish download
        video_path = os.path.join(self.download_path, self.video_file)
        n_wait = 0
        while not os.path.exists(video_path):
           time.sleep(1)
           n_wait += 1
           if n_wait > 100:
               raise RuntimeError('Video could not download in time')
        return 

    def extract_audio(self, audio_file='audio.mp3'):
        self.audio_file = audio_file
        full_video_path = os.path.join(self.download_path, self.video_file)
        full_audio_path = os.path.join(self.download_path, self.audio_file)
        os.system('ffmpeg -i {} -vn -q:a 0 -map a -y {}'.format(full_video_path, full_audio_path))
        return full_audio_path

    def extract_video(self, muted_video_file='video_no_audio.mp4'):
        self.muted_video_file = muted_video_file
        full_video_path = os.path.join(self.download_path, self.video_file)
        full_muted_video_path = os.path.join(self.download_path, self.muted_video_file)
        os.system('ffmpeg -i {} -c:v copy -an -y {}'.format(full_video_path, full_muted_video_path))
        return full_muted_video_path

    def extract_text(self, model):
        # assume model is OpenAI whisper model
        if not self.audio_file:
            raise RuntimeError('Cannot extract text before audio has been separated from video.')
        
        full_audio_path = os.path.join(self.download_path, self.audio_file)
        result = model.infer(full_audio_path)
        
        self.text = result
        return result

    def translate_text(self, model):
        if not self.text:
            raise RuntimeError('Cannot translate text before it has been extracted from video.')

        for segment in self.text:
            translated_text = model.infer(segment['text'])
            segment['translation'] = translated_text

        return self.text

    def translated_audio(self, model, translated_audio_file='translated_audio.mp3'):
        if not self.text[0]['translation']:
            raise RuntimeError('Cannot dub translated audio before original text has been translated.')

        self.translated_audio_file = translated_audio_file
        full_audio_path = os.path.join(self.download_path, self.translated_audio_file)
        translated_audio = model.infer(self.text, full_audio_path)
        
        return full_audio_path

    def combine_video_and_audio(self, translated_video_file='output.mp4'):
        if not self.translated_audio_file or not self.muted_video_file:
            raise RuntimeError('Must translate audio and extract silent video before combining translated audio with video.')

        self.translated_video_file = translated_video_file

        full_muted_video_path = os.path.join(self.download_path, self.muted_video_file)
        full_translated_audio_path = os.path.join(self.download_path, self.translated_audio_file)
        full_translated_video_path = os.path.join(self.download_path, self.translated_video_file)
        os.system('ffmpeg -i {} -i {} -c:v copy -c:a aac -strict experimental -y {}'.format(full_muted_video_path, full_translated_audio_path, full_translated_video_path))
        return full_translated_video_path
