from flask import Flask, render_template, request
from models import TranscribeModel, T5TranslateModel, GTranslateModel, TTSModel
from data_manager import Video
import os

app = Flask(__name__)

# @app.before_first_reqeust
def setup():
    transcribe_model = TranscribeModel()
    # translate_model = T5TranslateModel(size='t5-base')
    translate_model = GTranslateModel()
    tts_model = TTSModel()
    # delete this line later?
    return transcribe_model, translate_model, tts_model
    
# @app.route('/')
#def index():
#    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        output_video_path = process_video(user_input)
        videos_dir = './static/videos'
        final_path = os.path.join(videos_dir, 'output.mp4')
        os.system('mv {} {}'.format(output_video_path, final_path))
    return render_template('index.html', selected_video='output.mp4')

# @app.route('/submit', methods=['POST'])
def submit():
    video_url = request.form.get('user_input')
    video_url = 'https://www.youtube.com/watch?v=oqpfgUQET6A&ab_channel=HyprMX'
    final_video = process_video(video_url)
    return '{}'.format(video_url)

def process_video(video_url):
    video = Video(video_url)
    video_path = video.download()
    audio_path = video.extract_audio()
    muted_video_path = video.extract_video()
    transcribe_model, translate_model, tts_model = setup()
    text = video.extract_text(transcribe_model)
    translated_text = video.translate_text(translate_model)
    translated_audio_path = video.translated_audio(tts_model)
    combined_video_and_audio_path = video.combine_video_and_audio()
    return combined_video_and_audio_path
    
if __name__ == '__main__':
    app.run()
    #video_url = 'https://www.youtube.com/watch?v=oqpfgUQET6A&ab_channel=HyprMX'
    #process_video(video_url)
