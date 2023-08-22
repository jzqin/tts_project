from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    video_url = request.form.get('user_input')
    return '{}'.format(video_url)

if __name__ == '__main__':
    app.run()
