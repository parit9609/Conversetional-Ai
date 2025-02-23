import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file, send_from_directory, flash
from google.cloud import speech
from google.cloud import texttospeech_v1
from google.cloud import language_v2 

app = Flask(__name__)

# Configure upload and TTS folders
UPLOAD_FOLDER = 'uploads'
TTS_FOLDER = 'tts'
ALLOWED_EXTENSIONS = {'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TTS_FOLDER'] = TTS_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TTS_FOLDER, exist_ok=True)

# Initialize Google Cloud Clients
speech_client = speech.SpeechClient()
tts_client = texttospeech_v1.TextToSpeechClient()
language_client = language_v2.LanguageServiceClient()  # Sentiment Analysis Client

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_files(folder):
    """Get a list of files in a folder, sorted in reverse order."""
    return sorted(os.listdir(folder), reverse=True)

def sample_recognize(content):
   
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        language_code="en-UK",
        model="latest_long",
        audio_channel_count=1,
        enable_word_confidence=True,
        enable_word_time_offsets=True,
    )
    operation = speech_client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=90)

    txt = ''
    for result in response.results:
        txt += result.alternatives[0].transcript + '\n'

    return txt

def sample_synthesize_speech(text=None):
    
    input_data = texttospeech_v1.SynthesisInput(text=text)

    voice = texttospeech_v1.VoiceSelectionParams(
        language_code="en-UK"
    )

    audio_config = texttospeech_v1.AudioConfig(
        audio_encoding="LINEAR16"
    )

    request = texttospeech_v1.SynthesizeSpeechRequest(
        input=input_data,
        voice=voice,
        audio_config=audio_config,
    )

    response = tts_client.synthesize_speech(request=request)
    return response.audio_content

def sample_analyze_sentiment(text_content: str):
   
    document = language_v2.Document(
        content=text_content, type_=language_v2.Document.Type.PLAIN_TEXT
    )

    response = language_client.analyze_sentiment(request={"document": document})
    sentiment_score = response.document_sentiment.score

    if sentiment_score > 0.75:
        sentiment = "Positive"
    elif sentiment_score < -0.75:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment

# Routes
@app.route('/')
def index():
    audio_files = get_files(app.config['UPLOAD_FOLDER'])
    tts_files = get_files(app.config['TTS_FOLDER'])
    return render_template('index.html', audio_files=audio_files, tts_files=tts_files)

@app.route('/upload', methods=['POST'])
def upload_audio():
    """Handle uploading and processing of recorded audio."""
    if 'audio_data' not in request.files:
        flash('No audio data provided')
        return redirect(request.url)
    
    file = request.files['audio_data']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  + '.wav'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform speech-to-text
        with open(file_path, 'rb') as audio_file:
            audio_content = audio_file.read()

        transcript = sample_recognize(audio_content)

        # Perform sentiment analysis
        sentiment = sample_analyze_sentiment(transcript)

        # Save the transcript along with sentiment result
        transcript_filename = file_path.replace('.wav', '.txt')
        with open(transcript_filename, 'w') as transcript_file:
            transcript_file.write(f"Transcription:\n{transcript}\n\nSentiment: {sentiment}")

    return redirect('/')

@app.route('/upload_text', methods=['POST'])
def upload_text():
    """Handle text input, analyze sentiment, and generate TTS audio."""
    text = request.form['text']
    if text:
        # Perform sentiment analysis
        sentiment = sample_analyze_sentiment(text)

        # Generate unique filenames
        audio_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  + '.wav'
        text_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  + '.txt'

        # Generate TTS audio
        audio_content = sample_synthesize_speech(text=text)

        # Save the audio file
        audio_path = os.path.join(app.config['TTS_FOLDER'], audio_filename)
        with open(audio_path, 'wb') as audio_file:
            audio_file.write(audio_content)

        # Save the text file with sentiment result
        text_path = os.path.join(app.config['TTS_FOLDER'], text_filename)
        with open(text_path, 'w') as text_file:
            text_file.write(f"Text:\n{text}\n\nSentiment: {sentiment}")

    return redirect('/')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/tts/<filename>')
def get_tts_file(filename):
    """Serve TTS files."""
    return send_from_directory(app.config['TTS_FOLDER'], filename)

@app.route('/script.js', methods=['GET'])
def scripts_js():
    """Serve JavaScript file."""
    return send_file('./script.js')


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
    #app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8080)))


