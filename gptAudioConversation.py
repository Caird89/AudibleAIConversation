from google.cloud import texttospeech
import os
import pyaudio
import io
from pydub import AudioSegment
import speech_recognition as sr
import openai

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../Credentials/AudibleAIConversationCredentials/tts_google_cloud.json"

# Set up OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")


def text_to_speech(text):
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", name="en-US-Neural2-J"
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)

    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )

    return response.audio_content


def play_audio(audio_data):
    audio_io = io.BytesIO(audio_data)
    audio_segment = AudioSegment.from_file(audio_io, format="wav")

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(audio_segment.sample_width),
                    channels=audio_segment.channels,
                    rate=audio_segment.frame_rate,
                    output=True)
    stream.write(audio_segment.raw_data)

    stream.stop_stream()
    stream.close()
    p.terminate()


def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)

    try:
        credentials_json = "../Credentials/AudibleAIConversationCredentials/stt_google_cloud.json"
        recognized_text = r.recognize_google_cloud(audio, credentials_json=credentials_json)
        print("I think you said:", recognized_text)
        return recognized_text
    except sr.UnknownValueError:
        print("Google Cloud Speech could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Cloud Speech service: {e}")
        return None


def generate_response(user_response):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"\nHow can I assist you today?"
               f"\nHuman: {user_response}",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
    )

    response_text = response['choices'][0]['text']
    return response_text


# Example usage
print("Welcome to AI Conversation!")

opening_message = "Welcome to AI Conversation! Powered by Google Cloud and Open AI. So,What can I help you with today?"
audio_data = text_to_speech(opening_message)
play_audio(audio_data)

while True:
    # Convert user input to speech
    user_input = recognize_speech()
    if user_input:
        user_input = user_input.strip()
        if user_input.lower() == "exit":
            exit_message = "Well,My work here is done. It's back off to the cloud for me!"
            audio_data = text_to_speech(exit_message)
            play_audio(audio_data)
            print("Conversation ended.")
            break

        response_text = generate_response(user_input)
        print(response_text)

        # Convert AI response to speech
        audio_data = text_to_speech(response_text)
        play_audio(audio_data)

print("Thank you for using AI Conversation!")
