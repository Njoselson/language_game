import random
import time
import os
import shutil
from datetime import datetime
from colorama import Fore, Style
from transformers import pipeline
import torch
import numpy as np
import pyfiglet
import librosa
import soundfile as sf

# Load offline models for speech-to-text and text-to-speech
# Speech-to-Text (Wav2Vec2 model from Hugging Face)
speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-xlsr-53-spanish", device="cuda" if torch.cuda.is_available() else "cpu")

# Text-to-Speech (Coqui TTS for Spanish)
from TTS.api import TTS
tts = TTS(model_name="tts_models/es/css10/vits", progress_bar=False, gpu=False)

# Translation (Helsinki-NLP/opus-mt-es-en for Spanish to English)
translator = pipeline("translation_es_to_en", model="Helsinki-NLP/opus-mt-es-en")

# Load phrases from file
with open('phrases.txt', 'r', encoding='utf-8') as file:
    data = file.read()

# Organize phrases by topic and group them into dialogues
def parse_phrases(data):
    topics = {}
    current_topic = None
    current_dialogue = []
    for line in data.split('\n'):
        if line.strip().endswith("(Continued)") or line.strip().endswith("Continued)"):
            if current_topic and current_dialogue:
                topics[current_topic].append(current_dialogue)
            current_topic = line.split("/")[0].strip()
            topics[current_topic] = []
            current_dialogue = []
        elif line.strip() and current_topic:
            if "S1:" in line or "S2:" in line:
                current_dialogue.append(line.strip())
            else:
                if current_dialogue:
                    topics[current_topic].append(current_dialogue)
                current_dialogue = []
    if current_topic and current_dialogue:
        topics[current_topic].append(current_dialogue)
    return topics

topics = parse_phrases(data)

def blank_out_word(sentence):
    words = sentence.split()
    if len(words) == 0:
        return sentence, None
    word_to_blank = random.choice(words)
    blanked_sentence = sentence.replace(word_to_blank, '_____', 1)
    return blanked_sentence, word_to_blank

def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio), axis=0)
    return audio

def trim_audio(audio, sample_rate):
    # Trim the non-silent parts of the audio
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=20)
    return trimmed_audio

def recognize_speech():
    print(Fore.YELLOW + "Please say the missing word..." + Style.RESET_ALL)
    try:
        # Record audio from the microphone
        import sounddevice as sd

        fs = 16000  # Sample rate
        duration = 5  # Reduced duration to 5 seconds
        print("Recording...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished

        # Normalize the audio
        audio = normalize_audio(audio)

        # Trim the audio to just the part of the word
        audio = trim_audio(audio, fs)

        # Convert the numpy array to a format suitable for the speech recognizer
        audio = audio.flatten()
        audio = librosa.resample(audio, orig_sr=fs, target_sr=16000)

        # Transcribe the audio
        result = speech_recognizer(audio)
        text = result["text"].strip().lower()
        print(Fore.YELLOW + f"Transcription result: {text}" + Style.RESET_ALL)
        return text
    except Exception as e:
        print(Fore.RED + f"Error: {e}" + Style.RESET_ALL)
        return None

def speak_sentence(sentence):
    print(Fore.MAGENTA + f"Pronouncing: {sentence}" + Style.RESET_ALL)
    tts.tts_to_file(text=sentence, file_path="temp.wav")
    os.system("afplay temp.wav" if os.name == "posix" else "start temp.wav")  # macOS/Windows

def translate_sentence(sentence):
    translation = translator(sentence)[0]['translation_text']
    os.system('clear')
    print(center_text(Fore.YELLOW + f"Translation: {translation}" + Style.RESET_ALL))
    time.sleep(2)
    os.system('clear')
    return translation

def save_score(score, total, percentage):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('scores.txt', 'a') as file:
        file.write(f"{timestamp} - Score: {score}/{total} ({percentage:.2f}%)\n")

def get_average_stats():
    try:
        with open('scores.txt', 'r') as file:
            lines = file.readlines()
            if len(lines) < 5:
                return None
            last_five_scores = lines[-5:]
            total_correct = 0
            total_questions = 0
            for line in last_five_scores:
                parts = line.split(' - ')[1].split(' ')
                score = parts[1].split('/')[0]
                total = parts[1].split('/')[1]
                total_correct += int(score)
                total_questions += int(total)
            average_percentage = (total_correct / total_questions) * 100
            return total_correct, total_questions, average_percentage
    except FileNotFoundError:
        return None

def center_text(text):
    columns, rows = shutil.get_terminal_size()
    lines = text.split('\n')
    vertical_padding = (rows - len(lines)) // 2
    centered_lines = [line.center(columns) for line in lines]
    return '\n' * vertical_padding + '\n'.join(centered_lines) + '\n' * vertical_padding

def run_quiz(sleep_time, num_sentences, topic, input_method):
    correct_answers = 0
    total_questions = 0
    total_sentences = 0

    dialogues = topics.get(topic, [])
    if not dialogues:
        print(Fore.RED + f"No dialogues found for topic: {topic}" + Style.RESET_ALL)
        return

    random.shuffle(dialogues)

    for dialogue in dialogues:
        if total_sentences >= num_sentences:
            break
        for phrase in dialogue:
            if total_sentences >= num_sentences:
                break
            s1 = phrase.replace("S1:", "").replace("S2:", "").strip()
            os.system('clear')
            print(center_text(Fore.BLUE + f"Original: {s1}" + Style.RESET_ALL))
            time.sleep(sleep_time)
            os.system('clear')
            blanked_sentence, correct_word = blank_out_word(s1)
            if correct_word:
                print(center_text(Fore.GREEN + f"Fill in the blank: {blanked_sentence}" + Style.RESET_ALL))
                if input_method == "speech":
                    user_input = recognize_speech()  # Capture user's spoken input
                else:
                    user_input = input("Your answer: ").strip().lower()
                if user_input and user_input == correct_word.lower():
                    print(center_text(Fore.GREEN + "Correct!" + Style.RESET_ALL))
                    correct_answers += 1
                else:
                    print(center_text(Fore.RED + f"Incorrect. The correct word was: {correct_word}" + Style.RESET_ALL))
                total_questions += 1
                total_sentences += 1
                speak_sentence(s1)  # Pronounce the entire sentence
                translate_sentence(s1)  # Translate the sentence to English
            print()

    if total_questions > 0:
        score_percentage = (correct_answers / total_questions) * 100
        print(center_text(Fore.YELLOW + f"Your score: {correct_answers}/{total_questions} ({score_percentage:.2f}%)" + Style.RESET_ALL))
        save_score(correct_answers, total_questions, score_percentage)
    else:
        print(center_text(Fore.RED + "No questions were asked." + Style.RESET_ALL))

# Main menu
def main():
    print(Fore.CYAN + pyfiglet.figlet_format("Welcome to the Spanish Learning Game!", font="slant") + Style.RESET_ALL)
    print("Choose a topic:")
    for i, topic in enumerate(topics.keys(), 1):
        print(f"{i}. {topic}")
    choice = int(input("Enter the number of your choice: "))
    topic = list(topics.keys())[choice - 1]

    input_method = input("Choose input method (text/speech): ").strip().lower()
    if input_method not in ["text", "speech"]:
        print(Fore.RED + "Invalid input method. Defaulting to text." + Style.RESET_ALL)
        input_method = "text"

    sleep_time = int(input("Enter the display time for each sentence (in seconds): "))
    num_sentences = int(input("Enter the number of sentences to practice: "))

    run_quiz(sleep_time, num_sentences, topic, input_method)

    average_stats = get_average_stats()
    if average_stats:
        total_correct, total_questions, average_percentage = average_stats
        print(center_text(Fore.CYAN + f"Average stats of the last 5 games: {total_correct}/{total_questions} ({average_percentage:.2f}%)" + Style.RESET_ALL))
    else:
        print(center_text(Fore.RED + "Not enough data to calculate average stats." + Style.RESET_ALL))

if __name__ == "__main__":
    main()
