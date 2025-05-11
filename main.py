# Import necessary libraries
import ssl
import subprocess
import os
from datetime import timedelta
import time

# Audio/video processing and AI-related imports
import whisperx as whisper
from transformers import MarianMTModel, MarianTokenizer, pipeline
from pysubparser import parser  # For subtitle parsing
from TTS.api import TTS  # Text-to-speech
from pydub import AudioSegment  # Audio manipulation
from audiotsm import wsola  # Time-scale modification
from audiotsm.io.wav import WavReader, WavWriter

# Telegram bot related
import telebot
from config import *  # Contains BOT_TOKEN and other configurations
import tempfile

# SSL context configuration for secure connections
ssl._create_default_https_context = ssl._create_unverified_context

# Global constants and configurations
audio_tagging_time_resolution = 10  # Time resolution for audio tagging in seconds
model_name = 'medium'  # Whisper model version

# Load Whisper speech recognition model
model = whisper.load_model(model_name, download_root='models', device='cpu', compute_type='float32')

# Initialize text-to-speech engine
tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to('cpu')

# Set custom speaker
speaker = 'voices/speaker.wav'

# Initialize Telegram bot
bot = telebot.TeleBot(BOT_TOKEN)


def detect_leading_silence(sound, silence_threshold=-70, chunk_size=10):
    """
    Detect leading silence in an audio segment.
    Returns the duration in milliseconds of silence from the start.
    """
    trim_ms = 0  # Milliseconds to trim from start

    assert chunk_size > 0  # Prevent infinite loop
    # Find first non-silent chunk
    while sound[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms


def predict(sample_text, mmodel, tokenizer):
    """Translate text using MarianMT model"""
    batch = tokenizer(sample_text, return_tensors="pt")
    gen = mmodel.generate(**batch)
    translation_arr = tokenizer.batch_decode(gen, skip_special_tokens=True)
    print(translation_arr)
    return translation_arr[0]


def time_to_ms(time):
    """Convert datetime.time object to total milliseconds"""
    return ((time.hour * 60 + time.minute) * 60 + time.second) * 1000 + time.microsecond / 1000 / 2


def generate_audio(export_path, lang, speaker_wav, video_path, atempo_wav):
    """
    Generate dubbed audio file from subtitles with time synchronization.
    Processes each subtitle segment, generates speech, adjusts timing, and combines with video.
    """
    path = export_path
    print(f"Generating audio file for {path} with xtts_v2")

    subtitles = parser.parse(path)
    audio_sum = AudioSegment.empty()  # Final combined audio track

    with tempfile.TemporaryDirectory() as tmpdirname:
        print('Created temporary directory', tmpdirname)

        temp_file_path = os.path.join(tmpdirname, "temp.wav")
        prev_subtitle = None
        prev_audio_duration_ms = 0

        # Process each subtitle segment
        for subtitle in subtitles:
            late_time = 0  # Track accumulated timing discrepancies

            # Generate speech from text
            tts.tts_to_file(
                text=subtitle.text,
                speaker_wav=speaker,
                language=lang,
                file_path=temp_file_path,
                speed=1.25
            )

            audio_segment = AudioSegment.from_wav(temp_file_path)

            # Trim silence from start and end
            start_trim = detect_leading_silence(audio_segment)
            end_trim = detect_leading_silence(audio_segment.reverse())
            duration = len(audio_segment)
            audio_segment = audio_segment[start_trim:duration - end_trim]

            # Calculate timing parameters
            sub_time = time_to_ms(subtitle.end) - time_to_ms(subtitle.start)
            subtime = sub_time / 1000
            audiolength = audio_segment.duration_seconds

            print(f'Audio duration: {audiolength}, Segment duration: {subtime}')

            # Calculate needed silence before this segment
            if prev_subtitle is None:
                silence_duration_ms = time_to_ms(subtitle.start)
            else:
                silence_duration_ms = time_to_ms(subtitle.start) - len(audio_sum)

            # Handle timing overflow from previous segments
            if len(audio_sum) > time_to_ms(subtitle.start):
                late_time = len(audio_sum) - time_to_ms(subtitle.start)

            # Calculate speed adjustment factor
            try:
                atempo = audiolength / (subtime - late_time / 1000)
                atempo = round(atempo, 1)
            except:
                print('Exception in atempo calculation')
                atempo = 1

            print(f'Atempo factor: {atempo}')

            # Apply speed adjustment if needed
            if atempo < 0:
                atempo = 1.35
            if atempo > 1:
                if atempo >= 1.35:
                    atempo = 1.35
                    print('Capping atempo to 1.35')
                # Apply WSOLA time-stretching
                with WavReader(temp_file_path) as reader:
                    with WavWriter(atempo_wav, reader.channels, reader.samplerate) as writer:
                        tsm = wsola(reader.channels, speed=atempo)
                        tsm.run(reader, writer)
                audio_segment = AudioSegment.from_wav(atempo_wav)

            print(f'Audio duration after atempo: {audio_segment.duration_seconds}')

            # Update late time calculation
            if audio_segment.duration_seconds * 1000 > subtime * 1000:
                late_time += (audio_segment.duration_seconds * 1000 - subtime * 1000)

            # Adjust silence duration accounting for late time
            silence_duration_ms = silence_duration_ms - late_time
            if silence_duration_ms < 0:
                silence_duration_ms = 0

            # Combine audio segments
            audio_sum += AudioSegment.silent(duration=silence_duration_ms) + audio_segment

            # Update tracking variables
            prev_subtitle = subtitle
            prev_audio_duration_ms = len(audio_segment)

        # Add final silence to match video length
        audio_segment_sum_length = len(audio_sum)
        video_length = get_length(video_path) * 1000
        silence_duration_ms = video_length - audio_segment_sum_length

        if video_length > audio_segment_sum_length:
            audio_sum += AudioSegment.silent(duration=silence_duration_ms)

        # Export final audio track
        with open(os.path.splitext(path)[0] + '.wav', 'wb') as out_f:
            audio_sum.export(out_f, format='wav')


def get_length(filename):
    """Get video duration in seconds using ffprobe"""
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", filename
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)


def offline_video_dubber(message, video_path, target_lang):
    """
    Main video dubbing pipeline:
    1. Generate translated subtitles
    2. Create dubbed audio track
    3. Merge audio and video
    4. Send result back to user
    """
    with tempfile.TemporaryDirectory():
        # Send initial processing message
        lang_messages = {
            'ru': 'Russian',
            'ar': 'Arabic',
            'en': 'English'
        }
        if target_lang in lang_messages:
            bot.send_message(
                message.chat.id,
                f'Started translating your video into {lang_messages[target_lang]}...',
                reply_to_message_id=message.id,
                parse_mode='Markdown'
            )

        start_time = time.time()
        # Temporary file names
        subtitles_file = f"{message.chat.id}_{message.message_id}_subtitles.srt"
        output_file = f"{message.chat.id}_{message.message_id}_dubbing.mp4"
        tmp_dubbed_voice = f"{message.chat.id}_{message.message_id}_dubbing2.mp4"
        final_video = f"{message.chat.id}_{message.message_id}_voiceandvideo.mp4"
        subtitles_wav = f"{message.chat.id}_{message.message_id}_subtitles.wav"
        speaker_wav = f"{message.chat.id}_{message.message_id}_speaker.wav"
        atempo_wav = f"{message.chat.id}_{message.message_id}_temp_file.wav"

        # Generate subtitles
        bot.send_message(message.chat.id, 'Generating subtitles with translation...',
                         reply_to_message_id=message.id)
        summary = gen_subtitles_for_video(message, video_path, target_lang)

        if target_lang != 'sub':
            # Extract original audio
            bot.send_message(message.chat.id, 'Turning subtitles into voice...',
                             reply_to_message_id=message.id)
            command = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 22050 -vn {speaker_wav}"
            subprocess.call(command, shell=True)

            # Generate dubbed audio
            generate_audio(subtitles_file, target_lang, speaker_wav, video_path, atempo_wav)

            # Process audio timing
            sound = AudioSegment.from_wav(subtitles_wav)
            extract = sound[:int(get_length(video_path) * 1000)]  # Trim to video length
            extract.export(subtitles_wav, format="wav")

            # Burn subtitles into video
            ffmpeg_command = f"""ffmpeg -i {video_path} -vf "subtitles={subtitles_file}:force_style='FontName=Arial,FontSize=10,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=3,Outline=1,Shadow=1,Alignment=2,MarginV=10'" -c:a copy {output_file} -y"""
            subprocess.run(ffmpeg_command, shell=True)

            # Merge audio tracks
            ffmpeg_command = f"""ffmpeg -i {output_file} -af "volume=1" -c:v copy {tmp_dubbed_voice} -y"""
            subprocess.run(ffmpeg_command, shell=True)
            ffmpeg_command = f"""ffmpeg -y -i {tmp_dubbed_voice} -i {subtitles_wav} -c:v copy -filter_complex "[0:a]aformat=fltp:44100:stereo,apad[0a];[1]aformat=fltp:44100:stereo,volume=1.5[1a];[0a][1a]amerge[a]" -map 0:v -map "[a]" -ac 2 {final_video}"""
            subprocess.run(ffmpeg_command, shell=True)

            print(f"Processing time: {time.time() - start_time}")
        else:
            # Only burn subtitles without audio processing
            ffmpeg_command = f"""ffmpeg -i {video_path} -vf "subtitles={subtitles_file}:force_style='FontName=Arial,FontSize=10,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=3,Outline=1,Shadow=1,Alignment=2,MarginV=10'" -c:a copy {final_video} -y"""
            subprocess.run(ffmpeg_command, shell=True)

        # Send final result to user
        lang_captions = {
            'ru': 'Russian',
            'ar': 'Arabic',
            'en': 'English',
            'sub': 'subtitles'
        }
        caption = f'Translation of your video into {lang_captions.get(target_lang, "unknown")} is complete!'
        if target_lang != 'sub':
            caption += '\nSummary:\n' + summary

        bot.send_video(
            message.chat.id,
            video=open(final_video, 'rb'),
            reply_to_message_id=message.id,
            caption=caption,
            supports_streaming=True
        )


def gen_subtitles_for_video(message, video_path, target_lang):
    """
    Generate translated subtitles file (.srt) using Whisper and translation models.
    Returns generated summary of the content.
    """
    global current_state

    # Detect source language
    audio = whisper.load_audio(video_path)
    src_lang = model.detect_language(audio)
    print(f"Detected source language: {src_lang}")

    if target_lang == 'sub':
        # Generate subtitles without translation
        result = model.transcribe(video_path, batch_size=4)
        model_a, metadata = whisper.load_align_model(
            language_code=result["language"], device='cpu')
        result = whisper.align(result["segments"], model_a, metadata,
                               video_path, 'cpu', return_char_alignments=False)

        segments = result["segments"]
        text = ' '.join([line['text'] for line in segments[:50]])

        # Generate summary
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(text, min_length=5)[0]['summary_text']
        print(f"Generated summary: {summary}")

        # Write subtitles to .srt file
        if os.path.exists(f"{message.chat.id}_{message.message_id}_subtitles.srt"):
            os.remove(f"{message.chat.id}_{message.message_id}_subtitles.srt")

        for index, segment in enumerate(segments):
            # Format timestamps
            start_time = str(0) + str(timedelta(seconds=int(segment['start']))) + ',000'
            end_time = str(0) + str(timedelta(seconds=int(segment['end']))) + ',000'
            text = segment['text'].strip()

            # Write subtitle entry
            srt_entry = f"{index + 1}\n{start_time} --> {end_time}\n{text}\n\n"
            with open(f"{message.chat.id}_{message.message_id}_subtitles.srt", 'a', encoding='utf-8') as srt_file:
                srt_file.write(srt_entry)

        return summary
    else:
        # Translate to target language
        result = model.transcribe(video_path, batch_size=4, language='en', task='translate')
        model_a, metadata = whisper.load_align_model(
            language_code=result["language"], device='cpu')
        result = whisper.align(result["segments"], model_a, metadata,
                               video_path, 'cpu', return_char_alignments=False)

        # Load translation model if needed
        if target_lang != 'en':
            model_name = f'opus-mt-en-{target_lang}'
            mmodel = MarianMTModel.from_pretrained(f'models/{model_name}')
            tokenizer = MarianTokenizer.from_pretrained(f'models/{model_name}', trust_remote_code=True)

        segments = result["segments"]
        text = ' '.join([line['text'] for line in segments[:50]])

        # Generate and translate summary
        summarizer = pipeline("summarization")
        summary_text = summarizer(text, min_length=5)[0]['summary_text']
        if target_lang != 'en':
            summary = predict(summary_text, mmodel, tokenizer)
        else:
            summary = summary_text
        print(f"Translated summary: {summary}")

        # Create translated subtitles file
        if os.path.exists(f"{message.chat.id}_{message.message_id}_subtitles.srt"):
            os.remove(f"{message.chat.id}_{message.message_id}_subtitles.srt")

        for index, segment in enumerate(segments):
            # Format timestamps with milliseconds
            start_time = str(0) + str(timedelta(seconds=int(segment['start']))) + \
                         f',{str(segment["start"]).split(".")[1]}'
            end_time = str(0) + str(timedelta(seconds=int(segment['end']))) + \
                       f',{str(segment["end"]).split(".")[1]}'

            # Translate text
            if target_lang != 'en':
                text = predict(segment['text'], mmodel, tokenizer)
            else:
                text = segment['text']

            # Write subtitle entry
            srt_entry = f"{index + 1}\n{start_time} --> {end_time}\n{text.strip()}\n\n"
            with open(f"{message.chat.id}_{message.message_id}_subtitles.srt", 'a', encoding='utf-8') as srt_file:
                srt_file.write(srt_entry)

        return summary


# Telegram bot handlers
@bot.message_handler(commands=['start'])
def start_message(message):
    """Handle /start command with welcome message"""
    welcome_text = (
        "Hi!\n"
        "I can translate videos into Russian, Arabic, English, and also generate a summary!\n\n"
        "When sending a video, indicate in its description what language I should translate into:\n"
        "• ru - Russian (default)\n"
        "• ar - Arabic\n"
        "• en - English\n"
        "• sub - adding subtitles to video without translation"
    )
    bot.send_message(message.chat.id, welcome_text, parse_mode='Markdown')


@bot.message_handler(content_types=['video', 'text'])
def video_handler(message):
    """Handle incoming video files"""
    lang = 'ru'  # Default language
    try:
        # Get video file from message
        file_info = bot.get_file(message.video.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Set target language from caption
        if message.caption:
            lang = message.caption.lower()

        # Save video locally
        src = file_info.file_path
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)

        try:
            # Start dubbing process
            offline_video_dubber(message, src, lang)
        except Exception as e:
            print(f"Error during processing: {repr(e)}")
            bot.send_message(
                message.chat.id,
                "There was an error during translation. Please check: "
                "1. Video description format\n"
                "2. Source language\n"
                "3. Video content\n"
                "And try resending the video.",
                reply_to_message_id=message.id
            )
    except Exception as e:
        print(f"General error: {repr(e)}")
        bot.send_message(
            message.chat.id,
            "Error loading video. Please check:\n"
            "• Video size (max 20MB)\n"
            "• Video format",
            reply_to_message_id=message.id
        )


if __name__ == '__main__':
    print('Bot started')
    bot.polling(True)  # Start Telegram bot polling
