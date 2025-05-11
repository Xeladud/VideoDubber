# Simple Offline AI Video Translation Telegram Bot

A Telegram bot for translating videos between languages (Russian, Arabic, English) with dubbed audio and subtitles, using state-of-the-art AI models.

## Features

- 🎥 Video translation with dubbed audio
- 🌐 Supported languages: Russian (ru), Arabic (ar), English (en)
- 📝 Automatic subtitle generation
- 📄 Content summarization
- ⏱ Audio time-stretching for synchronization with modern Waveform Similarity based Overlap-Add (WSOLA)
- 🔉 Text-to-speech with voice cloning if it needed
- 🎞 Burn subtitles directly into video
- 📦 Temporary file cleanup
- 📈 Progress tracking and status updates

## Prerequisites

- Python 3.8+
- FFmpeg
- Telegram API token
- At least 8GB RAM (recommended)
- 5GB+ free disk space for models


## Examples
### Telegram Bot Initial messaging
![alt text](TelegramBotStartExample.png?raw=true)

### Telegram Bot Translation proccess (EN -> RU)
![alt text](TelegramBotExample.png?raw=true)

### Result Video
link: https://github.com/Xeladud/VideoDubber/blob/main/VideoResultExample.mp4

## Installation

1. Clone repository:
```bash
git clone https://github.com/Xeladud/VideoDubber.git
```

2. Install dependencies:
```bash
pip install whisperx transformers pysubparser TTS pydub audiotsm telebot python-dotenv
#OR
pip install -r requirements.txt
```

3. Install FFmpeg:
```bash
# On Ubuntu/Debian
sudo apt-get install ffmpeg

# On MacOS
brew install ffmpeg
```

4. Prepare models and voices:
```bash
mkdir -p models
# Place MarianMT models in models/ directory

mkdir -p voices
# Place speaker voice samples in voices/ directory
```

## Configuration

Create `config.py` with your credentials:
```python
BOT_TOKEN = "your_telegram_bot_token"
```

## Usage

1. Start the bot:
```bash
python main.py
```

2. In Telegram:
- Send `/start` for instructions
- Send video file with caption:
  - `ru` for Russian
  - `ar` for Arabic
  - `en` for English
  - `sub` for subtitles only

3. Processing flow:
```
Video Received → Transcription → Translation → Audio Generation → Video Mixing → Result Sent
```

## Technical Details

### Key Components
- `bot.py`: Main bot handler
- `config.py`: Configuration file
- `/voices`: Speaker voice samples for TTS
- `/models`: Contains:
  - Whisper models
  - MarianMT translation models

### AI Models Used
- Whisper for speech recognition
- MarianMT for translation
- XTTS v2 for text-to-speech
- BART for summarization

### Important Functions
- `generate_audio()`: Handles audio processing pipeline
- `gen_subtitles_for_video()`: Manages subtitle generation
- `offline_video_dubber()`: Main video processing controller

## Limitations

- Video size limited to 20MB (Telegram API restriction)
- Translation quality depends on source audio clarity
- Current speaker voice is hardcoded (`voices/speaker.wav`)
- Tested on CPU-only processing (GPU acceleration possible)
- Limited real-time feedback during processing

## Contributing

1. Fork the repository
2. Create feature branch:
```bash
git checkout -b feature/your-feature
```
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

MIT License

## Acknowledgments

- OpenAI for Whisper
- Facebook Research for MarianMT
- Coqui AI for TTS
- Telegram for messaging platform

---

**Note:** First run will download large model files (5GB+). Ensure stable internet connection.
