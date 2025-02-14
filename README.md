# Spanish Learning Game

This project is a Spanish learning game that helps users practice their Spanish language skills through interactive dialogues. The game uses speech recognition, text-to-speech, and translation models to provide an immersive learning experience.

## Features

- Interactive dialogues organized by topics
- Speech recognition for filling in the blanks
- Text-to-speech for pronunciation
- Translation of sentences to English
- Score tracking and average statistics

## Installation

### Prerequisites

- Python 3.11
- [pip](https://pip.pypa.io/en/stable/installation/)

### Clone the Repository

```bash
git clone https://github.com/yourusername/language_game.git
cd language_game
```

### Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Additional Setup

- Ensure you have `ffmpeg` installed for audio processing.
- Install `sounddevice` for recording audio from the microphone.

## Usage

Run the main script to start the game:

```bash
python spanish_game.py
```

Follow the on-screen instructions to choose a topic, input method, and start practicing your Spanish.

