import os
import uuid
import tempfile
from gtts import gTTS
from playsound import playsound

text = "Hello! I am Jarvis. Testing my voice."

# Create a unique temp path WITHOUT keeping the file open
tmp_path = os.path.join(tempfile.gettempdir(), f"jarvis_{uuid.uuid4().hex}.mp3")

try:
    tts = gTTS(text=text, lang="en")
    tts.save(tmp_path)          # write after we closed everything
    playsound(tmp_path)         # play it
finally:
    # Clean up (ignore errors if file is locked)
    try:
        os.remove(tmp_path)
    except Exception:
        pass
