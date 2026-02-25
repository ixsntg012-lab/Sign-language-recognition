import pyttsx3

try:
    engine = pyttsx3.init('sapi5')   # force Windows SAPI5
    engine.setProperty('rate', 150)

    voices = engine.getProperty('voices')
    print("Available voices:", voices)

    engine.setProperty('voice', voices[0].id)

    engine.say("Hello this is a working test.")
    engine.runAndWait()

    print("Speech finished successfully.")

except Exception as e:
    print("Error:", e)