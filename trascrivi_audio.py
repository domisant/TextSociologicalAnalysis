import whisper

# Scegli il modello (tiny, base, small, medium, large)
model = whisper.load_model("medium")

# Percorso del file audio (modifica con il tuo nome file)
file_audio = "audio.wav"

# Trascrizione (lingua italiana forzata per maggiore precisione)
result = model.transcribe(file_audio, language="it")

# Salva il testo su file
with open("trascrizione.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

print("✅ Trascrizione completata. File salvato in 'trascrizione.txt'")
