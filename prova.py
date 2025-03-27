import whisper

# Carica il modello (puoi usare "tiny", "base", "small", "medium", "large")
model = whisper.load_model("medium")

# Trascrivi il file audio
result = model.transcribe("C:\\GitHub\\audio\\audio.m4a", language="it")

# Stampa il testo
print(result["text"])

# Salva su file
with open("trascrizione.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])