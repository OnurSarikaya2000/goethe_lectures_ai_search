import os
import whisper


audioPath = "vl_mp3/LinADI 2a - Lineare Algebra und Diskrete Mathematik für die Informatik.mp3"

model = whisper.load_model("medium")

directory = 'vl_mp3'

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".mp3"):
        result = model.transcribe(os.path.join(
            directory, filename), language='de', fp16=False)
        text_file = open(filename.replace('mp3', 'txt'), "w")
        text_file.write(result['text'])
        text_file.close()
        continue
