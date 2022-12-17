import os
import whisper

model = whisper.load_model("medium")

mp3_dir = 'vl_mp3'
txt_dir = 'vl_txt'

if not os.path.exists(txt_dir):
    os.mkdir(txt_dir)

for mp3 in os.listdir(mp3_dir):
    mp3_filename = os.fsdecode(mp3)
    if mp3_filename.endswith(".mp3"):
        result = model.transcribe(os.path.join(
            mp3_dir, mp3_filename), language='de', fp16=False)
        txt_file_name = mp3_filename.replace('mp3', 'txt')

        text_file = open(os.path.join(txt_dir, txt_file_name), "w")
        text_file.write(result['text'])
        text_file.close()
        continue
