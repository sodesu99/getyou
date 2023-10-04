from pytube import YouTube

# 输入YouTube视频的URL
yt = YouTube('https://www.youtube.com/watch?v=0BxMmqxxRCE')
# 创建YouTube对象
print(yt.title)

# 选择视频流（例如，选择最高质量的音频流）
# audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').first()
# audio_stream = yt.streams.filter(type='audio')
# audio = audio_stream.get_by_itag("139")
audio_stream = yt.streams.filter(type='audio').first()
# print(audio)
print(audio_stream)

file_name = '2.mp'
# 下载音频
# audio_stream.download(output_path='/Users/zeng/project/python/youtube', filename= file_name)

# import moviepy.editor as mp
import os

output_path='/Users/zeng/project/python/youtube'
# # 输入MP4文件路径
mp4_file =os.path.join(output_path, file_name) 


import whisper

model = whisper.load_model("base")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio(mp4_file)
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
# result = whisper.decode(model, mel, options)

# print the recognized text
# print(result.text)
# # 输出WAV文件路径
# wav_file = os.path.join(output_path, "1.wav")
# # 使用moviepy库将MP4转换为WAV
# # clip = mp.VideoFileClip(mp4_file)
# clip = mp.AudioFileClip(mp4_file)
# clip.write_audiofile(wav_file)

# # 删除临时文件（可选）
# clip.close()

# # 打印转换后的WAV文件路径
# print("已转换为WAV文件：", wav_file)

# import speech_recognition as sr

# # 创建语音识别器对象
# recognizer = sr.Recognizer()

# # 读取下载的音频文件
# audio_file = mp4_file

# # 打开音频文件并识别文本
# with sr.AudioFile(audio_file) as source:
#     audio_data = recognizer.record(source)
#     try:
#         text = recognizer.recognize_google(audio_data, language="zh-CN")  # 或其他支持的语言
#         print("识别结果：", text)
#     except sr.UnknownValueError:
#         print("无法识别音频内容")
#     except sr.RequestError as e:
#         print(f"请求错误：{e}")
