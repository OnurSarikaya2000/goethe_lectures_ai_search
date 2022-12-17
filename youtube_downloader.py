import os

import re
from pytube import Playlist
import moviepy.editor as mp

LINADI_PLAYLIST = "https://www.youtube.com/playlist?list=PLu7FZgjILMWFFiWsOkvvnBBPeBydXGlJJ"


def downloadVideos():
    YOUTUBE_STREAM_AUDIO = '140'  # modify the value to download a different stream

    playlist = Playlist(LINADI_PLAYLIST)

    # this fixes the empty playlist.videos list
    playlist._video_regex = re.compile(r"\"url\":\"(/watch\?v=[\w-]*)")

    print(len(playlist.video_urls))

    for url in playlist.video_urls:
        print(url)

    # physically downloading the audio track
    for video in playlist.videos:
        audioStream = video.streams.get_by_itag(YOUTUBE_STREAM_AUDIO)
        audioStream.download(output_path='vl_mp3')

    mp4_to_mp3()


def mp4_to_mp3():
    for file in os.listdir('vl_mp3'):
        if re.search('mp4', file):
            mp4_path = os.path.join('vl_mp3', file)
            mp3_path = os.path.join('vl_mp3', os.path.splitext(file)[0]+'.mp3')
            new_file = mp.AudioFileClip(mp4_path)
            new_file.write_audiofile(mp3_path)
            os.remove(mp4_path)


downloadVideos()
