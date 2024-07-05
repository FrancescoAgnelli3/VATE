from pytube import Playlist

def scroll_playlist(url, overwrite = False):
    playlist = Playlist(url)

    # Iterate through video URLs and print them
    if overwrite:
        ow = "w"
    else:
        ow = "a"

    with open("input.txt", ow) as file:
        for video_url in playlist.video_urls:
            file.write(f"{video_url}\n")
    
# Replace with your playlist URL
playlist_url_1 = "https://www.youtube.com/playlist?list=PLjrML-f5aMc6vkuhQo6w-m53yBDisz4nU"

scroll_playlist(playlist_url_1, overwrite=True)

playlist_url_2 = "https://www.youtube.com/playlist?list=PLJic7bfGlo3qxHqFNEADdFjp074mqebyx"

scroll_playlist(playlist_url_2)

playlist_url_3 = "https://www.youtube.com/playlist?list=PLjrML-f5aMc6vkuhQo6w-m53yBDisz4nU"

scroll_playlist(playlist_url_3)

