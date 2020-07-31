import AudioAnalizer
import VideoEmotion
import concurrent
import time


if __name__ == "__main__":
 
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(AudioAnalizer.run)
        video_emotions = VideoEmotion.run()
        audio_emotions = future.result()
        total_emotions = [x + y for x, y in zip(video_emotions, audio_emotions)]
        p = [e/sum(total_emotions) for e in total_emotions]
        print(f'Enojo: {p[0]}, Disgusto: {p[1]}, Miedo: {p[2]}, Felicidad:{p[3]}, Tristeza: {p[4]}, Sorpresa: {p[5]}, Neutral: {p[6]}')
        
        
    
