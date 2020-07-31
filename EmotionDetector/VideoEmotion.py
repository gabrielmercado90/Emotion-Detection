import numpy as np
from NeuralNet import EmotionClassifier
import cv2

def run():
    detector = EmotionClassifier('modelo/model.json')
    detector.loadWeights('modelo/pesos_23.07.2020-17_11_44.H5')

    video = cv2.VideoCapture(0)
    while video.isOpened():
        ret, frame = video.read()
        if frame is not None:
            output = detector.predict(frame)
            #mostrar
            cv2.imshow('capture', output) 

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            #p = detector.getPercentages()
            #print(f'Enojo: {"{:.2f}".format(p[0])}%, Disgusto: {"{:.2f}".format(p[1])}%, Miedo: {"{:.2f}".format(p[2])}%, Felicidad:{"{:.2f}".format(p[3])}%, Tristeza: {"{:.2f}".format(p[4])}%, Sorpresa: {"{:.2f}".format(p[5])}%, Neutral: {"{:.2f}".format(p[6])}%')
            break
        
    video.release()
    cv2.destroyAllWindows()
    return detector.contadores
