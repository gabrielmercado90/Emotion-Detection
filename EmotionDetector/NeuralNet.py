import tensorflow as tf
import cv2
import numpy as np

class EmotionClassifier:
    
    emociones = ['Enojo', 'Disgusto', 'Miedo', 'Felicidad', 'Tristeza', 'Sorpresa', 'Neutral']
    colores = [(0,0,255), (0,255,0), (255,0,171), (0,255,255), (255,0,0), (228,162,255), (175,175,175)] #en BGR

    def __init__(self, json):
       #importar el modelo desde el JSON
        json_file = open(json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = tf.keras.models.model_from_json(loaded_model_json) 
        self.face_cascade = cv2.CascadeClassifier('C:/Users/User/miniconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        #contadores de emociones
        self.contadores = [0,0,0,0,0,0,0]

    def loadWeights(self, fileName):
        #importar los pesos
        self.loaded_model.load_weights(fileName)

    def predict(self, inputImg):
        #gray image
        inputImgGray = cv2.cvtColor(inputImg, cv2.COLOR_BGR2GRAY)
        output = inputImg.copy()
        #deteccion de cara
        faces = self.face_cascade.detectMultiScale(inputImgGray,1.3,5)
        
        for (x,y,w,h) in faces:
            #region de interes
            face = inputImgGray[y : y + h , x : x + w]            
            #clasificacion de emocion
            face = cv2.resize(face, (48,48), cv2.INTER_AREA).reshape(1,48,48,1)
            self.loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            outputValues = self.loaded_model.predict(face)
            confidence = outputValues.max()
            index = np.argmax(outputValues)
            #add to counters
            self.contadores[index] += 1
            if confidence > 0.5:
                #pintar rectangulo
                cv2.rectangle(output, (x,y), (x+w,y+h), self.colores[index], 2)
                #obtener la clase ganadora y la probabilidad
                categoria = self.emociones[index]
                cv2.putText(output, categoria+": %.2f"%(confidence*100)+'%', (x + 4, y + 20), cv2.FONT_HERSHEY_PLAIN,1, self.colores[index], 2)
        return output

    def getPercentages(self):
        return [item*100/sum(self.contadores) for item in self.contadores]

        

