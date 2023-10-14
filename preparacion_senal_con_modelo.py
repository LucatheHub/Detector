import librosa
import numpy as np
from scipy import signal
import noisereduce as nr
from PIL import Image
import matplotlib.pyplot as plt
import  joblib


modelo_guardado = joblib.load('Logistic Regression.plk')

def preparacion_senal(max_valores, archivo_audio):
    # Cargar el archivo de audio
    y, sr = librosa.load(archivo_audio)
    y = y[0:int(1.5*sr)]
    
    # Filtar el audio
    cutoff_freq = 500
    nyquist_freq = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = signal.butter(1, normal_cutoff, btype='high', analog=False)
    y_filtered = signal.filtfilt(b, a, y)

    # Reducción de ruido
    reduced_audio = nr.reduce_noise(y=y_filtered, sr=sr)

    # Crear el espectrograma
    D = librosa.amplitude_to_db(np.abs(librosa.stft(reduced_audio)), ref=np.max)

    # Guardar el espectrograma como una imagen
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='gray')
    plt.savefig('SHINI.png',  bbox_inches='tight', pad_inches=0, dpi=200)  # Corrección aquí

    # Abrir la imagen
    imagen = Image.open("SHINI.png")

    # Recortar la imagen
    imagen_array = np.array(imagen)  # Corrección aquí
    altura_original = imagen_array.shape[0]
    nueva_altura = altura_original * 7 // 9
    imagen_recortada_array = imagen_array[:nueva_altura, :]
    imagen_recortada = Image.fromarray(imagen_recortada_array)

    # Redimensionar la imagen
    nueva_resolucion = (100, 100)
    imagen_resized = imagen_recortada.resize(nueva_resolucion)

    # Convertir la imagen en un arreglo NumPy
    imagen_array = np.array(imagen_resized)

    # Flattening: Convertir la matriz 2D en un vector 1D
    vector_imagen = imagen_array.flatten()

    # Muestrear el vector si es necesario
    if len(vector_imagen) > max_valores:
        factor_muestreo = len(vector_imagen) // max_valores
        vector_imagen_muestreado = vector_imagen[::factor_muestreo]
    else:
        vector_imagen_muestreado = vector_imagen

    return vector_imagen_muestreado

# Ejemplo de uso de la función
max_valores = 20000
archivo_audio = "ambulance_test.mp3"
vector_imagen_muestreado = preparacion_senal(max_valores, archivo_audio)
vector_imagen_muestreado = vector_imagen_muestreado[13:20000]

vector_imagen_muestreado = vector_imagen_muestreado.reshape(1, -1)


Predicción = modelo_guardado.predict(vector_imagen_muestreado)


if Predicción > 0:
    Predicción = 'Ambulancia'
if Predicción == 1:
    Predicción = 'Policia'
if Predicción == 2:
    Predicción = 'Bombreo'
if Predicción == 3:
    Predicción = 'Nada'

print(Predicción)
