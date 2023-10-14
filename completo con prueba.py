import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import noisereduce as nr
from scipy import signal
from PIL import Image
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
import joblib


num_splits = 5
skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42) #cuanto mas cercano a 100 mejor





# creo las variables necesarias para ejecutar las funciones y alamacenar los datos.
main_folder = "/Users/lucacorleto/Desktop/Detector/sireNNet"  #sireNNet3 es una modificacion del DF original, pero reducido.
subfolders = ["ambulance", "police", "firetruck"] 
wav_files = []

data = {
        'X': [],
        'y':[]     
        }
diccionario = {}
diccionario["vehiculos"] = []
max_valores = 20000



for i in range(max_valores):
    nombre_columna = f"columna {i + 1}"
    diccionario[nombre_columna] = []


#creo una lista con todos los directorios del DF para poder analizar las pistas de audio.
for subfolder in subfolders:
    subfolder_path = os.path.join(main_folder, subfolder)
    
    veicle_folders = [folder for folder in os.listdir(subfolder_path) if folder.startswith("sound")]
    
    veicle_folders.sort()
    
    for veicle_folder in veicle_folders:
        veicle_path = os.path.join(subfolder_path, veicle_folder)
        
        
        wav_files.append(veicle_path)


#creo los espectrogramas de cada pista de audio
i = 0 
for wav_file in wav_files:
   
    if wav_file.count("_") == 1 or "traffic" in wav_file:
        vehiculo = wav_file.split("/")
        diccionario['vehiculos'].append(vehiculo[-2])
        i = i + 1
        audio_file_path = wav_file
        y, sr = librosa.load(audio_file_path, sr=None)
        bloques = wav_file.split("/")
        vehiculo = bloques[-2]
        
        # Aplico un filtro pasa alto a partir de 500 Hz
        cutoff_freq = 500
        nyquist_freq = 0.5 * sr
        normal_cutoff = cutoff_freq / nyquist_freq
        b, a = signal.butter(1, normal_cutoff, btype='high', analog=False)
        y_filtered = signal.filtfilt(b, a, y)
        
        reduced_audio = nr.reduce_noise(y=y_filtered, prop_decrease=1, sr=sr) #limpio de ruido la señal
    
        
        # D es el espectrograma
        D = librosa.amplitude_to_db(np.abs(librosa.stft(reduced_audio)), ref=np.max)
        
        
    
        # Visualizar el espectrograma en blanco y negro, para que la
        # informacion tenga forma de vector y no de matriz.
        
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='gray')
        
        plt.savefig(f"{vehiculo}_{i}.png",  bbox_inches='tight', pad_inches=0, dpi=300) #guardo cada imagen, ya que la clasificacion sera a partir de reconocimiento de imagenes, no de sonido.
      
        # plt.show()
    
        # Cargo la unfirmacion de a img
        imagen = Image.open(f"{vehiculo}_{i}.png")
    
        # Convertir la imagen en un arreglo NumPy
        imagen_array = np.array(imagen)
        # estas lineas son para recortar las 2/9 partes inferiores de la imagen
        # ya que la mayoria es ruido y no aporta al analisis.
        # Obtener la altura original de la imagen
        altura_original = imagen_array.shape[0]
    
        # Calcular la nueva altura de la imagen (recortando las 2/9 partes de abajo)
        nueva_altura = altura_original * 7 // 9  # Usamos // para asegurarnos de que el resultado sea un número entero
    
        # Recortar la imagen
        imagen_recortada_array = imagen_array[:nueva_altura, :]
    
        # Convertir la matriz en una imagen PIL
        imagen_recortada = Image.fromarray(imagen_recortada_array)
    
        # Mostrar la nueva imagen
        # imagen_recortada.show()
        print(f'imagen{vehiculo}_{i} recortada')

        
        
    
    
        # Redimensionar la imagen a una resolución deseada (por ejemplo, 100x50)
        nueva_resolucion = (200, 200)
        imagen_resized = imagen_recortada.resize(nueva_resolucion)
    
        # Convertir la imagen en un arreglo NumPy
        imagen_array = np.array(imagen_resized)
    
        # Flattening: Convertir la matriz 2D en un vector 1D
        vector_imagen = imagen_array.flatten()
    
        # Muestrear el vector si tiene más de 5000 valores
        if len(vector_imagen) > max_valores:
            factor_muestreo = len(vector_imagen) // max_valores
            vector_imagen_muestreado = vector_imagen[::factor_muestreo]
        else:
            vector_imagen_muestreado = vector_imagen
    
        # Ahora 'vector_imagen_muestreado' contiene tu imagen en un vector de
        # máximo 5000 valores.
    
        # Mostrar la imagen resultante
        plt.imshow(imagen_resized, cmap='gray')  # cmap='gray' para mostrarla en escala de grises
        plt.title("Imagen Redimensionada")
        plt.axis('off')  # Ocultar ejes
        plt.show()
        
        
        
        # agrego la informacion al diccionario data
        data['X'].append(vector_imagen_muestreado)
        data['y'].append(vehiculo)
        print(f'imagen{vehiculo}_{i} agregada a "data"')



    


# itero por cada vector del diccionario data para que cada valor forme un nuevo
# valor del dicciorio "diccionario" y asi que cada valor sea una volumna del
# df cuando lo transforme
for vector in data['X']:
    for i, valor in enumerate(vector):
        nombre_columna = f"columna {i + 1}"
        diccionario[nombre_columna].append(valor)
        
df = pd.DataFrame(diccionario)


#analizando las imagenes creadas y el df puedo ver que las primeras 13 columnas
# tienen como valor 255(full black) asi que las elimino
df = df.drop(df.columns[1:14], axis=1)


mapeo = {'ambulance': 0, 'police': 1, 'firetruck': 2, 'traffic': 3}
for i in range(int(df.shape[0])):
    df['vehiculos'][i] = mapeo.get(df['vehiculos'][i])
df['vehiculos'] = df['vehiculos'].astype(int)


X = df.iloc[:, 1:]
y = df['vehiculos']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


modelos = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Support Vector Regressor": SVR(),
    "K Neighbors Regressor": KNeighborsRegressor()
}

resultados = {}
resultado_print = ''

for nombre_modelo, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    resultados[nombre_modelo] = mse
    nombre_archivo = f"{nombre_modelo}.plk"
    joblib.dump(modelo, nombre_archivo)

