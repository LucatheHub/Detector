import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import noisereduce as nr



# programa funciona con un array de microfonos dispuestos de la siguiente forma:
#el microfono 1 en una posicion(0,0), el microfono 2 en una posicion(-2,0) 
# y el microfono 3 en una posicion (0,-2) donde cada unidad es un metro.
# Cargar un archivo de audio WAV
archivo_de_audio = 'sound_1_1.wav'

#en realidad aqui deberia ir una lista con los 3 audios para hacer los calculos para la posicion:
# archivo_de_audio = ['grabacion_1.wav', 'grabacion_2.wav', 'grabacion_3.wav']  

fs, señal_original = wavfile.read(archivo_de_audio)
señal_original = señal_original[0:int(fs/2)]
tamaño_señal_original = señal_original.shape
if tamaño_señal_original[1]!=1:
    señal_original = señal_original[:,1]

# Parámetros del filtro
frecuencias_filtro = [[4000,8000],[8000,16000]]
orden_del_filtro = 4
señales_filtradas = []

señal_original = nr.reduce_noise(y=señal_original, prop_decrease=1, sr=fs) #limpio de ruido la señal


# Diseñar el filtro pasa banda
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

for frecuencia_filtro in frecuencias_filtro:
    b, a = butter_bandpass(frecuencia_filtro[0], frecuencia_filtro[1], fs, orden_del_filtro)
    señal_filtrada = lfilter(b, a, señal_original)
    señales_filtradas.append(señal_filtrada)

#dB de la señal filtrada
senal_db = 20 * np.log10(np.abs(señales_filtradas))

#vector de tiempo
tiempo = np.arange(len(señal_filtrada)) / fs


def filtro_media_movil(signal, window_size):
   
    # Verifica que el tamaño de la ventana sea válido
    if window_size <= 0 or window_size >= len(signal):
        raise ValueError("El tamaño de la ventana debe ser mayor que 0 y menor que la longitud de la señal.")
    
    # Crea una lista para almacenar la señal suavizada
    smoothed_signal = []

    # Aplica el filtro de media móvil
    for i in range(len(signal)):
        start = max(0, i - window_size // 2)
        end = min(len(signal), i + window_size // 2 + 1)
        window = signal[start:end]
        smoothed_value = np.mean(window)
        smoothed_signal.append(smoothed_value)

    return np.array(smoothed_signal)

window_size = 500
smoothed_signals = []

for i in range(int(len(frecuencias_filtro))):
    smoothed_signal = filtro_media_movil(senal_db[i,:], window_size)
    smoothed_signals.append(smoothed_signal)

    #print(smoothed_signal)

    plt.figure(figsize=(5, 3))
    plt.plot(tiempo, senal_db[i,:], label='Señal original', marker='o')
    plt.plot(tiempo, smoothed_signal, label='Señal suavizada', linestyle='--')
    plt.xlabel('Tiempo')
    plt.ylabel('Valor de la señal')
    plt.title('Señal Original vs. Señal Suavizada')
    plt.legend()
    plt.grid(True)
    plt.show()

def filtrar_y_calcular_promedio(arreglo):
    promedio_original = np.mean(arreglo)
    umbral = 100
    arreglo_filtrado = arreglo[abs(arreglo - promedio_original) <= umbral]
    nuevo_promedio = np.mean(arreglo_filtrado)
    return arreglo_filtrado, nuevo_promedio

segmentos = np.array_split(senal_db, 20, axis=1)

# Inicializar listas para almacenar los índices de máximo y mínimo en cada segmento
indices_maximos = []
indices_minimos = []



# Encontrar los índices de máximo y mínimo en cada segmento
for segmento in segmentos:
    indice_maximo = np.argmax(segmento, axis=1)
    indice_minimo = np.argmin(segmento, axis=1)
    indices_maximos.append(indice_maximo)
    indices_minimos.append(indice_minimo)
    

indices_minimos = np.array(indices_minimos)
indices_maximos = np.array(indices_maximos)



#creo una muestar igual a la que cargue pero con un pequeno delay para mostrar
#como funciona el codigo, aunque en la realidad seria la grabacion de
#otro microfono.

muestras_de_ruido = np.random.normal(0, 1, 257)
senal_db2 = np.zeros_like(senal_db)
senal_db2[0,:][0:257] = muestras_de_ruido
senal_db2[1,:][0:257] = muestras_de_ruido
senal_db2[0,:][257:] = senal_db[0,:][0:(22050-257)]
senal_db2[1,:][257:] = senal_db[1,:][0:(22050-257)]
segmentos2 = np.array_split(senal_db2, 20, axis=1)
indices_maximos2 = []
indices_minimos2 = []


#hago otra muestra ya que solo con 2 el resultado es ambiguo y necesito
#un 3er microfono.


# Encontrar los índices de máximo y mínimo en cada segmento
for segmento in segmentos2:
    indice_maximo = np.argmax(segmento, axis=1)
    indice_minimo = np.argmin(segmento, axis=1)
    indices_maximos2.append(indice_maximo)
    indices_minimos2.append(indice_minimo)
    

indices_minimos2 = np.array(indices_minimos2)
indices_maximos2 = np.array(indices_maximos2)
resultado_hor1 = indices_maximos2 - indices_maximos
resultado_hor2 = indices_minimos2 - indices_minimos
promedio_maximos = np.mean(resultado_hor1)
promedio_minimos = np.mean(resultado_hor2)

arreglo_filtrado_ver1, nuevo_promedio_ver1 = filtrar_y_calcular_promedio(promedio_maximos)
arreglo_filtrado_ver2, nuevo_promedio_ver2 = filtrar_y_calcular_promedio(promedio_minimos)


resultado_hor = (nuevo_promedio_ver1 + nuevo_promedio_ver2) / 2



    
    
    
muestras_de_ruido = np.random.normal(0, 1, 100)
senal_db3 = np.zeros_like(senal_db)
senal_db3[0,:][0:100] = muestras_de_ruido
senal_db3[0,:][100:] = senal_db[0,:][0:(22050-100)]
senal_db3[1,:][0:100] = muestras_de_ruido
senal_db3[1,:][100:] = senal_db[1,:][0:(22050-100)]
segmentos3 = np.array_split(senal_db3, 20, axis=1)
indices_maximos3 = []
indices_minimos3 = []
for segmento in segmentos3:
    indice_maximo = np.argmax(segmento, axis=1)
    indice_minimo = np.argmin(segmento, axis=1)
    indices_maximos3.append(indice_maximo)
    indices_minimos3.append(indice_minimo)
    

indices_maximos3 = np.array(indices_maximos3)
indices_minimos3 = np.array(indices_minimos3)
resultado_ver1 = indices_maximos - indices_maximos3
resultado_ver2 = indices_minimos - indices_minimos3
promedio_maximos_ver = np.mean(resultado_ver1)
promedio_minimos_ver = np.mean(resultado_ver2)




arreglo_filtrado_ver1, nuevo_promedio_ver1 = filtrar_y_calcular_promedio(resultado_ver1)
arreglo_filtrado_ver2, nuevo_promedio_ver2 = filtrar_y_calcular_promedio(resultado_ver2)



resultado_ver = (nuevo_promedio_ver1 + nuevo_promedio_ver2) / 2   






def clasificar_resultado(resultado):
    if 230 <= resultado <= 257:
        return 1
    elif 182 <= resultado < 230:
        return 2
    elif 115 <= resultado < 182:
        return 3
    elif 0 <= resultado < 115:
        return 4
    elif -115 <= resultado < 0:
        return 5
    elif -181 <= resultado < -115:
        return 6
    elif -182 <= resultado < -230:
        return 7
    elif -257 <= resultado < -230:
        return 8
    else:
        return None 
posicion_vertical = clasificar_resultado(resultado_ver)
posicion_horizontal = clasificar_resultado(resultado_hor)


#el resultado se interpreta de la siguiente fora:
#dividi el plano en 8 segmentos con origen en el microfono uno y separados
# cada 22.5 grados desde el eje x en sentido antihorario. 



    
    









    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
