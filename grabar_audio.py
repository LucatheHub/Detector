import sounddevice as sd

print(sd.query_devices())
def grabar_audio(selected_input_device_index, duration=1.5, fs=44100):   
    # Grabar audio utilizando el dispositivo de entrada seleccionado
    y = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16', device=selected_input_device_index)
    sd.wait()  # Espera a que la grabación termine
    return y
    

selected_device_index = 0  # Reemplaza con el índice del dispositivo deseado
y = grabar_audio(selected_device_index)
