import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def blackman_tukey(x, M=None):   
    
    # N = len(x)
    x_z = x.shape
    
    N = np.max(x_z)
    
    if M is None:
        M = N//5
    
    r_len = 2*M-1

    # hay que aplanar los arrays por np.correlate.
    # usaremos el modo same que simplifica el tratamiento
    # de la autocorr
    xx = x.ravel()[:r_len]

    r = np.correlate(xx, xx, mode='same') / r_len

    Px = np.abs(np.fft.fft(r * sig.windows.blackman(r_len), n = N) )

    Px = Px.reshape(x_z)

    return Px

def print_info_senal(nombre, senal, fs):
    
    """
    Imprime información básica de una señal.
    nombre: str, nombre de la señal
    senal: np.ndarray, señal (vector columna o fila)
    fs: float, frecuencia de muestreo en Hz
    """
    N = senal.shape[0]
    df = fs / N
    print(f'------{nombre}------')
    print('Cantidad de muestras: ', N)
    print(f'Frecuencia de muestreo: {fs} Hz')
    print(f'Duracion de la señal: {N / fs} seg')
    print(f'Df: {df:0.3f} Hz')
    print('Potencia: ', np.round(np.var(senal)))
    print()



def calcular_ancho_banda(f, psd, umbral=0.95):
    """
    Calcula el ancho de banda efectivo de una señal a partir de su PSD.
    
    Parámetros:
    - f: array de frecuencias (Hz)
    - psd: array de densidad espectral (misma longitud que f)
    - umbral: fracción de energía total (0.95 o 0.98 típicamente)
    
    Retorna:
    - fmin, fmax: límites de frecuencia que contienen el 'umbral' de la energía
    """
    psd = np.asarray(psd)
    f = np.asarray(f)

    # Energía total
    energia_total = np.trapz(psd, f)

    # Energía acumulada
    energia_acumulada = np.cumsum(psd) * (f[1] - f[0])  # aproximación trapezoidal
    energia_acumulada /= energia_total  # normaliza a [0, 1]

    # fmin: donde empieza a superar el 0.5*(1 - umbral)
    idx_min = np.searchsorted(energia_acumulada, (1 - umbral)/2)
    idx_max = np.searchsorted(energia_acumulada, 1 - (1 - umbral)/2)

    fmin = f[idx_min]
    fmax = f[min(idx_max, len(f) - 1)]

    return fmin, fmax


