# coding=utf-8
import os
import time
import serial
import numpy as np


# Calcula el vector diferencia del vector values pasado
def get_increments( values ):
    length = len(values)
    
    differences = []
    i = 0
    while i < length-1:
        
        f = values[i]
        s = values[i+1]
        
        differences.append(int(s - f))
        
        i += 1

    return differences

# Obtiene la matriz de transición, construyendo una matriz dimension X dimension
# basandose en los valores de data_array
def get_transition_matrix( dimension, data_array):
    middle = int(dimension/2)
    length = len(data_array)
    transition_matrix = np.zeros([dimension,dimension])

    if length > 1:
        i = 0
        while i < length-1:
            # Sumamos 1 a la posicion data_array[i],data_array[i+1] que indica
            # que la transicion data_array[i] --> data_array[i+1] se ha producido
            transition_matrix[data_array[i]+middle][data_array[i+1]+middle] += 1
            i += 1
    
    # Todas las posiciones que contienen un 0, se le suma el valor 10e-6
    transition_matrix[ transition_matrix == 0] += 10e-6

    i = 0
    while i < dimension:
        # Para que la suma de cada una de las filas sume 1, dividimos por
        # la fila por la suma de todos los elementos de la fila.
        transition_matrix[i] = transition_matrix[i]/transition_matrix[i].sum() 
        i += 1
    
    return transition_matrix # a square matrix

# En esta funcion se comprueba que la matriz de transición 
# cumple la condición de que todas sus filas sumen 1.
def satisfy_condition(transition_matrix, dimension):
    i = 0
    condition = True 
    while i < dimension and condition:
        if float(transition_matrix[i].sum()) < 0.999999999:
            condition = False
        
        i += 1

    if not condition:
        print("The transition matrix is malformed.")
    else:
        print("The transition matrix is formed correctly.")
    
    return

# Calcula el minimo likelihood, es decir, devuelve el umbral necesario
# para detectar las anomalías basándonos en unos valores pasados, en una 
# matriz de transición obtenida anteriormente, y un tamaño de ventana.
def minimum_likelihood( values, transition_matrix, window_size ):
    length = len(values)
    middle = int(transition_matrix.shape[1]/2)
    minimum = 1

    i = 0
    # Calculamos para cada ventana, de dimension indicada en window_size, 
    # con incrementos de 1, el likelihood. Guardamos el minimo de ellos.
    while i < length-(window_size+1):
        j = 0
        likeli = 1

        # El bucle produce el likelihood para una ventana
        while j < window_size:
            likeli *= transition_matrix[values[i+(j-1)]+middle][values[i+(j-1)+1]+middle]
            j += 1
        
        if likeli < minimum:
            minimum = likeli

        i += 1 

    return minimum

# Detectamos las anomalias de forma Offline para un fichero que contenga la información
def detect_anomalies_from_file(file, transition_matrix, threshold, window_size):
    if os.path.exists(file):
        with open(file) as f:
            try:
                middle = int(transition_matrix.shape[1]/2) # square matrix

                values = [int(linea) for linea in f]
                increments = get_increments(values)

                length = len(increments)

                if length > 1:
                    i = 0
                    # Vamos leyendo el vector increments con ventanas de tamaño window_size
                    while i < length-(window_size+1):
                        j = 0
                        likeli = 1
                        # Calculamos el likelihood para una ventana
                        while j < window_size:
                            likeli *= transition_matrix[increments[i+(j-1)]+middle][increments[i+(j-1)+1]+middle]
                            j += 1

                        if likeli < threshold:
                            # i + 2 --> +2 because "i" stats in 0 and increments has length (len(values)-1),
                            # and we want to show the line of the file passed by argument.
                            print("We have found an anomaly in lines [", i+2, ", ", i+(j-1)+2, "] with value ", likeli)

                        i += 1
            except IOError:
                print("\nThe file ",file,"doesn't exist.\n")
    return

# Desplaza en uno hacia la derecha el vector pasado como parametros.
def displace( vector ):
    i = 1
    while i < len(vector):
        vector[i-1] = vector[i]
        i += 1
    vector[i-1] = 0
    return vector

# Detectamos en tiempo real las anomalías
def detect_anomalies_real_time(dev, baudrate, transition_matrix, threshold, window_size):
    
    middle = int(transition_matrix.shape[1]/2) # square matrix

    arduino = serial.Serial(dev, baudrate, timeout = 1.0)

    # Provocamos un reseteo manual de la placa para leer desde
    # el principio, ver http://stackoverflow.com/a/21082531/554319
    arduino.setDTR(False)
    time.sleep(1)
    arduino.flushInput()
    arduino.setDTR(True)

    i = 0
    pila = np.zeros(window_size) # Vector de tamaño window_size
    while True:
        try:
            # Leemos linea del puerto Serial
            line = arduino.readline()
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        
        if not line:
            # Descartamos lineas vacias
            continue

        try:
            next_value = int(line)
        except ValueError:
            # Si produce fallo al pasarlo a int, letras o carácteres no válidos
            continue
        
        if i < window_size:   
            # Llenamos la pila
            pila[i] = next_value
            i += 1
        else:
            # Calculamos los incrementos
            increments = get_increments( pila )
            
            j = 0
            likeli = 1
            # Calculamos el likelihood del vector increments
            while j < window_size-2: # increments length is window_size-1
                likeli *= transition_matrix[increments[j]+middle][increments[j+1]+middle]    
                j += 1

            if likeli < threshold:
                print("We have found an anomaly in ", pila, " with value ", likeli)

            # Añadimos el siguiente valor
            pila = displace(pila)
            pila[i-1] = next_value

        print(pila)
    return
