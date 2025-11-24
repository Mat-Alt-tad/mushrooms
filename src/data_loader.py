import pandas as pd

class DataLoader:

    def __init__(self, ruta):
        self.ruta = ruta
    
    def cargar_datos(self):
 
        datos = pd.read_csv(self.ruta)
        print(f"Datos cargados correctamente: {datos.shape[0]} filas y {datos.shape[1]} columnas.")
        print(f"Distribución de clases: {datos['class'].value_counts().to_dict()}")
        return datos