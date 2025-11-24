import pandas as pd

class Preprocesador:
    
    
    CARACTERISTICAS_SELECCIONADAS = [
        'bruises', 'gill-color', 'gill-size', 'gill-spacing', 'habitat', 
        'odor', 'population', 'ring-type', 'spore-print-color', 
        'stalk-color-above-ring', 'stalk-color-below-ring', 'stalk-root', 
        'stalk-surface-above-ring', 'stalk-surface-below-ring'
    ]
    
    def preparar_datos(self, datos):
        
        print("Realizando procesamiento de los datos de hongos...")
        
        df = datos.copy()
        
       
        print(f"Valores faltantes por columna: {df.isnull().sum().sum()}")
        
       
        y = df['class'].copy()
        
       
        y = y.map({'e': 0, 'p': 1})
        
        X = df[self.CARACTERISTICAS_SELECCIONADAS].copy()
        

        X_encoded = pd.get_dummies(X, drop_first=False)
        
        print(f"Características seleccionadas: {len(self.CARACTERISTICAS_SELECCIONADAS)}")
        print(f"Características después del encoding: {X_encoded.shape[1]}")
        print(f"Distribución de clases: {y.value_counts().to_dict()}")
        
        return X_encoded, y
    
    def obtener_caracteristicas(self, datos):
        
        return self.CARACTERISTICAS_SELECCIONADAS