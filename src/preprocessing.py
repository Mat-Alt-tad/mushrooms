import pandas as pd

class Preprocesador:
    """Simplifica los pasos de limpieza y transformación para el dataset de hongos."""
    
    # Características seleccionadas con correlación >= abs(0.25)
    CARACTERISTICAS_SELECCIONADAS = [
        'bruises', 'gill-color', 'gill-size', 'gill-spacing', 'habitat', 
        'odor', 'population', 'ring-type', 'spore-print-color', 
        'stalk-color-above-ring', 'stalk-color-below-ring', 'stalk-root', 
        'stalk-surface-above-ring', 'stalk-surface-below-ring'
    ]
    
    def preparar_datos(self, datos):
        """Prepara los datos para el entrenamiento del modelo."""
        print("Realizando procesamiento de los datos de hongos...")
        
        # Copia de seguridad
        df = datos.copy()
        
        # No hay valores faltantes en este dataset (según la descripción oficial)
        # Pero por seguridad, verificamos
        print(f"Valores faltantes por columna: {df.isnull().sum().sum()}")
        
        # Preparar las características eliminando la columna target 'class'
        # Separar target variable
        y = df['class'].copy()
        
        # Convertir 'class' a valores numéricos (0 = edible, 1 = poisonous)
        y = y.map({'e': 0, 'p': 1})
        
        # Preparar features (X) - Seleccionar solo características relevantes
        X = df[self.CARACTERISTICAS_SELECCIONADAS].copy()
        
        # Crear variables dummy para todas las características categóricas
        # Esto mantiene todas las características ya que todas son categóricas
        X_encoded = pd.get_dummies(X, drop_first=False)
        
        print(f"Características seleccionadas: {len(self.CARACTERISTICAS_SELECCIONADAS)}")
        print(f"Características después del encoding: {X_encoded.shape[1]}")
        print(f"Distribución de clases: {y.value_counts().to_dict()}")
        
        return X_encoded, y
    
    def obtener_caracteristicas(self, datos):
        """Obtiene las características originales seleccionadas para mostrar en la interfaz."""
        return self.CARACTERISTICAS_SELECCIONADAS