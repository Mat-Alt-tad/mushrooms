from src.data_loader import DataLoader
from src.train_model import MushroomModel

def main():
    print("🍄 Entrenando modelo de predicción de hongos...\n")
    
    # Cargar datos
    loader = DataLoader("data/raw/Mushrooms-Dataset.csv")
    df = loader.cargar_datos()
    
    # Entrenar modelo
    modelo = MushroomModel()
    metricas_modelo = modelo.entrenar(df)
    
    print("\n✅ Entrenamiento completado con éxito.")
    print(f"📊 Métricas del modelo: {metricas_modelo}")
    print("💾 Modelo guardado en la carpeta 'models/'")

if __name__ == "__main__":
    main()