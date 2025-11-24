import streamlit as st
import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CARACTERISTICAS_SELECCIONADAS = [
    'bruises', 'gill-color', 'gill-size', 'gill-spacing', 'habitat', 
    'odor', 'population', 'ring-type', 'spore-print-color', 
    'stalk-color-above-ring', 'stalk-color-below-ring', 'stalk-root', 
    'stalk-surface-above-ring', 'stalk-surface-below-ring'
]

def main():
    st.set_page_config(
        page_title="Predicción de Hongos - ML App",
        page_icon="🍄",
        layout="wide"
    )
    
    st.title("Predicción de Hongos - ML App")
    st.write("Ingrese las características del hongo para predecir si es **comestible** o **venenoso**.")
    
    model_path = "models/modelo_hongos.joblib"
    
    if not os.path.exists(model_path):
        st.error("⚠️ **Modelo no encontrado**")
        st.info("💡 **Solución:** Ejecuta primero `python main.py` para entrenar el modelo.")
        
        if st.button("🚀 Entrenar Modelo Ahora"):
            from src.data_loader import DataLoader
            from src.train_model import MushroomModel
            
            with st.spinner("Entrenando modelo..."):
                loader = DataLoader("data/raw/Mushrooms-Dataset.csv")
                df = loader.cargar_datos()
                
                modelo = MushroomModel()
                metricas = modelo.entrenar(df)
                
                st.success("✅ Modelo entrenado exitosamente!")
                st.json(metricas)
                st.rerun()
        return
    
    try:
        model = joblib.load(model_path)
        st.success("✅ Modelo cargado exitosamente")
        
        st.info(f"📊 Modelo entrenado con {len(model.feature_names_in_)} características")
        
    except Exception as e:
        st.error(f"❌ Error cargando modelo: {e}")
        return
    
    st.subheader(f"📋 Características del Hongo ({len(CARACTERISTICAS_SELECCIONADAS)} características seleccionadas)")
    
    with st.form("mushroom_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🌿 Características Principales**")
            
            bruises = st.selectbox(
                "¿Tiene moretones?",
                ['', 'yes=t', 'no=f'],
                help="Si el hongo se daña y cambia de color"
            )
            
            odor = st.selectbox(
                "Olor del hongo",
                ['', 'almond=a', 'anise=l', 'creosote=c', 'fishy=y', 'foul=f', 'musty=m', 'none=n', 'pungent=p', 'spicy=s'],
                help="Olor que desprende el hongo",
                index=7  
            )
            
            st.markdown("**🔸 Características de las Lamelas**")
            
            gill_size = st.selectbox(
                "Tamaño de lamelas",
                ['', 'broad=b', 'narrow=n'],
                help="Ancho de las lamelas"
            )
            
            gill_spacing = st.selectbox(
                "Espaciado de lamelas",
                ['', 'close=c', 'crowded=w', 'distant=d'],
                help="Distancia entre lamelas"
            )
            
            gill_color = st.selectbox(
                "Color de las lamelas",
                ['', 'black=k', 'brown=n', 'buff=b', 'chocolate=h', 'gray=g', 'green=r', 'orange=o', 'pink=p', 'purple=u', 'red=e', 'white=w', 'yellow=y'],
                help="Color de las lamelas"
            )
        
        with col2:
            st.markdown("**📏 Características del Tallo**")
            
            stalk_root = st.selectbox(
                "Raíz del tallo",
                ['', 'bulbous=b', 'club=c', 'cup=u', 'equal=e', 'rhizomorphs=z', 'rooted=r', 'missing=?'],
                help="Tipo de raíz del tallo"
            )
            
            stalk_surface_above_ring = st.selectbox(
                "Superficie del tallo (sobre anillo)",
                ['', 'fibrous=f', 'scaly=y', 'silky=k', 'smooth=s'],
                help="Textura del tallo arriba del anillo"
            )
            
            stalk_surface_below_ring = st.selectbox(
                "Superficie del tallo (bajo anillo)",
                ['', 'fibrous=f', 'scaly=y', 'silky=k', 'smooth=s'],
                help="Textura del tallo abajo del anillo"
            )
            
            stalk_color_above_ring = st.selectbox(
                "Color del tallo (sobre anillo)",
                ['', 'brown=n', 'buff=b', 'cinnamon=c', 'gray=g', 'orange=o', 'pink=p', 'red=e', 'white=w', 'yellow=y'],
                help="Color del tallo arriba del anillo"
            )
            
            stalk_color_below_ring = st.selectbox(
                "Color del tallo (bajo anillo)",
                ['', 'brown=n', 'buff=b', 'cinnamon=c', 'gray=g', 'orange=o', 'pink=p', 'red=e', 'white=w', 'yellow=y'],
                help="Color del tallo abajo del anillo"
            )
        
        st.markdown("**🔘 Características Adicionales**")
        col3, col4 = st.columns(2)
        
        with col3:
            ring_type = st.selectbox(
                "Tipo de anillo",
                ['', 'cobwebby=c', 'evanescent=e', 'flaring=f', 'large=l', 'none=n', 'pendant=p', 'sheathing=s', 'zone=z'],
                help="Forma del anillo"
            )
            
            spore_print_color = st.selectbox(
                "Color de impresión de esporas",
                ['', 'black=k', 'brown=n', 'buff=b', 'chocolate=h', 'green=r', 'orange=o', 'purple=u', 'red=e', 'white=w', 'yellow=y'],
                help="Color de la impresión de esporas"
            )
        
        with col4:
            population = st.selectbox(
                "Población",
                ['', 'abundant=a', 'clustered=c', 'numerous=n', 'scattered=s', 'several=v', 'solitary=y'],
                help="Cómo crecen los hongos"
            )
            

            habitat = st.selectbox(
                "Hábitat",
                ['', 'grasses=g', 'leaves=l', 'meadows=m', 'paths=p', 'urban=u', 'waste=w', 'woods=d'],
                help="Lugar donde crece el hongo"
            )
        

        submitted = st.form_submit_button("🔍 Predecir si es Comestible", use_container_width=True)
    
    if submitted:
        campos_requeridos = [
            bruises, odor, gill_size, gill_spacing, gill_color,
            stalk_root, stalk_surface_above_ring, stalk_surface_below_ring,
            stalk_color_above_ring, stalk_color_below_ring, ring_type,
            spore_print_color, population, habitat
        ]
        
        if any(campo == '' for campo in campos_requeridos):
            st.error("❌ **Por favor completa todos los campos antes de hacer la predicción**")
            return
        
        try:
            input_data = {
                'bruises': [bruises.split('=')[1]],
                'gill-color': [gill_color.split('=')[1]],
                'gill-size': [gill_size.split('=')[1]],
                'gill-spacing': [gill_spacing.split('=')[1]],
                'habitat': [habitat.split('=')[1]],
                'odor': [odor.split('=')[1]],
                'population': [population.split('=')[1]],
                'ring-type': [ring_type.split('=')[1]],
                'spore-print-color': [spore_print_color.split('=')[1]],
                'stalk-color-above-ring': [stalk_color_above_ring.split('=')[1]],
                'stalk-color-below-ring': [stalk_color_below_ring.split('=')[1]],
                'stalk-root': [stalk_root.split('=')[1]],
                'stalk-surface-above-ring': [stalk_surface_above_ring.split('=')[1]],
                'stalk-surface-below-ring': [stalk_surface_below_ring.split('=')[1]]
            }
            
            X_input = pd.DataFrame(input_data)
            
            X_encoded = pd.get_dummies(X_input, drop_first=False)
            
            model_features = model.feature_names_in_
            X_final = pd.DataFrame(0, index=[0], columns=model_features)
            
            for col in X_encoded.columns:
                if col in model_features:
                    X_final[col] = X_encoded[col].values[0]
            
            for feature in model_features:
                if feature not in X_final.columns or X_final[feature].isna().any():
                    X_final[feature] = 0
            
            pred = model.predict(X_final)[0]
            proba = model.predict_proba(X_final)[0]
            
            st.subheader("📊 Resultado de la Predicción")
            
            st.info(f"🔍 **Debug:** Características procesadas: {len(model_features)}")
            
            edible_prob = proba[0] * 100
            poisonous_prob = proba[1] * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("🍃 Probabilidad Comestible", f"{edible_prob:.1f}%")
            with col2:
                st.metric("☠️ Probabilidad Venenoso", f"{poisonous_prob:.1f}%")
            
            
            if pred == 1:  # Venenoso
                st.error("🍄 **EL HONGO ES VENENOSO** ⚠️")
                st.warning("❌ **NO LO CONSUMAS** - Es peligroso para la salud")
            else:  # Comestible
                st.success("✅ **EL HONGO ES COMESTIBLE**")
                st.info("👍 **Parece seguro para el consumo**")
            
            # Mostrar datos utilizados
            with st.expander("📋 Ver datos utilizados para la predicción"):
                st.json(input_data)
                
        except Exception as e:
            st.error(f"❌ Error en la predicción: {e}")
            st.info("💡 Verifica que todas las características estén seleccionadas")
            
            # Debug info
            with st.expander("🔍 Información de Debug"):
                st.write("**Error:**", str(e))
                if 'input_data' in locals():
                    st.write("**Datos de entrada:**", input_data)
    
    # Información adicional
    st.markdown("---")
    st.markdown("### 📚 Información del Proyecto")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Dataset:**
        - 8,124 muestras de hongos
        - 14 características seleccionadas (Mayor correlación)
        - Clasificación: Comestible vs Venenoso
        
        **🤖 Modelo:**
        - Regresión Logística
        - Características con |correlación| >= 0.25
        - Variables dummy encoding
        """)
    
    with col2:
        st.markdown("""
        **⚠️ Importante:**
        - Este es un proyecto educativo
        - NO uses para identificar hongos reales
        - Consulta siempre un micólogo profesional
        
        **📊 Métricas del Modelo:**
        """)
        
        # Mostrar métricas si existen
        try:
            import json
            if os.path.exists("reports/metricas.json"):
                with open("reports/metricas.json", 'r') as f:
                    metrics = json.load(f)
                st.json({k: v for k, v in metrics.items() if not k.startswith('cv')})
        except:
            st.info("Métricas no disponibles")

if __name__ == "__main__":
    main()