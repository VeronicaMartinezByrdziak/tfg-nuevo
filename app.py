import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =====================================================
# CONFIGURACIÓN
# =====================================================
st.set_page_config(
    page_title="Predicción de deterioro cognitivo leve a Alzheimer",
    layout="centered"
)

st.title("Predicción de deterioro cognitivo leve a Alzheimer")

# =====================================================
# FUNCIÓN NECESARIA POR SI EL MODELO LA USA AL CARGAR
# =====================================================
def seleccionar_columnas_clinicas(X):
    return X

# =====================================================
# VARIABLES DEL MODELO CLÍNICO GROOT
# =====================================================
vars_clinicas = [
    "ADAS13",
    "RAVLT_immediate",
    "LDELTOTAL",
    "MOCA",
    "EcogPtTotal",
    "EcogSPMem",
    "EcogSPLang",
    "EcogSPVisspat",
    "EcogSPOrgan",
    "EcogSPDivatt",
    "EcogSPTotal"
]

# =====================================================
# VARIABLES DE MRI GROOT
# =====================================================
vars_mri = [
    "ST111TA",
    "ST12SV",
    "ST29SV",
    "ST55TA",
    "ST72TS",
    "ST84TS",
    "ST99TA"
]

# =====================================================
# ETIQUETAS BONITAS
# =====================================================
labels_clinico = {
    "ADAS13": "ADAS-13 – estado cognitivo global (0–85)",
    "RAVLT_immediate": "RAVLT memoria inmediata – recuerdo inmediato (0–75)",
    "LDELTOTAL": "Memoria diferida (Logical Memory) – recuerdo tras demora (0–25)",
    "MOCA": "Evaluación Cognitiva de Montreal (MOCA) – cribado cognitivo global (0–30)",
    "EcogPtTotal": "ECog paciente total – percepción global de cognición (1–4)",
    "EcogSPMem": "ECog memoria – dificultades de memoria (1–4)",
    "EcogSPLang": "ECog lenguaje – dificultades de lenguaje (1–4)",
    "EcogSPVisspat": "ECog visoespacial – dificultades espaciales (1–4)",
    "EcogSPOrgan": "ECog organización – dificultades organizativas (1–4)",
    "EcogSPDivatt": "ECog atención dividida – multitarea (1–4)",
    "EcogSPTotal": "ECog total – funcionamiento cognitivo global (1–4)"
}

labels_biomedico = {
    "ST111TA": "Grosor cortical precúneo derecho (mm) (1.7–2.6)",
    
    "ST12SV": "Volumen amígdala izquierda (mm³) (400–2400)",
    
    "ST29SV": "Volumen hipocampo izquierdo (mm³) (1500–5000)",
    
    "ST55TA": "Grosor cortical frontal medio rostral izquierdo (mm) (1.7–2.6)",
    
    "ST72TS": "Variabilidad grosor temporal superior derecho (mm) (0.25–0.75)",
    
    "ST84TS": "Variabilidad grosor frontal derecho (mm) (0.3–1.0)",
    
    "ST99TA": "Grosor cortical temporal medio derecho (mm) (1.5–3.0)"
}

labels_todas = {}
labels_todas.update(labels_clinico)
labels_todas.update(labels_biomedico)

# =====================================================
# VARIABLES CLAVE PARA SIMULACIÓN
# =====================================================
variables_simulacion = {
    "ADAS13": "sube",
    "RAVLT_immediate": "baja",
    "LDELTOTAL": "baja",
    "MOCA": "baja",
    "EcogPtTotal": "sube",
    "EcogSPMem": "sube",
    "EcogSPLang": "sube",
    "EcogSPVisspat": "sube",
    "EcogSPOrgan": "sube",
    "EcogSPDivatt": "sube",
    "EcogSPTotal": "sube",
    "ST111TA": "baja",
    "ST12SV": "baja",
    "ST29SV": "baja",
    "ST55TA": "baja",
    "ST72TS": "sube",
    "ST84TS": "sube",
    "ST99TA": "baja"
}

# =====================================================
# EJEMPLOS CLÍNICO
# =====================================================
ejemplos_clinico = {
    "Manual": {},
    "Caso límite (cercano al threshold)": {
        "ADAS13": 20.0,
        "RAVLT_immediate": 22.0,
        "LDELTOTAL": 3.0,
        "MOCA": 17.0,
        "EcogPtTotal": 1.51282,
        "EcogSPMem": 2.625,
        "EcogSPLang": 1.44444,
        "EcogSPVisspat": 2.14286,
        "EcogSPOrgan": 2.0,
        "EcogSPDivatt": 2.25,
        "EcogSPTotal": 1.94595
    },
    "TP – alto riesgo": {
        "ADAS13": 16.67,
        "RAVLT_immediate": 26.0,
        "LDELTOTAL": 3.0,
        "MOCA": 13.0,
        "EcogPtTotal": 1.48571,
        "EcogSPMem": 3.85714,
        "EcogSPLang": 3.44444,
        "EcogSPVisspat": 3.42857,
        "EcogSPOrgan": 3.66667,
        "EcogSPDivatt": 3.5,
        "EcogSPTotal": 3.59459
    },
    "TN – bajo riesgo": {
        "ADAS13": 7.0,
        "RAVLT_immediate": 54.0,
        "LDELTOTAL": 9.0,
        "MOCA": 26.0,
        "EcogPtTotal": 1.35897,
        "EcogSPMem": 1.125,
        "EcogSPLang": 1.0,
        "EcogSPVisspat": 1.0,
        "EcogSPOrgan": 1.0,
        "EcogSPDivatt": 1.25,
        "EcogSPTotal": 1.05128
    },
    "FP – caso dudoso": {
        "ADAS13": 18.0,
        "RAVLT_immediate": 30.0,
        "LDELTOTAL": 7.0,
        "MOCA": 14.0,
        "EcogPtTotal": 1.2973,
        "EcogSPMem": 3.875,
        "EcogSPLang": 2.77778,
        "EcogSPVisspat": 2.57143,
        "EcogSPOrgan": 3.83333,
        "EcogSPDivatt": 3.25,
        "EcogSPTotal": 3.33333
    }
}

# =====================================================
# EJEMPLOS COMPLETO GROOT
# =====================================================
ejemplos_biomedico = {
    "Manual": {},
    "Caso límite (cercano al threshold)": {
        "ADAS13": 20.0,
        "RAVLT_immediate": 22.0,
        "LDELTOTAL": 3.0,
        "MOCA": 17.0,
        "EcogPtTotal": 1.51282,
        "EcogSPMem": 2.625,
        "EcogSPLang": 1.44444,
        "EcogSPVisspat": 2.14286,
        "EcogSPOrgan": 2.0,
        "EcogSPDivatt": 2.25,
        "EcogSPTotal": 1.94595,
        "ST111TA": 1.878,
        "ST12SV": 976.2,
        "ST29SV": 2365.2,
        "ST55TA": 2.094,
        "ST72TS": 0.495,
        "ST84TS": 0.508,
        "ST99TA": 2.398
    },
    "TP – alto riesgo": {
        "ADAS13": 16.67,
        "RAVLT_immediate": 26.0,
        "LDELTOTAL": 3.0,
        "MOCA": 13.0,
        "EcogPtTotal": 1.48571,
        "EcogSPMem": 3.85714,
        "EcogSPLang": 3.44444,
        "EcogSPVisspat": 3.42857,
        "EcogSPOrgan": 3.66667,
        "EcogSPDivatt": 3.5,
        "EcogSPTotal": 3.59459,
        "ST111TA": 1.859,
        "ST12SV": 1019.2,
        "ST29SV": 2765.6,
        "ST55TA": 1.942,
        "ST72TS": 0.619,
        "ST84TS": 0.488,
        "ST99TA": 2.382
    },
    "TN – bajo riesgo": {
        "ADAS13": 7.0,
        "RAVLT_immediate": 54.0,
        "LDELTOTAL": 9.0,
        "MOCA": 26.0,
        "EcogPtTotal": 1.35897,
        "EcogSPMem": 1.125,
        "EcogSPLang": 1.0,
        "EcogSPVisspat": 1.0,
        "EcogSPOrgan": 1.0,
        "EcogSPDivatt": 1.25,
        "EcogSPTotal": 1.05128,
        "ST111TA": 2.34,
        "ST12SV": 1660.9,
        "ST29SV": 4317.9,
        "ST55TA": 2.351,
        "ST72TS": 0.373,
        "ST84TS": 0.531,
        "ST99TA": 2.689
    },
    "FP – caso dudoso": {
        "ADAS13": 18.0,
        "RAVLT_immediate": 30.0,
        "LDELTOTAL": 7.0,
        "MOCA": 14.0,
        "EcogPtTotal": 1.2973,
        "EcogSPMem": 3.875,
        "EcogSPLang": 2.77778,
        "EcogSPVisspat": 2.57143,
        "EcogSPOrgan": 3.83333,
        "EcogSPDivatt": 3.25,
        "EcogSPTotal": 3.33333,
        "ST111TA": 1.995,
        "ST12SV": 958.0,
        "ST29SV": 3150.9,
        "ST55TA": 2.078,
        "ST72TS": 0.515,
        "ST84TS": 0.843,
        "ST99TA": 2.598
    }
}

# =====================================================
# CARGAR MODELOS
# =====================================================
@st.cache_resource
def cargar_modelos():
    modelo_clinico = joblib.load("modelo_clinico_groot.joblib")
    modelo_biomedico = joblib.load("modelo_alzheimer_groot.joblib")
    return modelo_clinico, modelo_biomedico

modelo_clinico, modelo_biomedico = cargar_modelos()

# =====================================================
# ESTADO DE LA APP
# =====================================================
if "pantalla" not in st.session_state:
    st.session_state.pantalla = "inicio"

if "modelo" not in st.session_state:
    st.session_state.modelo = None

# =====================================================
# FUNCIONES AUXILIARES
# =====================================================
def limpiar_valor_entrada(valor):
    if valor is None:
        return np.nan

    valor = str(valor).strip()

    if valor == "":
        return np.nan

    valor = valor.replace(",", ".")
    return pd.to_numeric(valor, errors="coerce")

def obtener_variables_faltantes(df, columnas_modelo):
    faltantes = []
    for col in columnas_modelo:
        if pd.isna(df.at[0, col]):
            faltantes.append(col)
    return faltantes

def obtener_pruebas_recomendadas(faltantes):
    pruebas = []

    faltan_clinicas = any(v in faltantes for v in vars_clinicas)
    faltan_mri = any(v in faltantes for v in vars_mri)

    if faltan_clinicas:
        pruebas.append("evaluación cognitiva y funcional")
    if faltan_mri:
        pruebas.append("resonancia magnética cerebral")

    return pruebas

def mostrar_aviso_pruebas_recomendadas(faltantes):
    pruebas = obtener_pruebas_recomendadas(faltantes)

    if len(pruebas) == 0:
        return

    texto = "Pruebas recomendadas: " + ", ".join(pruebas) + "."
    st.info(texto)

def graficar_simulacion_empeoramiento(df_base, pipe, columnas_modelo, threshold):
    niveles = [0, 5, 10, 15, 20, 25]
    riesgos = []

    variables_usadas = []
    for var, direccion in variables_simulacion.items():
        if var in df_base.columns and pd.notna(df_base.at[0, var]):
            variables_usadas.append((var, direccion))

    if len(variables_usadas) == 0:
        st.info("No hay suficientes variables clave completadas para mostrar la simulación de empeoramiento.")
        return

    for nivel in niveles:
        df_sim = df_base.copy()
        factor = nivel / 100.0

        for var, direccion in variables_usadas:
            valor_original = float(df_base.at[0, var])

            if direccion == "sube":
                if valor_original == 0:
                    nuevo_valor = valor_original + factor
                else:
                    nuevo_valor = valor_original * (1 + factor)
            else:
                nuevo_valor = valor_original * (1 - factor)

            df_sim.at[0, var] = nuevo_valor

        proba_sim = pipe.predict_proba(df_sim[columnas_modelo])[:, 1][0]
        riesgos.append(proba_sim * 100)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(niveles, riesgos, marker="o")
    ax.axhline(threshold * 100, linestyle="--", label="Umbral del modelo")
    ax.set_xlabel("Empeoramiento simulado aplicado a variables clave (%)")
    ax.set_ylabel("Riesgo predicho (%)")
    ax.set_title("Simulación de empeoramiento y cambio del riesgo")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()

    st.pyplot(fig)

    nombres_bonitos = [labels_todas.get(var, var) for var, _ in variables_usadas]
    st.caption("Variables usadas en la simulación: " + "; ".join(nombres_bonitos))
    st.caption(
        "Esta gráfica muestra cómo cambia la salida del modelo si empeoran de forma simulada "
        "algunas variables clave seleccionadas. No equivale a una evolución temporal real del paciente."
    )

def obtener_diccionario_ejemplos(modelo):
    if modelo == "clinico":
        return ejemplos_clinico
    return ejemplos_biomedico

# =====================================================
# PANTALLA 1 — SELECCIÓN DE MODELO
# =====================================================
if st.session_state.pantalla == "inicio":

    st.header("Selecciona el tipo de modelo")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Modelo clínico"):
            st.session_state.modelo = "clinico"
            st.session_state.pantalla = "prediccion"
            st.rerun()

    with col2:
        if st.button("Modelo clínico + biomedico"):
            st.session_state.modelo = "biomedico"
            st.session_state.pantalla = "prediccion"
            st.rerun()

# =====================================================
# PANTALLA 2 — PREDICCIÓN
# =====================================================
if st.session_state.pantalla == "prediccion":

    if st.button("⬅ Volver"):
        st.session_state.pantalla = "inicio"
        st.rerun()

    if st.session_state.modelo == "clinico":
        artefacto = modelo_clinico
        st.subheader("Modelo clínico")
    else:
        artefacto = modelo_biomedico
        st.subheader("Modelo clínico + biomedico")

    pipe = artefacto["pipeline"]
    columnas = artefacto["columnas_modelo"]
    threshold = artefacto["threshold"]

    ejemplos_disponibles = obtener_diccionario_ejemplos(st.session_state.modelo)
    lista_ejemplos = list(ejemplos_disponibles.keys())

    if "Caso límite (cercano al threshold)" in lista_ejemplos:
        indice_por_defecto = lista_ejemplos.index("Caso límite (cercano al threshold)")
    else:
        indice_por_defecto = 0

    opcion_ejemplo = st.selectbox(
        "Selecciona un caso de ejemplo o introduce los datos manualmente",
        lista_ejemplos,
        index=indice_por_defecto,
        key=f"selectbox_ejemplo_{st.session_state.modelo}"
    )

    if opcion_ejemplo != "Manual":
        if "TP" in opcion_ejemplo:
            st.info("Caso de ejemplo cargado: riesgo alto.")
        elif "TN" in opcion_ejemplo:
            st.info("Caso de ejemplo cargado: riesgo bajo.")
        else:
            st.info("Caso de ejemplo cargado: perfil cercano al umbral.")

    st.write("Introduce los datos del paciente")
    st.caption("Puedes dejar campos vacíos. También puedes escribir decimales con coma o con punto.")

    ejemplo_actual = ejemplos_disponibles.get(opcion_ejemplo, {})
    datos = {}

    for col in columnas:
        etiqueta = labels_todas.get(col, col)

        if col in ejemplo_actual:
            valor_inicial = str(ejemplo_actual[col])
        else:
            valor_inicial = ""

        datos[col] = st.text_input(
            etiqueta,
            value=valor_inicial,
            key=f"input_{st.session_state.modelo}_{opcion_ejemplo}_{col}"
        )

    if st.button("Predecir"):

        rellenadas = sum(1 for v in datos.values() if str(v).strip() != "")
        total = len(columnas)
        porcentaje = rellenadas / total

        st.write(f"Variables introducidas: {rellenadas}/{total} ({porcentaje*100:.1f}%)")

        df = pd.DataFrame([datos])

        for c in df.columns:
            df[c] = df[c].apply(limpiar_valor_entrada)

        df = df[columnas]
        faltantes = obtener_variables_faltantes(df, columnas)

        # =====================================================
        # MUY POCOS DATOS: NO PREDICCIÓN
        # =====================================================
        if porcentaje < 0.40:
            st.error("No hay suficientes datos para realizar una predicción fiable.")
            mostrar_aviso_pruebas_recomendadas(faltantes)
            st.stop()

        # =====================================================
        # PREDICCIÓN
        # =====================================================
        proba = pipe.predict_proba(df)[:, 1][0]
        pred = int(proba >= threshold)

        st.write(f"Probabilidad: {proba:.3f}")

        # =====================================================
        # DATOS INCOMPLETOS PERO SUFICIENTES PARA PREDECIR
        # =====================================================
        if porcentaje < 0.60:
            st.warning("La predicción se ha realizado con datos incompletos.")

            margen_incertidumbre = 0.05
            caso_incierto = abs(proba - threshold) < margen_incertidumbre

            if pred == 1:
                st.error("Alto riesgo de progresión.")
                mostrar_aviso_pruebas_recomendadas(faltantes)
            else:
                st.success("Bajo riesgo de progresión.")

                if caso_incierto:
                    mostrar_aviso_pruebas_recomendadas(faltantes)
        else:
            if pred == 1:
                st.error("Alto riesgo de progresión.")
            else:
                st.success("Bajo riesgo de progresión.")

        # =====================================================
        # GRÁFICA SOLO EN MODELO COMPLETO
        # =====================================================
        if st.session_state.modelo == "biomedico":
            st.subheader("Simulación de empeoramiento")
            st.info(
                "La siguiente gráfica muestra cómo cambia el riesgo predicho si empeoran de forma simulada "
                "algunas de las variables más influyentes del modelo."
            )
            graficar_simulacion_empeoramiento(df, pipe, columnas, threshold)