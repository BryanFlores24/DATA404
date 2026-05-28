import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from io import BytesIO

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================

st.set_page_config(
    page_title="DATA404",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# DISEÑO CSS
# =====================================================

st.markdown("""
<style>
/* Fondo general */
.stApp {
    background-color: #0e1117;
}

/* Contenedor principal */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Título principal */
.titulo-principal {
    background: linear-gradient(90deg, #1f4e79, #2e86c1);
    padding: 35px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
}

.titulo-principal h1 {
    color: white;
    font-size: 44px;
    margin-bottom: 8px;
}

.titulo-principal p {
    color: #f0f6ff;
    font-size: 18px;
}

/* Tarjetas */
.card {
    background-color: #161b22;
    color: #f0f6fc;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #30363d;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.35);
    margin-bottom: 24px;
}

/* Textos dentro de tarjetas */
.card h1, .card h2, .card h3, .card p, .card span, .card label {
    color: #f0f6fc !important;
}

/* Subtítulos */
.subtitulo {
    color: #58a6ff !important;
    font-weight: bold;
    font-size: 24px;
}

/* Caja informativa */
.info-box {
    background-color: #102a43;
    color: #e6edf3 !important;
    padding: 16px;
    border-radius: 12px;
    border-left: 6px solid #58a6ff;
    font-size: 17px;
    font-weight: 500;
}

.info-box * {
    color: #e6edf3 !important;
}

/* Botones */
.stButton > button {
    background-color: #238636;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1rem;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #2ea043;
    color: white;
}

/* Botón de descarga */
.stDownloadButton > button {
    background-color: #1f6feb;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1rem;
    font-weight: bold;
}

.stDownloadButton > button:hover {
    background-color: #388bfd;
    color: white;
}

/* Upload */
[data-testid="stFileUploader"] {
    background-color: #161b22;
    border-radius: 14px;
    padding: 12px;
    border: 1px solid #30363d;
}

/* Tablas */
[data-testid="stDataFrame"] {
    background-color: #161b22;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="titulo-principal">
        <h1>📊 DATA404 </h1>
        <p>Ajuste de distribuciones de probabilidad y generación de datos aleatorios para simulación</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <h3>📌 Instrucciones</h3>
    <p>
    Para comenzar, sube un archivo Excel/CSV o ingresa los datos manualmente.
    Luego podrás editar la tabla, seleccionar una columna, generar el histograma,
    ajustar distribuciones y crear nuevos datos aleatorios.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# FUNCIONES
# =====================================================

def limpiar_datos(serie):
    serie = pd.to_numeric(serie, errors="coerce")
    serie = serie.dropna()
    return serie



    distribuciones = {
        "Normal": stats.norm,
        "Exponencial": stats.expon,
        "Uniforme": stats.uniform,
        "Lognormal": stats.lognorm,
        "Gamma": stats.gamma,
        "Weibull": stats.weibull_min,
        "Triangular": stats.triang
    }

    resultados = []

    for nombre, dist in distribuciones.items():
        try:
            parametros = dist.fit(datos)
            ks_stat, p_valor = stats.kstest(datos, dist.name, args=parametros)

            resultados.append({
                "Distribución": nombre,
                "Estadístico KS": ks_stat,
                "P-valor": p_valor,
                "Parámetros": parametros
            })

        except Exception:
            pass

    tabla = pd.DataFrame(resultados)

    if len(tabla) > 0:
        tabla = tabla.sort_values(by="Estadístico KS", ascending=True)

    return tabla

def obtener_distribuciones(tipo="Básicas"):
    basicas = {
        "Normal": stats.norm,
        "Exponencial": stats.expon,
        "Uniforme": stats.uniform,
        "Lognormal": stats.lognorm,
        "Gamma": stats.gamma,
        "Weibull": stats.weibull_min,
        "Triangular": stats.triang
    }

    avanzadas = {
        "Beta": stats.beta,
        "Pareto": stats.pareto,
        "Rayleigh": stats.rayleigh,
        "Cauchy": stats.cauchy,
        "Logística": stats.logistic,
        "Gumbel derecha": stats.gumbel_r,
        "Gumbel izquierda": stats.gumbel_l,
        "Chi-cuadrado": stats.chi2,
        "Student t": stats.t,
        "F": stats.f,
        "Erlang": stats.erlang,
        "Inversa Gaussiana": stats.invgauss,
        "Laplace": stats.laplace,
        "Maxwell": stats.maxwell
    }

    if tipo == "Básicas":
        return basicas
    elif tipo == "Avanzadas":
        return avanzadas
    else:
        return {**basicas, **avanzadas}

def ajustar_distribuciones(datos, criterio="KS", tipo="Básicas"):
    distribuciones = obtener_distribuciones(tipo)

    resultados = []
    n = len(datos)

    for nombre, dist in distribuciones.items():
        try:
            parametros = dist.fit(datos)

            ks_stat, p_valor = stats.kstest(datos, dist.cdf, args=parametros)

            log_likelihood = np.sum(dist.logpdf(datos, *parametros))

            if not np.isfinite(log_likelihood):
                continue

            k = len(parametros)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood

            resultados.append({
                "Distribución": nombre,
                "Estadístico KS": ks_stat,
                "P-valor": p_valor,
                "AIC": aic,
                "BIC": bic,
                "Parámetros": parametros
            })

        except Exception:
            pass

    tabla = pd.DataFrame(resultados)

    if len(tabla) > 0:
        if criterio == "KS":
            tabla = tabla.sort_values(by=["Estadístico KS", "P-valor"], ascending=[True, False])
        elif criterio == "AIC":
            tabla = tabla.sort_values(by="AIC", ascending=True)
        elif criterio == "BIC":
            tabla = tabla.sort_values(by="BIC", ascending=True)

    return tabla

def generar_datos(distribucion_nombre, parametros, cantidad, semilla=None):
    # Usar todas las distribuciones disponibles
    distribuciones = obtener_distribuciones("Todas")

    if semilla is not None:
        np.random.seed(semilla)

    # Esto evita el KeyError
    if distribucion_nombre not in distribuciones:
        st.error(f"La distribución {distribucion_nombre} no está disponible.")
        return []

    dist = distribuciones[distribucion_nombre]
    datos_generados = dist.rvs(*parametros, size=cantidad)
    return datos_generados

def graficar_histograma(datos, titulo):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(datos, bins="auto", density=True, alpha=0.70, edgecolor="black")
    ax.set_title(titulo)
    ax.set_xlabel("Valores")
    ax.set_ylabel("Densidad")
    ax.grid(True, alpha=0.3)
    return fig

def graficar_ajuste_multiple(datos, distribuciones_parametros):
    fig, ax = plt.subplots(figsize=(9, 5))

    # Dibujar histograma de datos originales
    ax.hist(datos, bins="auto", density=True, alpha=0.5, color="skyblue", label="Datos reales", edgecolor="black")

    # Dibujar cada distribución
    for nombre, parametros in distribuciones_parametros:
        dist = obtener_distribuciones("Todas")[nombre]
        
        # Ajustar rango x según la distribución
        if nombre in ["F", "Beta", "Pareto", "Erlang", "Inversa Gaussiana"]:
            x = np.linspace(max(min(datos), 0), max(datos), 400)
        else:
            x = np.linspace(min(datos), max(datos), 400)
        
        try:
            y = dist.pdf(x, *parametros)
        except TypeError:
            y = dist.pdf(x, *list(parametros))
        
        y = np.nan_to_num(y)  # reemplaza NaN e infinitos por 0
        ax.plot(x, y, linewidth=2.5, label=f"{nombre}")

    ax.set_title("Histograma con múltiples distribuciones ajustadas")
    ax.set_xlabel("Valores")
    ax.set_ylabel("Densidad")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig

def convertir_excel(df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Datos editados")

    return output.getvalue()


# =====================================================
# MENÚ LATERAL
# =====================================================

st.sidebar.title("⚙️ Menú de trabajo")

modo = st.sidebar.radio(
    "Selecciona el modo de ingreso:",
    ["Subir archivo Excel/CSV", "Ingresar datos manualmente"]
)

st.sidebar.markdown("---")
st.sidebar.info("Primero carga o ingresa datos. Luego selecciona la columna que deseas analizar.")


# =====================================================
# CARGA Y EDICIÓN DE DATOS
# =====================================================

df_editado = None
datos = None

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="subtitulo">1. Carga y edición de datos</p>', unsafe_allow_html=True)

if modo == "Subir archivo Excel/CSV":

    archivo = st.file_uploader(
        "Sube tu archivo Excel o CSV",
        type=["xlsx", "csv"]
    )

    if archivo is not None:

        if archivo.name.endswith(".xlsx"):
            excel = pd.ExcelFile(archivo)

            hoja = st.selectbox(
                "Selecciona la hoja del Excel:",
                excel.sheet_names
            )

            df = pd.read_excel(archivo, sheet_name=hoja)

        else:
            df = pd.read_csv(archivo)

        st.write("Puedes editar los datos directamente en esta tabla:")

        df_editado = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True
        )

        archivo_editado = convertir_excel(df_editado)

        st.download_button(
            label="📥 Descargar Excel editado",
            data=archivo_editado,
            file_name="datos_editados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        columnas_numericas = df_editado.columns.tolist()

        columna = st.selectbox(
            "Selecciona la columna que deseas analizar:",
            columnas_numericas
        )

        datos = limpiar_datos(df_editado[columna])

else:
    st.write("Pega aquí los datos copiados desde Excel o escríbelos manualmente.")

    texto = st.text_area(
        "Datos manuales:",
        placeholder="""Puedes pegar datos así:
10
12
15
18
20

O también así:
10    12    15    18    20

O separados por coma:
10, 12, 15, 18, 20""",
        height=220
    )

    if texto:
        # Permite pegar datos separados por coma, punto y coma, espacios, tabulaciones o saltos de línea
        texto_limpio = texto.replace(";", ",")
        texto_limpio = texto_limpio.replace("\t", ",")
        texto_limpio = texto_limpio.replace("\n", ",")
        texto_limpio = texto_limpio.replace(" ", ",")

        lista = texto_limpio.split(",")

        datos = limpiar_datos(pd.Series(lista))

        df_manual = pd.DataFrame({
            "Datos ingresados": datos
        })

        st.write("Vista previa de los datos ingresados:")
        st.dataframe(df_manual, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# =====================================================
# ANÁLISIS ESTADÍSTICO
# =====================================================

if datos is not None and len(datos) > 0:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="subtitulo">2. Resumen estadístico</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cantidad", len(datos))
    col2.metric("Media", round(datos.mean(), 4))
    col3.metric("Mínimo", round(datos.min(), 4))
    col4.metric("Máximo", round(datos.max(), 4))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Mediana", round(datos.median(), 4))
    col6.metric("Desv. estándar", round(datos.std(), 4))
    col7.metric("Varianza", round(datos.var(), 4))
    col8.metric("Asimetría", round(stats.skew(datos), 4))

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="subtitulo">3. Histograma de datos</p>', unsafe_allow_html=True)

    st.pyplot(graficar_histograma(datos, "Histograma de datos seleccionados"))

    st.markdown('</div>', unsafe_allow_html=True)


    # =====================================================
    # AJUSTE DE DISTRIBUCIONES
    # =====================================================

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="subtitulo">4. Ajuste automático de distribuciones</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        La aplicación compara varias distribuciones de probabilidad y recomienda la mejor según el estadístico Kolmogorov-Smirnov.
        Mientras menor sea el estadístico KS, mejor es el ajuste.
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    criterio_ajuste = st.selectbox(
    "Selecciona el criterio de evaluación:",
    ["KS", "AIC", "BIC"]
    )

    tipo_distribuciones = st.selectbox(
        "Grupo de distribuciones a probar:",
        ["Básicas", "Avanzadas", "Todas"]
    )

    if st.button("🔍 Ajustar distribuciones"):
        resultados = ajustar_distribuciones(datos, criterio_ajuste, tipo_distribuciones)
        st.session_state["resultados"] = resultados

    if "resultados" in st.session_state:
        resultados = st.session_state["resultados"]

        if len(resultados) > 0:

            tabla_visible = resultados.copy()

            # Ordenar por Estadístico KS ascendente y P-valor descendente
            tabla_visible = tabla_visible.sort_values(by=["Estadístico KS", "P-valor"], ascending=[True, False])

            # Resetear el índice para que fila 0 sea la mejor
            tabla_visible = tabla_visible.reset_index(drop=True)

            tabla_visible["Estadístico KS"] = tabla_visible["Estadístico KS"].round(5)
            tabla_visible["P-valor"] = tabla_visible["P-valor"].round(5)
            tabla_visible["AIC"] = tabla_visible["AIC"].round(4)
            tabla_visible["BIC"] = tabla_visible["BIC"].round(4)
            tabla_visible["Parámetros"] = tabla_visible["Parámetros"].apply(
                lambda x: ", ".join([str(round(valor, 4)) for valor in x])
            )

            st.write("Ranking de distribuciones:")
            st.dataframe(tabla_visible, use_container_width=True)

            # Bloque completo de multiselect y graficado
            distribuciones_seleccionadas = st.multiselect(
                "Selecciona las distribuciones que quieres mostrar en el gráfico",
                options=resultados["Distribución"].tolist(),
                default=[resultados.iloc[0]["Distribución"]]
            )

            # Construir lista segura de tuplas (nombre, parámetros)
            dist_param = []
            for nombre in distribuciones_seleccionadas:
                parametros_raw = resultados[resultados["Distribución"] == nombre]["Parámetros"].iloc[0]
                
                if isinstance(parametros_raw, (np.ndarray, list, tuple)):
                    parametros = tuple(parametros_raw)
                else:
                    parametros = (parametros_raw,)
                
                dist_param.append((nombre, parametros))

            # Graficar
            st.pyplot(graficar_ajuste_multiple(datos, dist_param))

            mejor = resultados.iloc[0]

            st.success(f"Distribución recomendada: {mejor['Distribución']}")

            st.write("Parámetros estimados:")
            st.write(mejor["Parámetros"])


        else:
            st.error("No se pudo ajustar ninguna distribución.")

    st.markdown('</div>', unsafe_allow_html=True)


    # =====================================================
    # GENERACIÓN DE DATOS ALEATORIOS
    # =====================================================

    if "resultados" in st.session_state and len(st.session_state["resultados"]) > 0:

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="subtitulo">5. Generación de datos aleatorios</p>', unsafe_allow_html=True)

        resultados = st.session_state["resultados"]

        distribucion_elegida = st.selectbox(
            "Selecciona la distribución para generar datos:",
            resultados["Distribución"].tolist()
        )

        parametros_elegidos = resultados[
            resultados["Distribución"] == distribucion_elegida
        ]["Parámetros"].iloc[0]

        col_a, col_b = st.columns(2)

        cantidad = col_a.number_input(
            "Cantidad de datos a generar:",
            min_value=1,
            max_value=100000,
            value=1000
        )

        usar_semilla = col_b.checkbox("Usar semilla aleatoria")

        semilla = None

        if usar_semilla:
            semilla = col_b.number_input(
                "Semilla:",
                min_value=1,
                max_value=999999,
                value=123
            )

        if st.button("🎲 Generar datos aleatorios"):
            datos_generados = generar_datos(
                distribucion_elegida,
                parametros_elegidos,
                cantidad,
                semilla
            )

            df_generados = pd.DataFrame({
                "Datos generados": datos_generados
            })

            st.write("Vista previa de los datos generados:")
            st.dataframe(df_generados.head(30), use_container_width=True)

            st.pyplot(
                graficar_histograma(
                    df_generados["Datos generados"],
                    "Histograma de datos generados"
                )
            )

            csv = df_generados.to_csv(index=False).encode("utf-8")
            excel_generado = convertir_excel(df_generados)

            # Convertir datos a diferentes formatos
            txt = df_generados.to_csv(index=False, header=False, sep="\t").encode("utf-8")

            col_desc1, col_desc2, col_desc3 = st.columns(3)

            col_desc1.download_button(
                "📥 Descargar CSV",
                csv,
                "datos_generados.csv",
                "text/csv"
            )

            col_desc2.download_button(
                "📥 Descargar Excel",
                excel_generado,
                "datos_generados.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            col_desc3.download_button(
                "📥 Descargar TXT",
                txt,
                "datos_generados.txt",
                "text/plain"
            )

            st.write("También puedes copiar los datos desde aquí:")

            texto_copiable = df_generados.to_csv(
                index=False,
                header=False,
                sep="\t"
            )

            st.text_area(
                "Datos generados para copiar:",
                value=texto_copiable,
                height=250
            )

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="card">
        <h3>📌 Instrucciones</h3>
        <p>
        Para comenzar, sube un archivo Excel/CSV o ingresa los datos manualmente.
        Luego podrás editar la tabla, seleccionar una columna, generar el histograma,
        ajustar distribuciones y crear nuevos datos aleatorios.
        </p>
    </div>
    """, unsafe_allow_html=True)