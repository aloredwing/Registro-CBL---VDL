# app_cbl_visualizador.py
# Visualizador CBL para archivos .LAS, .CSV y .XLSX
# Autor: ChatGPT
# Uso:
#   pip install -r requirements_cbl_visualizador.txt
#   streamlit run app_cbl_visualizador.py

import io
import re
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import lasio
except ImportError:
    lasio = None


# =========================================================
# CONFIGURACION GENERAL
# =========================================================
st.set_page_config(
    page_title="Visualizador CBL - Registro de Cemento",
    page_icon="🛢️",
    layout="wide",
)


# =========================================================
# FUNCIONES DE LECTURA
# =========================================================
def clean_column_name(col) -> str:
    """Normaliza nombres de columnas para facilitar deteccion."""
    col = str(col).strip()
    col = re.sub(r"\s+", "_", col)
    return col


def read_las_file(uploaded_file) -> pd.DataFrame:
    """Lee un archivo LAS y devuelve un DataFrame."""
    if lasio is None:
        raise ImportError("Falta instalar lasio. Ejecuta: pip install lasio")

    raw = uploaded_file.read()

    # Intentos comunes de decodificacion
    text = None
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        raise ValueError("No se pudo decodificar el archivo LAS.")

    las = lasio.read(io.StringIO(text))
    df = las.df().reset_index()

    # Renombrar la primera columna como la primera curva del LAS si aplica
    if len(las.curves) > 0:
        first_curve = las.curves[0].mnemonic
        if df.columns[0].lower() in ["index", "dept", "depth"]:
            df = df.rename(columns={df.columns[0]: first_curve})

    df.columns = [clean_column_name(c) for c in df.columns]
    df = df.replace([-999.25, -999.0, -9999, -9999.0], np.nan)
    return df


def read_csv_file(uploaded_file) -> pd.DataFrame:
    """Lee CSV intentando detectar separador."""
    raw = uploaded_file.read()
    text = None
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        raise ValueError("No se pudo decodificar el CSV.")

    # sep=None intenta inferir coma, punto y coma, tab, etc.
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    df.columns = [clean_column_name(c) for c in df.columns]
    return df


def read_excel_file(uploaded_file) -> pd.DataFrame:
    """Lee Excel. Por defecto toma la primera hoja."""
    df = pd.read_excel(uploaded_file)
    df.columns = [clean_column_name(c) for c in df.columns]
    return df


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Carga archivo segun extension."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".las"):
        return read_las_file(uploaded_file)
    if filename.endswith(".csv"):
        return read_csv_file(uploaded_file)
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return read_excel_file(uploaded_file)

    raise ValueError("Formato no soportado. Usa .las, .csv, .xlsx o .xls")


# =========================================================
# DETECCION AUTOMATICA DE COLUMNAS
# =========================================================
def find_first_matching_column(columns: List[str], patterns: List[str]) -> Optional[str]:
    """Busca la primera columna que contenga alguno de los patrones."""
    upper_cols = {c: c.upper() for c in columns}
    for pattern in patterns:
        pattern = pattern.upper()
        for original, upper in upper_cols.items():
            if pattern in upper:
                return original
    return None


def detect_columns(df: pd.DataFrame) -> dict:
    """Detecta columnas tipicas para CBL."""
    columns = list(df.columns)

    depth_col = find_first_matching_column(
        columns,
        ["DEPTH", "DEPT", "MD", "TVD", "PROF", "PROFUNDIDAD"]
    )

    amp3_col = find_first_matching_column(
        columns,
        ["AMP3FT", "AMP_3FT", "CBL", "AMP", "AMPLITUDE", "AMPLITUD"]
    )

    amp5_col = find_first_matching_column(
        columns,
        ["AMP5FT", "AMP_5FT"]
    )

    gr_col = find_first_matching_column(
        columns,
        ["GR", "GK", "GAMMA"]
    )

    tt_col = find_first_matching_column(
        columns,
        ["TT3FT", "TT", "TRAVEL"]
    )

    ccl_col = find_first_matching_column(
        columns,
        ["CCL", "COLLAR"]
    )

    return {
        "depth": depth_col,
        "amp3": amp3_col,
        "amp5": amp5_col,
        "gr": gr_col,
        "tt": tt_col,
        "ccl": ccl_col,
    }


# =========================================================
# PROCESAMIENTO CBL
# =========================================================
def to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def classify_cbl_value(value: float, good_max: float, regular_max: float,
                       alert_max: float, bad_min: float, critical_min: float) -> str:
    """Clasificacion cualitativa por amplitud CBL."""
    if pd.isna(value):
        return "Sin dato"
    if value >= critical_min:
        return "Critico"
    if value >= bad_min:
        return "Mala adherencia"
    if value > regular_max and value < bad_min:
        return "Alerta"
    if value > good_max and value <= regular_max:
        return "Regular"
    return "Bueno"


def prepare_cbl_dataframe(df: pd.DataFrame, depth_col: str, cbl_col: str,
                          good_max: float, regular_max: float,
                          alert_max: float, bad_min: float,
                          critical_min: float) -> pd.DataFrame:
    """Limpia y clasifica el DataFrame para el analisis CBL."""
    out = df.copy()
    out[depth_col] = to_numeric_series(out[depth_col])
    out[cbl_col] = to_numeric_series(out[cbl_col])
    out = out.dropna(subset=[depth_col, cbl_col]).sort_values(depth_col).reset_index(drop=True)

    out["CBL_clasificacion"] = out[cbl_col].apply(
        lambda x: classify_cbl_value(x, good_max, regular_max, alert_max, bad_min, critical_min)
    )
    return out


def estimate_depth_step(depth: pd.Series) -> float:
    diffs = depth.diff().abs().dropna()
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.5
    return float(diffs.median())


def find_bad_bond_intervals(df: pd.DataFrame, depth_col: str, cbl_col: str,
                            threshold: float, min_length_ft: float) -> pd.DataFrame:
    """Encuentra intervalos continuos donde CBL supera el umbral indicado."""
    if df.empty:
        return pd.DataFrame()

    data = df[[depth_col, cbl_col]].dropna().sort_values(depth_col).reset_index(drop=True)
    step = estimate_depth_step(data[depth_col])
    max_gap = max(step * 2.5, 0.75)

    intervals = []
    in_interval = False
    start_depth = None
    values = []
    last_depth = None

    for _, row in data.iterrows():
        depth = float(row[depth_col])
        amp = float(row[cbl_col])
        is_bad = amp >= threshold

        # Si hay salto fuerte en profundidad, cerrar intervalo previo
        if in_interval and last_depth is not None and abs(depth - last_depth) > max_gap:
            end_depth = last_depth
            length = abs(end_depth - start_depth)
            if length >= min_length_ft:
                intervals.append({
                    "Desde_ft": min(start_depth, end_depth),
                    "Hasta_ft": max(start_depth, end_depth),
                    "Longitud_ft": length,
                    "CBL_max_mV": np.nanmax(values),
                    "CBL_prom_mV": np.nanmean(values),
                    "N_puntos": len(values),
                })
            in_interval = False
            start_depth = None
            values = []

        if is_bad:
            if not in_interval:
                in_interval = True
                start_depth = depth
                values = [amp]
            else:
                values.append(amp)
        else:
            if in_interval:
                end_depth = last_depth if last_depth is not None else depth
                length = abs(end_depth - start_depth)
                if length >= min_length_ft:
                    intervals.append({
                        "Desde_ft": min(start_depth, end_depth),
                        "Hasta_ft": max(start_depth, end_depth),
                        "Longitud_ft": length,
                        "CBL_max_mV": np.nanmax(values),
                        "CBL_prom_mV": np.nanmean(values),
                        "N_puntos": len(values),
                    })
                in_interval = False
                start_depth = None
                values = []

        last_depth = depth

    # Cerrar intervalo al final
    if in_interval and start_depth is not None and last_depth is not None:
        end_depth = last_depth
        length = abs(end_depth - start_depth)
        if length >= min_length_ft:
            intervals.append({
                "Desde_ft": min(start_depth, end_depth),
                "Hasta_ft": max(start_depth, end_depth),
                "Longitud_ft": length,
                "CBL_max_mV": np.nanmax(values),
                "CBL_prom_mV": np.nanmean(values),
                "N_puntos": len(values),
            })

    result = pd.DataFrame(intervals)
    if not result.empty:
        result = result.sort_values(["CBL_max_mV", "Longitud_ft"], ascending=[False, False]).reset_index(drop=True)
    return result


# =========================================================
# GRAFICAS
# =========================================================
def make_2d_tracks(df: pd.DataFrame, depth_col: str, cbl_col: str,
                   gr_col: Optional[str], amp5_col: Optional[str],
                   tt_col: Optional[str], ccl_col: Optional[str],
                   bad_min: float, critical_min: float):
    """Genera tracks 2D similares a un registro de pozo."""
    tracks = []

    if gr_col and gr_col in df.columns:
        tracks.append((gr_col, "GR / GK"))

    tracks.append((cbl_col, "CBL principal"))

    if amp5_col and amp5_col in df.columns and amp5_col != cbl_col:
        tracks.append((amp5_col, "AMP 5FT"))

    if tt_col and tt_col in df.columns:
        tracks.append((tt_col, "Travel Time"))

    if ccl_col and ccl_col in df.columns:
        tracks.append((ccl_col, "CCL"))

    fig = make_subplots(
        rows=1,
        cols=len(tracks),
        shared_yaxes=True,
        horizontal_spacing=0.025,
        subplot_titles=[t[1] for t in tracks],
    )

    for idx, (col, name) in enumerate(tracks, start=1):
        y = df[depth_col]
        x = pd.to_numeric(df[col], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                hovertemplate=f"{col}: %{{x}}<br>Profundidad: %{{y}} ft<extra></extra>",
            ),
            row=1,
            col=idx,
        )

        if col == cbl_col:
            fig.add_vline(x=bad_min, line_dash="dash", annotation_text="Mala", row=1, col=idx)
            fig.add_vline(x=critical_min, line_dash="dot", annotation_text="Crítico", row=1, col=idx)

    fig.update_yaxes(title_text="Profundidad, ft", autorange="reversed", row=1, col=1)
    fig.update_layout(
        height=820,
        title="Registro CBL / Cemento - Vista 2D",
        showlegend=False,
        margin=dict(l=20, r=20, t=80, b=20),
    )
    return fig


def make_cbl_3d_cylinder(df: pd.DataFrame, depth_col: str, cbl_col: str,
                         bad_min: float, critical_min: float,
                         max_points: int = 1500):
    """Genera vista pseudo 3D como cilindro coloreado por amplitud CBL."""
    data = df[[depth_col, cbl_col]].dropna().sort_values(depth_col).copy()

    # Reducir puntos si el archivo es muy grande
    if len(data) > max_points:
        data = data.iloc[np.linspace(0, len(data) - 1, max_points).astype(int)]

    depth = data[depth_col].to_numpy(dtype=float)
    amp = data[cbl_col].to_numpy(dtype=float)

    theta = np.linspace(0, 2 * np.pi, 48)

    # Radio base del casing. Se deforma levemente con amplitud solo para visualizar severidad.
    amp_min = np.nanmin(amp)
    amp_max = np.nanmax(amp)
    denom = amp_max - amp_min if amp_max != amp_min else 1.0
    amp_norm = (amp - amp_min) / denom
    radius = 1.0 + 0.20 * amp_norm

    R = np.outer(radius, np.ones_like(theta))
    X = R * np.cos(theta)
    Y = R * np.sin(theta)
    Z = np.outer(depth, np.ones_like(theta))
    C = np.outer(amp, np.ones_like(theta))

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                surfacecolor=C,
                colorscale="Turbo",
                colorbar=dict(title=f"{cbl_col}, mV"),
                hovertemplate=(
                    "Profundidad: %{z:.2f} ft<br>"
                    f"{cbl_col}: %{{surfacecolor:.2f}} mV<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title="Vista pseudo 3D del CBL: severidad de adherencia alrededor del casing",
        height=850,
        scene=dict(
            xaxis=dict(title="X", showticklabels=False),
            yaxis=dict(title="Y", showticklabels=False),
            zaxis=dict(title="Profundidad, ft", autorange="reversed"),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=3.3),
        ),
        margin=dict(l=10, r=10, t=70, b=10),
    )

    return fig


def make_class_distribution(df: pd.DataFrame):
    """Grafica cantidad de puntos por clasificacion."""
    counts = df["CBL_clasificacion"].value_counts().reset_index()
    counts.columns = ["Clasificacion", "Puntos"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=counts["Clasificacion"],
            y=counts["Puntos"],
            text=counts["Puntos"],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Distribución de puntos por clasificación CBL",
        xaxis_title="Clasificación",
        yaxis_title="Cantidad de puntos",
        height=420,
    )
    return fig


# =========================================================
# INTERFAZ PRINCIPAL
# =========================================================
st.title("🛢️ Visualizador CBL - Registro de Cemento")
st.caption("Carga un archivo .LAS, .CSV o .XLSX para revisar amplitud CBL, intervalos sospechosos y vista pseudo 3D.")

with st.expander("📌 ¿Qué hace esta aplicación?", expanded=True):
    st.markdown(
        """
        Esta aplicación permite revisar el **registro de cemento CBL** usando la curva de amplitud.
        La interpretación se basa en que una amplitud baja suele asociarse a mejor acoplamiento casing-cemento-formación,
        mientras que una amplitud alta puede indicar pobre adherencia, canalización, cemento deficiente o tendencia a free pipe.

        La gráfica 3D es una **vista pseudo 3D**: proyecta la severidad del CBL alrededor de un cilindro que representa el casing.
        No muestra azimut real porque el archivo CBL convencional no trae orientación radial. Por eso, la gráfica sirve para ubicar
        **profundidades críticas**, no para decir en qué lado exacto del casing está el problema.
        """
    )

st.markdown("### Constraints / límites operativos sugeridos")
st.info(
    "Estos límites son referenciales para screening operativo. Deben ajustarse con la calibración del servicio, tamaño de casing, tipo de herramienta, presión, fluido, centralización y reporte final de la compañía de registros."
)

constraints_df = pd.DataFrame(
    [
        {"Rango CBL": "≤ 3 mV", "Clasificación": "Bueno", "Uso operativo": "Cemento/adherencia aceptable relativa."},
        {"Rango CBL": "> 3 y ≤ 8 mV", "Clasificación": "Regular", "Uso operativo": "Zona a observar; no concluir falla solo con este dato."},
        {"Rango CBL": "> 8 y < 10 mV", "Clasificación": "Alerta", "Uso operativo": "Posible pérdida parcial de adherencia."},
        {"Rango CBL": "≥ 10 mV", "Clasificación": "Mala adherencia", "Uso operativo": "Zona sospechosa; revisar continuidad y curvas auxiliares."},
        {"Rango CBL": "≥ 15 mV", "Clasificación": "Crítico", "Uso operativo": "Alta amplitud; posible pobre adherencia severa o tendencia a free pipe."},
        {"Rango CBL": "Longitud ≥ 5 ft", "Clasificación": "Criterio operativo", "Uso operativo": "Prioriza tramos continuos y evita sobrerreaccionar a picos aislados."},
    ]
)
st.dataframe(constraints_df, use_container_width=True, hide_index=True)

uploaded_file = st.file_uploader(
    "Carga tu archivo de registro",
    type=["las", "csv", "xlsx", "xls"],
    help="Puede ser LAS original, CSV exportado o Excel con columnas de profundidad y amplitud CBL."
)

if uploaded_file is None:
    st.warning("Carga un archivo para iniciar el análisis.")
    st.stop()

try:
    df_raw = load_uploaded_file(uploaded_file)
except Exception as e:
    st.error(f"No se pudo leer el archivo: {e}")
    st.stop()

if df_raw.empty:
    st.error("El archivo se leyó, pero no contiene datos.")
    st.stop()

# Detectar columnas
suggested = detect_columns(df_raw)
columns = list(df_raw.columns)

st.sidebar.header("Configuración del análisis")

def select_col(label: str, suggested_col: Optional[str], allow_none: bool = False):
    options = ["Ninguno"] + columns if allow_none else columns
    if suggested_col in columns:
        index = options.index(suggested_col)
    else:
        index = 0
    selected = st.sidebar.selectbox(label, options, index=index)
    if selected == "Ninguno":
        return None
    return selected

# Si no detecta, toma primera columna para profundidad y segunda para CBL como respaldo
if suggested["depth"] is None and len(columns) > 0:
    suggested["depth"] = columns[0]
if suggested["amp3"] is None and len(columns) > 1:
    suggested["amp3"] = columns[1]

depth_col = select_col("Columna de profundidad", suggested["depth"], allow_none=False)
cbl_col = select_col("Columna CBL principal", suggested["amp3"], allow_none=False)
gr_col = select_col("Columna GR / GK", suggested["gr"], allow_none=True)
amp5_col = select_col("Columna AMP5FT / secundaria", suggested["amp5"], allow_none=True)
tt_col = select_col("Columna TT / Travel Time", suggested["tt"], allow_none=True)
ccl_col = select_col("Columna CCL", suggested["ccl"], allow_none=True)

st.sidebar.subheader("Constraints CBL")
good_max = st.sidebar.number_input("Bueno hasta, mV", value=3.0, step=0.5)
regular_max = st.sidebar.number_input("Regular hasta, mV", value=8.0, step=0.5)
alert_max = st.sidebar.number_input("Alerta hasta, mV", value=10.0, step=0.5)
bad_min = st.sidebar.number_input("Mala adherencia desde, mV", value=10.0, step=0.5)
critical_min = st.sidebar.number_input("Crítico desde, mV", value=15.0, step=0.5)
min_length_ft = st.sidebar.number_input("Longitud mínima operativa, ft", value=5.0, min_value=0.0, step=0.5)

st.sidebar.subheader("Filtro de profundidad")

try:
    temp_depth = pd.to_numeric(df_raw[depth_col], errors="coerce").dropna()
except Exception:
    st.error("La columna de profundidad seleccionada no se puede convertir a número.")
    st.stop()

if temp_depth.empty:
    st.error("No hay valores numéricos válidos en la columna de profundidad.")
    st.stop()

min_depth = float(temp_depth.min())
max_depth = float(temp_depth.max())
selected_depth_range = st.sidebar.slider(
    "Intervalo de profundidad, ft",
    min_value=min_depth,
    max_value=max_depth,
    value=(min_depth, max_depth),
)

# Preparar datos
try:
    df = prepare_cbl_dataframe(
        df_raw,
        depth_col=depth_col,
        cbl_col=cbl_col,
        good_max=good_max,
        regular_max=regular_max,
        alert_max=alert_max,
        bad_min=bad_min,
        critical_min=critical_min,
    )
except Exception as e:
    st.error(f"Error preparando los datos: {e}")
    st.stop()

df = df[(df[depth_col] >= selected_depth_range[0]) & (df[depth_col] <= selected_depth_range[1])].copy()

if df.empty:
    st.error("No hay datos dentro del intervalo de profundidad seleccionado.")
    st.stop()

# Convertir opcionales a numerico cuando existan
for optional_col in [gr_col, amp5_col, tt_col, ccl_col]:
    if optional_col and optional_col in df.columns:
        df[optional_col] = pd.to_numeric(df[optional_col], errors="coerce")

# Métricas principales
bad_intervals = find_bad_bond_intervals(
    df,
    depth_col=depth_col,
    cbl_col=cbl_col,
    threshold=bad_min,
    min_length_ft=min_length_ft,
)

critical_intervals = find_bad_bond_intervals(
    df,
    depth_col=depth_col,
    cbl_col=cbl_col,
    threshold=critical_min,
    min_length_ft=0.5,
)

st.markdown("---")
st.subheader("Resumen rápido del archivo")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Profundidad mínima", f"{df[depth_col].min():.2f} ft")
with col2:
    st.metric("Profundidad máxima", f"{df[depth_col].max():.2f} ft")
with col3:
    st.metric("CBL máximo", f"{df[cbl_col].max():.2f} mV")
with col4:
    st.metric("Intervalos operativos malos", len(bad_intervals))

if not bad_intervals.empty:
    main_interval = bad_intervals.iloc[0]
    st.success(
        f"Principal intervalo sospechoso: {main_interval['Desde_ft']:.2f} - {main_interval['Hasta_ft']:.2f} ft "
        f"| Longitud: {main_interval['Longitud_ft']:.2f} ft "
        f"| CBL máx: {main_interval['CBL_max_mV']:.2f} mV"
    )
else:
    st.warning("No se encontraron intervalos continuos que superen el umbral operativo configurado.")

# Tabs de visualización
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Registro 2D",
    "🧱 Vista pseudo 3D",
    "📍 Intervalos",
    "📊 Clasificación",
    "🧾 Datos",
])

with tab1:
    st.markdown(
        """
        Esta vista permite comparar el CBL principal con curvas auxiliares como GR, AMP5FT, TT y CCL.
        Para interpretación operativa, no se recomienda usar solo un pico aislado de amplitud; debe revisarse continuidad,
        coherencia con otras curvas y contexto mecánico del pozo.
        """
    )
    fig_2d = make_2d_tracks(df, depth_col, cbl_col, gr_col, amp5_col, tt_col, ccl_col, bad_min, critical_min)
    st.plotly_chart(fig_2d, use_container_width=True)

with tab2:
    st.markdown(
        """
        La pseudo 3D muestra el casing como un cilindro. El color representa la amplitud CBL en cada profundidad.
        Como el registro no trae información azimutal, la severidad se replica alrededor de todo el cilindro.
        Por eso esta gráfica responde principalmente: **¿a qué profundidad hay mayor sospecha de mala adherencia?**
        """
    )
    fig_3d = make_cbl_3d_cylinder(df, depth_col, cbl_col, bad_min, critical_min)
    st.plotly_chart(fig_3d, use_container_width=True)

with tab3:
    st.markdown("### Intervalos con mala adherencia según el umbral operativo")
    st.caption(f"Criterio actual: {cbl_col} ≥ {bad_min} mV y longitud ≥ {min_length_ft} ft")
    if bad_intervals.empty:
        st.info("No hay intervalos operativos con los criterios actuales.")
    else:
        st.dataframe(bad_intervals, use_container_width=True, hide_index=True)

    st.markdown("### Eventos críticos cortos")
    st.caption(f"Criterio actual: {cbl_col} ≥ {critical_min} mV y longitud mínima 0.5 ft")
    if critical_intervals.empty:
        st.info("No hay eventos críticos con los criterios actuales.")
    else:
        st.dataframe(critical_intervals, use_container_width=True, hide_index=True)

with tab4:
    fig_dist = make_class_distribution(df)
    st.plotly_chart(fig_dist, use_container_width=True)

    class_table = df["CBL_clasificacion"].value_counts().rename_axis("Clasificacion").reset_index(name="Puntos")
    st.dataframe(class_table, use_container_width=True, hide_index=True)

with tab5:
    st.markdown("### Datos procesados")
    st.dataframe(df, use_container_width=True)

    csv_export = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar datos procesados en CSV",
        data=csv_export,
        file_name="cbl_datos_procesados.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Nota: Este visualizador es para screening técnico. Para decisiones de cementación, aislamiento o workover, validar con el reporte oficial, calibración de herramienta, condiciones de pozo, centralización, presión, fluido y registros complementarios."
)
