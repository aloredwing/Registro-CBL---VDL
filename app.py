import io
import re
import zipfile
from pathlib import Path

import lasio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Visualizador CBL 3D",
    page_icon="🛢️",
    layout="wide",
)

# ------------------------------------------------------------
# Utilidades de lectura
# ------------------------------------------------------------

def normalize_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def load_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".las":
        raw = uploaded_file.getvalue()
        text = raw.decode("utf-8", errors="ignore")
        las = lasio.read(io.StringIO(text))
        df = las.df().reset_index()
        df.columns = [str(c).strip() for c in df.columns]
        return df

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded_file)

    if suffix == ".csv":
        raw = uploaded_file.getvalue()
        text = raw.decode("utf-8", errors="ignore")
        return pd.read_csv(io.StringIO(text), sep=None, engine="python")

    raise ValueError("Formato no soportado. Usa LAS, CSV, XLSX o XLS.")


def guess_depth_column(df: pd.DataFrame):
    preferred = ["DEPT", "DEPTH", "MD", "TVD", "PROFUNDIDAD", "PROF"]
    normalized = {col: normalize_name(col) for col in df.columns}
    for key in preferred:
        for col, ncol in normalized.items():
            if key == ncol or key in ncol:
                return col
    return df.columns[0]


def guess_cbl_column(df: pd.DataFrame):
    preferred_exact = ["AMP3FT", "CBL", "CBL3FT", "AMPLITUDE", "AMPLITUD", "AMP"]
    normalized = {col: normalize_name(col) for col in df.columns}

    for key in preferred_exact:
        for col, ncol in normalized.items():
            if key == ncol:
                return col

    for key in ["AMP3", "CBL", "AMPL", "AMP"]:
        for col, ncol in normalized.items():
            if key in ncol:
                return col

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]


def find_optional_column(df: pd.DataFrame, keys):
    normalized = {col: normalize_name(col) for col in df.columns}
    for key in keys:
        for col, ncol in normalized.items():
            if normalize_name(key) == ncol or normalize_name(key) in ncol:
                return col
    return None


def clean_data(df: pd.DataFrame, depth_col: str, cbl_col: str, gr_col=None, ccl_col=None, tt_col=None):
    cols = [depth_col, cbl_col]
    for col in [gr_col, ccl_col, tt_col]:
        if col and col not in cols:
            cols.append(col)

    out = df[cols].copy()
    rename = {depth_col: "DEPTH", cbl_col: "CBL"}
    if gr_col:
        rename[gr_col] = "GR"
    if ccl_col:
        rename[ccl_col] = "CCL"
    if tt_col:
        rename[tt_col] = "TT"
    out = out.rename(columns=rename)

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["DEPTH", "CBL"])
    out = out.sort_values("DEPTH").drop_duplicates(subset=["DEPTH"])
    out = out.reset_index(drop=True)
    return out


# ------------------------------------------------------------
# Interpretación
# ------------------------------------------------------------

def classify_cbl(value, good_lim, regular_lim, bad_lim, critical_lim):
    if value <= good_lim:
        return "Bueno"
    if value <= regular_lim:
        return "Regular"
    if value < bad_lim:
        return "Alerta"
    if value < critical_lim:
        return "Mala adherencia"
    return "Crítico"


def severity_from_cbl(cbl, bad_lim, critical_lim):
    sev = (cbl - bad_lim) / max(critical_lim - bad_lim, 0.001)
    return np.clip(sev, 0, 1)


def detect_intervals(df: pd.DataFrame, limit: float, min_len_ft: float):
    if df.empty:
        return pd.DataFrame()

    work = df[["DEPTH", "CBL", "CLASS"]].copy()
    work["flag"] = work["CBL"] >= limit

    if len(work) > 1:
        sample_step = float(np.nanmedian(np.abs(np.diff(work["DEPTH"]))))
        if not np.isfinite(sample_step) or sample_step <= 0:
            sample_step = 0.5
    else:
        sample_step = 0.5

    max_gap = max(sample_step * 2.5, 1.0)
    groups = []
    active = False
    start_idx = None
    prev_depth = None

    for idx, row in work.iterrows():
        depth = float(row["DEPTH"])
        flag = bool(row["flag"])

        if flag and not active:
            active = True
            start_idx = idx
        elif flag and active and prev_depth is not None and abs(depth - prev_depth) > max_gap:
            groups.append((start_idx, idx - 1))
            start_idx = idx
        elif not flag and active:
            groups.append((start_idx, idx - 1))
            active = False
            start_idx = None

        if flag:
            prev_depth = depth

    if active and start_idx is not None:
        groups.append((start_idx, len(work) - 1))

    rows = []
    for start_idx, end_idx in groups:
        block = work.iloc[start_idx:end_idx + 1]
        start = float(block["DEPTH"].min())
        end = float(block["DEPTH"].max())
        length = abs(end - start)
        max_amp = float(block["CBL"].max())
        mean_amp = float(block["CBL"].mean())
        rows.append({
            "Desde_ft": round(start, 2),
            "Hasta_ft": round(end, 2),
            "Longitud_ft": round(length, 2),
            "CBL_max_mV": round(max_amp, 2),
            "CBL_prom_mV": round(mean_amp, 2),
            "Condición": "Operativo" if length >= min_len_ft else "Evento corto",
            "Interpretación": classify_cbl(max_amp, 3, 8, 10, 15),
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Gráficos
# ------------------------------------------------------------

def make_log_plot(df, good_lim, regular_lim, bad_lim, critical_lim):
    has_gr = "GR" in df.columns
    has_ccl = "CCL" in df.columns
    has_tt = "TT" in df.columns

    titles = []
    if has_gr:
        titles.append("GR")
    titles.append("CBL amplitud")
    if has_tt:
        titles.append("TT")
    if has_ccl:
        titles.append("CCL")

    fig = make_subplots(rows=1, cols=len(titles), shared_yaxes=True, horizontal_spacing=0.03, subplot_titles=titles)
    col = 1

    if has_gr:
        fig.add_trace(go.Scatter(x=df["GR"], y=df["DEPTH"], mode="lines", name="GR"), row=1, col=col)
        fig.update_xaxes(title_text="GR", row=1, col=col)
        col += 1

    fig.add_trace(go.Scatter(x=df["CBL"], y=df["DEPTH"], mode="lines", name="CBL mV"), row=1, col=col)
    for x, label in [(good_lim, "Bueno"), (regular_lim, "Regular"), (bad_lim, "Mala adherencia"), (critical_lim, "Crítico")]:
        fig.add_vline(x=x, line_width=1, line_dash="dash", annotation_text=label, row=1, col=col)
    fig.update_xaxes(title_text="mV", row=1, col=col)
    col += 1

    if has_tt:
        fig.add_trace(go.Scatter(x=df["TT"], y=df["DEPTH"], mode="lines", name="TT"), row=1, col=col)
        fig.update_xaxes(title_text="TT", row=1, col=col)
        col += 1

    if has_ccl:
        fig.add_trace(go.Scatter(x=df["CCL"], y=df["DEPTH"], mode="lines", name="CCL"), row=1, col=col)
        fig.update_xaxes(title_text="CCL", row=1, col=col)

    fig.update_yaxes(autorange="reversed", title_text="Profundidad, ft")
    fig.update_layout(height=750, showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def make_2d_schematic(df, good_lim, regular_lim, bad_lim, critical_lim):
    plot_df = df.copy()
    plot_df["SEV"] = severity_from_cbl(plot_df["CBL"], bad_lim, critical_lim)

    z = plot_df["DEPTH"].to_numpy()
    sev = plot_df["SEV"].to_numpy()
    cbl = plot_df["CBL"].to_numpy()

    # Matriz lateral: izquierda y derecha representan el anular esquemático.
    x = np.linspace(-1.4, 1.4, 80)
    image = np.zeros((len(z), len(x)))

    for i, s in enumerate(sev):
        for j, xx in enumerate(x):
            in_casing = abs(xx) <= 0.34
            annulus = 0.34 < abs(xx) <= 1.00
            formation = abs(xx) > 1.00

            if in_casing:
                image[i, j] = -0.25
            elif annulus:
                image[i, j] = s
            elif formation:
                image[i, j] = -0.55

    colorscale = [
        [0.00, "#f1f1f1"],
        [0.20, "#f1f1f1"],
        [0.21, "#666666"],
        [0.32, "#666666"],
        [0.33, "#1f78b4"],
        [0.48, "#35a853"],
        [0.65, "#f2c94c"],
        [0.82, "#f2994a"],
        [1.00, "#d7191c"],
    ]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], shared_yaxes=True,
                        horizontal_spacing=0.02, subplot_titles=["Esquema casing cemento formación", "Curva CBL"])

    # Reescalado para color porque image tiene valores negativos.
    color_data = (image + 0.55) / 1.55
    fig.add_trace(go.Heatmap(
        x=x,
        y=z,
        z=color_data,
        colorscale=colorscale,
        showscale=False,
        hovertemplate="Profundidad: %{y:.2f} ft<br>Severidad relativa: %{z:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_vline(x=-0.34, line_width=3, line_color="black", row=1, col=1)
    fig.add_vline(x=0.34, line_width=3, line_color="black", row=1, col=1)
    fig.add_vline(x=-1.00, line_width=1, line_dash="dot", line_color="gray", row=1, col=1)
    fig.add_vline(x=1.00, line_width=1, line_dash="dot", line_color="gray", row=1, col=1)

    fig.add_trace(go.Scatter(x=cbl, y=z, mode="lines", name="CBL"), row=1, col=2)
    for xline, label in [(good_lim, "Bueno"), (regular_lim, "Regular"), (bad_lim, "Mala adherencia"), (critical_lim, "Crítico")]:
        fig.add_vline(x=xline, line_width=1, line_dash="dash", annotation_text=label, row=1, col=2)

    fig.update_yaxes(autorange="reversed", title_text="Profundidad, ft")
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text="CBL, mV", row=1, col=2)
    fig.update_layout(height=850, margin=dict(l=20, r=20, t=70, b=20), showlegend=False)
    return fig


def cylinder_surface(radius, z_values, n_theta=72, color_value=None):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    z = np.asarray(z_values)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = radius * np.cos(theta_grid)
    y = radius * np.sin(theta_grid)
    if color_value is None:
        color_value = np.zeros_like(z_grid)
    return x, y, z_grid, color_value


def make_3d_cement_model(df, bad_lim, critical_lim, max_points=700, sector_mode=False, sector_degrees=90):
    plot_df = df[["DEPTH", "CBL"]].dropna().copy()
    if len(plot_df) > max_points:
        idx = np.linspace(0, len(plot_df) - 1, max_points).astype(int)
        plot_df = plot_df.iloc[idx].copy()

    depth = plot_df["DEPTH"].to_numpy()
    cbl = plot_df["CBL"].to_numpy()
    severity = severity_from_cbl(cbl, bad_lim, critical_lim)

    n_theta = 96
    theta = np.linspace(0, 2 * np.pi, n_theta)
    theta_grid, z_grid = np.meshgrid(theta, depth)

    # Geometría esquemática. No representa pulgadas reales, sino capas relativas.
    casing_radius = 0.62
    cement_radius_base = 1.00
    formation_radius = 1.18

    # El radio externo se abre más donde la adherencia es peor para que la zona crítica se vea como banda o canal.
    if sector_mode:
        sector_width = np.deg2rad(max(15, min(180, sector_degrees)))
        center = np.deg2rad(40)
        angular_distance = np.angle(np.exp(1j * (theta_grid - center)))
        angular_weight = np.exp(-(angular_distance ** 2) / (2 * (sector_width / 2.355) ** 2))
        sev_matrix = severity[:, None] * angular_weight
    else:
        sev_matrix = np.repeat(severity[:, None], n_theta, axis=1)

    cement_radius = cement_radius_base + 0.32 * sev_matrix
    x_cement = cement_radius * np.cos(theta_grid)
    y_cement = cement_radius * np.sin(theta_grid)

    x_pipe, y_pipe, z_pipe, c_pipe = cylinder_surface(casing_radius, depth, n_theta=n_theta, color_value=np.zeros_like(z_grid))
    x_form, y_form, z_form, c_form = cylinder_surface(formation_radius, depth, n_theta=n_theta, color_value=np.zeros_like(z_grid))

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=x_form,
        y=y_form,
        z=z_form,
        surfacecolor=c_form,
        colorscale=[[0, "#d9d9d9"], [1, "#d9d9d9"]],
        opacity=0.12,
        showscale=False,
        name="Formación",
        hoverinfo="skip",
    ))

    fig.add_trace(go.Surface(
        x=x_pipe,
        y=y_pipe,
        z=z_pipe,
        surfacecolor=c_pipe,
        colorscale=[[0, "#3a3a3a"], [1, "#3a3a3a"]],
        opacity=0.45,
        showscale=False,
        name="Casing",
        hoverinfo="skip",
    ))

    colorscale = [
        [0.00, "#1f78b4"],
        [0.35, "#35a853"],
        [0.55, "#f2c94c"],
        [0.75, "#f2994a"],
        [1.00, "#d7191c"],
    ]

    fig.add_trace(go.Surface(
        x=x_cement,
        y=y_cement,
        z=z_grid,
        surfacecolor=sev_matrix,
        cmin=0,
        cmax=1,
        colorscale=colorscale,
        colorbar=dict(title="Severidad CBL", len=0.72),
        opacity=0.92,
        name="Cemento interpretado",
        customdata=np.repeat(cbl[:, None], n_theta, axis=1),
        hovertemplate="Profundidad: %{z:.2f} ft<br>CBL: %{customdata:.2f} mV<br>Severidad: %{surfacecolor:.2f}<extra></extra>",
    ))

    fig.update_layout(
        height=850,
        scene=dict(
            xaxis=dict(title="X", showbackground=False),
            yaxis=dict(title="Y", showbackground=False),
            zaxis=dict(title="Profundidad, ft", autorange="reversed"),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=3.2),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.1)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def make_summary_table(good_lim, regular_lim, bad_lim, critical_lim, min_len_ft):
    return pd.DataFrame([
        {"Rango CBL": f"≤ {good_lim:g} mV", "Condición": "Bueno", "Uso en la app": "Cemento con buena adherencia relativa casing cemento."},
        {"Rango CBL": f"> {good_lim:g} y ≤ {regular_lim:g} mV", "Condición": "Regular", "Uso en la app": "Zona a observar, no necesariamente problema operativo."},
        {"Rango CBL": f"> {regular_lim:g} y < {bad_lim:g} mV", "Condición": "Alerta", "Uso en la app": "Posible pérdida parcial de adherencia."},
        {"Rango CBL": f"≥ {bad_lim:g} y < {critical_lim:g} mV", "Condición": "Mala adherencia", "Uso en la app": "Zona sospechosa de mal acoplamiento casing cemento."},
        {"Rango CBL": f"≥ {critical_lim:g} mV", "Condición": "Crítico", "Uso en la app": "Alta amplitud, posible free pipe o muy pobre adherencia."},
        {"Rango CBL": f"Longitud ≥ {min_len_ft:g} ft", "Condición": "Criterio operativo", "Uso en la app": "Separa eventos puntuales de intervalos relevantes."},
    ])


# ------------------------------------------------------------
# Interfaz
# ------------------------------------------------------------

st.title("Visualizador CBL con esquema 2D y modelo 3D interpretativo")
st.caption("Carga LAS, XLSX, XLS o CSV. La app interpreta amplitud CBL y genera una representación esquemática del casing, cemento y formación.")

with st.expander("Qué mide CBL y qué aporta VDL", expanded=True):
    st.markdown(
        """
        **CBL** trabaja principalmente con la amplitud de la señal acústica recibida. Cuando la amplitud es baja, normalmente hay mejor acoplamiento entre casing y cemento. Cuando la amplitud es alta, puede indicar mala adherencia, pobre cemento o condición cercana a free pipe.

        **VDL** no es simplemente otra curva de amplitud. Es la imagen del tren de onda acústico por profundidad y tiempo. Ayuda a diferenciar señales de casing, cemento y formación. Por eso se usa para evaluar cualitativamente si hay llegada de formación, canalización o pobre acoplamiento cemento formación.

        Entonces tu idea está bien encaminada, pero con una precisión: **CBL evalúa mejor la adherencia casing cemento**; **VDL ayuda a interpretar la adherencia cemento formación y la calidad global de aislamiento**, pero no la mide de forma única sin análisis de la forma de onda.

        Si el LAS solo trae AMP3FT, AMP5FT, TT, GR o CCL, se puede hacer análisis CBL. Para VDL real se necesita una matriz waveform, con muchas muestras de amplitud por cada profundidad.
        """
    )

st.subheader("Constraints operativos usados por la app")

col_a, col_b, col_c, col_d, col_e = st.columns(5)
with col_a:
    good_lim = st.number_input("Bueno hasta, mV", min_value=0.0, value=3.0, step=0.5)
with col_b:
    regular_lim = st.number_input("Regular hasta, mV", min_value=good_lim, value=8.0, step=0.5)
with col_c:
    bad_lim = st.number_input("Mala adherencia desde, mV", min_value=regular_lim, value=10.0, step=0.5)
with col_d:
    critical_lim = st.number_input("Crítico desde, mV", min_value=bad_lim, value=15.0, step=0.5)
with col_e:
    min_len_ft = st.number_input("Longitud operativa mínima, ft", min_value=0.5, value=5.0, step=0.5)

st.dataframe(make_summary_table(good_lim, regular_lim, bad_lim, critical_lim, min_len_ft), use_container_width=True, hide_index=True)

st.info(
    "La gráfica 3D es interpretativa. Como el CBL convencional no trae azimut, una zona roja no indica el lado exacto del casing. "
    "Por defecto se representa como banda de 360 grados. El modo sectorial solo es didáctico y no debe usarse como evidencia de orientación real."
)

uploaded_file = st.file_uploader("Carga tu archivo LAS, XLSX, XLS o CSV", type=["las", "xlsx", "xls", "csv"])

if uploaded_file is None:
    st.stop()

try:
    raw_df = load_file(uploaded_file)
except Exception as exc:
    st.error(f"No se pudo leer el archivo: {exc}")
    st.stop()

if raw_df.empty:
    st.error("El archivo se leyó, pero no contiene datos tabulares.")
    st.stop()

st.success(f"Archivo cargado: {uploaded_file.name}")

with st.expander("Selección de curvas", expanded=True):
    depth_guess = guess_depth_column(raw_df)
    cbl_guess = guess_cbl_column(raw_df)
    gr_guess = find_optional_column(raw_df, ["GR", "GK", "GAMMA", "GK1"])
    ccl_guess = find_optional_column(raw_df, ["CCL"])
    tt_guess = find_optional_column(raw_df, ["TT", "TT3FT", "TRANSIT", "TRAVEL"])

    columns = list(raw_df.columns)
    depth_col = st.selectbox("Curva de profundidad", columns, index=columns.index(depth_guess) if depth_guess in columns else 0)
    cbl_col = st.selectbox("Curva CBL o amplitud", columns, index=columns.index(cbl_guess) if cbl_guess in columns else 0)

    optional = ["Ninguna"] + columns
    gr_col = st.selectbox("Gamma Ray, opcional", optional, index=optional.index(gr_guess) if gr_guess in optional else 0)
    tt_col = st.selectbox("Travel Time, opcional", optional, index=optional.index(tt_guess) if tt_guess in optional else 0)
    ccl_col = st.selectbox("CCL, opcional", optional, index=optional.index(ccl_guess) if ccl_guess in optional else 0)

    gr_col = None if gr_col == "Ninguna" else gr_col
    tt_col = None if tt_col == "Ninguna" else tt_col
    ccl_col = None if ccl_col == "Ninguna" else ccl_col

try:
    df = clean_data(raw_df, depth_col, cbl_col, gr_col=gr_col, ccl_col=ccl_col, tt_col=tt_col)
except Exception as exc:
    st.error(f"No se pudo preparar la información: {exc}")
    st.stop()

if len(df) < 3:
    st.error("Hay muy pocos puntos válidos para graficar.")
    st.stop()

# Clasificación
df["CLASS"] = df["CBL"].apply(lambda x: classify_cbl(x, good_lim, regular_lim, bad_lim, critical_lim))
df["SEVERITY"] = severity_from_cbl(df["CBL"].to_numpy(), bad_lim, critical_lim)

# Métricas principales
m1, m2, m3, m4 = st.columns(4)
m1.metric("Desde ft", f"{df['DEPTH'].min():.2f}")
m2.metric("Hasta ft", f"{df['DEPTH'].max():.2f}")
m3.metric("CBL máximo mV", f"{df['CBL'].max():.2f}")
m4.metric("Puntos analizados", f"{len(df):,}")

intervals = detect_intervals(df, bad_lim, min_len_ft)
operational = intervals[intervals["Condición"] == "Operativo"] if not intervals.empty else pd.DataFrame()
short_events = intervals[intervals["Condición"] == "Evento corto"] if not intervals.empty else pd.DataFrame()

st.subheader("Resultado interpretativo")
if not operational.empty:
    st.warning("Se encontraron intervalos operativos con posible mala adherencia.")
    st.dataframe(operational, use_container_width=True, hide_index=True)
else:
    st.info("No se encontraron intervalos continuos que superen el criterio operativo de mala adherencia.")

if not short_events.empty:
    with st.expander("Eventos cortos o puntuales"):
        st.dataframe(short_events, use_container_width=True, hide_index=True)

st.subheader("Vista tipo log")
st.plotly_chart(make_log_plot(df, good_lim, regular_lim, bad_lim, critical_lim), use_container_width=True)

st.subheader("Esquema 2D tipo casing cemento formación")
st.caption("Este esquema convierte la amplitud CBL en severidad visual. No reemplaza la interpretación oficial del registro.")
st.plotly_chart(make_2d_schematic(df, good_lim, regular_lim, bad_lim, critical_lim), use_container_width=True)

st.subheader("Modelo 3D interpretativo")
col3a, col3b, col3c = st.columns([1, 1, 1])
with col3a:
    max_points = st.slider("Puntos máximos para 3D", min_value=200, max_value=1500, value=700, step=100)
with col3b:
    sector_mode = st.checkbox("Mostrar sector didáctico", value=False)
with col3c:
    sector_degrees = st.slider("Apertura del sector didáctico", min_value=30, max_value=180, value=90, step=15)

st.plotly_chart(make_3d_cement_model(df, bad_lim, critical_lim, max_points=max_points, sector_mode=sector_mode, sector_degrees=sector_degrees), use_container_width=True)

st.subheader("Datos procesados")
with st.expander("Ver tabla completa"):
    st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Descargar datos procesados CSV", csv, file_name="cbl_datos_procesados.csv", mime="text/csv")

if not intervals.empty:
    csv_int = intervals.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar intervalos interpretados CSV", csv_int, file_name="cbl_intervalos_interpretados.csv", mime="text/csv")
