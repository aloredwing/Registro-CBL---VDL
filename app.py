import io
import re
from pathlib import Path

import lasio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="CBL VDL 3D",
    page_icon="🛢️",
    layout="wide",
)

# ============================================================
# Lectura y limpieza de archivos
# ============================================================

def normalize_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def load_file(uploaded_file) -> pd.DataFrame:
    """Lee LAS, CSV, XLSX o XLS y devuelve un DataFrame."""
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
        nkey = normalize_name(key)
        for col, ncol in normalized.items():
            if nkey == ncol or nkey in ncol:
                return col
    return None


def numeric_columns(df: pd.DataFrame):
    cols = []
    for col in df.columns:
        test = pd.to_numeric(df[col], errors="coerce")
        if test.notna().sum() > 0:
            cols.append(col)
    return cols


def clean_cbl_data(df: pd.DataFrame, depth_col: str, cbl_col: str, gr_col=None, ccl_col=None, tt_col=None):
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


# ============================================================
# Interpretación CBL
# ============================================================

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
    """0 = buena adherencia relativa, 1 = severidad alta."""
    sev = (np.asarray(cbl, dtype=float) - bad_lim) / max(critical_lim - bad_lim, 0.001)
    return np.clip(sev, 0, 1)


def cement_coverage_from_cbl(cbl, good_lim, critical_lim):
    """1 = cemento visualmente completo, 0 = cemento muy deficiente en el esquema."""
    cov = 1 - (np.asarray(cbl, dtype=float) - good_lim) / max(critical_lim - good_lim, 0.001)
    return np.clip(cov, 0, 1)


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


# ============================================================
# Preparación VDL opcional
# ============================================================

def auto_waveform_columns(df: pd.DataFrame, depth_col: str):
    """Detecta columnas que parecen muestras waveform."""
    exclude_tokens = ["DEPT", "DEPTH", "MD", "TVD", "GR", "GK", "CCL", "CBL", "AMP", "TT", "TIME", "TRAVEL"]
    candidates = []
    for col in numeric_columns(df):
        if col == depth_col:
            continue
        ncol = normalize_name(col)
        if any(tok in ncol for tok in ["WF", "WAVE", "VDL", "SAMP", "SAMPLE"]):
            candidates.append(col)
            continue
        if not any(tok in ncol for tok in exclude_tokens):
            candidates.append(col)
    return candidates


def prepare_vdl_index(vdl_raw: pd.DataFrame, depth_col: str, mode: str, selected_cols, index_col=None, invert=False):
    """
    Devuelve DEPTH y VDL_INDEX normalizado entre 0 y 1.
    Para waveform, usa RMS de las columnas seleccionadas como índice visual.
    """
    out = pd.DataFrame()
    out["DEPTH"] = pd.to_numeric(vdl_raw[depth_col], errors="coerce")

    if mode == "Matriz waveform":
        if not selected_cols:
            raise ValueError("Selecciona columnas waveform para calcular el índice VDL.")
        matrix = vdl_raw[selected_cols].apply(pd.to_numeric, errors="coerce")
        values = np.sqrt(np.nanmean(np.square(matrix.to_numpy(dtype=float)), axis=1))
        out["VDL_INDEX_RAW"] = values
    else:
        if not index_col:
            raise ValueError("Selecciona la curva resumen VDL.")
        out["VDL_INDEX_RAW"] = pd.to_numeric(vdl_raw[index_col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["DEPTH", "VDL_INDEX_RAW"])
    out = out.sort_values("DEPTH").drop_duplicates(subset=["DEPTH"])

    if out.empty:
        raise ValueError("No quedaron datos VDL válidos después de la limpieza.")

    p05 = float(np.nanpercentile(out["VDL_INDEX_RAW"], 5))
    p95 = float(np.nanpercentile(out["VDL_INDEX_RAW"], 95))
    if abs(p95 - p05) < 1e-12:
        out["VDL_INDEX"] = 0.5
    else:
        out["VDL_INDEX"] = (out["VDL_INDEX_RAW"] - p05) / (p95 - p05)
        out["VDL_INDEX"] = out["VDL_INDEX"].clip(0, 1)

    if invert:
        out["VDL_INDEX"] = 1 - out["VDL_INDEX"]

    return out[["DEPTH", "VDL_INDEX", "VDL_INDEX_RAW"]].reset_index(drop=True)


def interpolate_vdl_to_cbl_depth(cbl_depth, vdl_df):
    if vdl_df is None or vdl_df.empty:
        return None
    v = vdl_df.dropna().sort_values("DEPTH")
    if len(v) < 2:
        return None
    interp = np.interp(cbl_depth, v["DEPTH"].to_numpy(), v["VDL_INDEX"].to_numpy(), left=np.nan, right=np.nan)
    return interp


# ============================================================
# Gráficos 2D y logs
# ============================================================

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


def make_2d_annulus_schematic(df, good_lim, bad_lim, critical_lim, vdl_interp=None):
    plot_df = df.copy()
    z = plot_df["DEPTH"].to_numpy()
    cbl = plot_df["CBL"].to_numpy()
    severity = severity_from_cbl(cbl, bad_lim, critical_lim)
    coverage = cement_coverage_from_cbl(cbl, good_lim, critical_lim)

    # x esquemático: casing, línea invisible, cemento y VDL externo.
    x = np.linspace(-1.75, 1.75, 140)
    image = np.full((len(z), len(x)), np.nan)

    casing_r = 0.30
    interface_r = 0.42
    max_cement_r = 1.02
    vdl_inner_r = 1.12
    vdl_outer_r = 1.48

    for i, (sev, cov) in enumerate(zip(severity, coverage)):
        # Menor cobertura CBL = menor espesor visual de cemento plomo.
        cement_outer = interface_r + cov * (max_cement_r - interface_r)
        for j, xx in enumerate(x):
            ax = abs(xx)
            if ax <= casing_r:
                image[i, j] = -0.55              # casing
            elif casing_r < ax <= interface_r:
                image[i, j] = -0.10              # espacio / línea invisible
            elif interface_r < ax <= cement_outer:
                image[i, j] = 0.35 + 0.35 * cov  # cemento plomo
            elif cement_outer < ax <= max_cement_r:
                image[i, j] = 0.10 + 0.20 * sev  # bajo cemento / vacío visual
            elif vdl_inner_r <= ax <= vdl_outer_r:
                if vdl_interp is None or np.isnan(vdl_interp[i]):
                    image[i, j] = 0.02            # VDL pendiente
                else:
                    image[i, j] = 0.75 + 0.25 * vdl_interp[i]

    colorscale = [
        [0.00, "#222222"],
        [0.18, "#222222"],
        [0.19, "#ffffff"],
        [0.33, "#ffffff"],
        [0.34, "#f7f7f7"],
        [0.47, "#f7f7f7"],
        [0.48, "#d9d9d9"],
        [0.68, "#9e9e9e"],
        [0.69, "#c7d8ff"],
        [0.84, "#5b8def"],
        [1.00, "#3f1dcb"],
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.58, 0.42],
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=["Esquema casing cemento VDL", "Curva CBL"],
    )

    fig.add_trace(go.Heatmap(
        x=x,
        y=z,
        z=image,
        zmin=-0.55,
        zmax=1.0,
        colorscale=colorscale,
        showscale=False,
        hovertemplate="Profundidad: %{y:.2f} ft<extra></extra>",
    ), row=1, col=1)

    # Líneas de referencia.
    for xline, color, width, dash in [
        (-casing_r, "black", 3, "solid"),
        (casing_r, "black", 3, "solid"),
        (-interface_r, "rgba(0,0,0,0.18)", 1, "dot"),
        (interface_r, "rgba(0,0,0,0.18)", 1, "dot"),
        (-max_cement_r, "gray", 1, "dash"),
        (max_cement_r, "gray", 1, "dash"),
        (-vdl_inner_r, "#5b8def", 1, "dot"),
        (vdl_inner_r, "#5b8def", 1, "dot"),
        (-vdl_outer_r, "#5b8def", 1, "dash"),
        (vdl_outer_r, "#5b8def", 1, "dash"),
    ]:
        fig.add_vline(x=xline, line_width=width, line_dash=dash, line_color=color, row=1, col=1)

    fig.add_trace(go.Scatter(x=cbl, y=z, mode="lines", name="CBL"), row=1, col=2)
    for xline, label in [(bad_lim, "Mala adherencia"), (critical_lim, "Crítico")]:
        fig.add_vline(x=xline, line_width=1, line_dash="dash", annotation_text=label, row=1, col=2)

    fig.update_yaxes(autorange="reversed", title_text="Profundidad, ft")
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text="CBL, mV", row=1, col=2)
    fig.update_layout(height=850, margin=dict(l=20, r=20, t=70, b=20), showlegend=False)
    return fig


# ============================================================
# Modelo 3D tubular
# ============================================================

def cylinder_grid(radius_values, depth, n_theta=96):
    depth = np.asarray(depth, dtype=float)
    theta = np.linspace(0, 2 * np.pi, n_theta)
    theta_grid, z_grid = np.meshgrid(theta, depth)

    if np.isscalar(radius_values):
        r_grid = np.full_like(z_grid, float(radius_values), dtype=float)
    else:
        radius_values = np.asarray(radius_values, dtype=float)
        r_grid = np.repeat(radius_values[:, None], n_theta, axis=1)

    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)
    return x, y, z_grid, theta_grid


def ring_line(radius, z, color="black", width=3, name="ring"):
    theta = np.linspace(0, 2 * np.pi, 160)
    return go.Scatter3d(
        x=radius * np.cos(theta),
        y=radius * np.sin(theta),
        z=np.full_like(theta, z),
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
        showlegend=False,
        hoverinfo="skip",
    )


def make_3d_tubular_model(
    df,
    good_lim,
    bad_lim,
    critical_lim,
    vdl_interp=None,
    max_points=700,
    sector_mode=False,
    sector_degrees=90,
    show_reference_rings=True,
):
    plot_df = df[["DEPTH", "CBL"]].dropna().copy()
    if len(plot_df) > max_points:
        idx = np.linspace(0, len(plot_df) - 1, max_points).astype(int)
        plot_df = plot_df.iloc[idx].copy()
        if vdl_interp is not None:
            vdl_interp = np.asarray(vdl_interp)[idx]

    depth = plot_df["DEPTH"].to_numpy(dtype=float)
    cbl = plot_df["CBL"].to_numpy(dtype=float)
    severity = severity_from_cbl(cbl, bad_lim, critical_lim)
    coverage = cement_coverage_from_cbl(cbl, good_lim, critical_lim)

    n_theta = 112
    casing_r = 0.50
    interface_r = 0.64        # línea invisible casing cemento
    cement_outer_max = 1.08   # cemento completo
    vdl_r = 1.34              # capa externa VDL
    formation_r = 1.55        # formación referencial

    theta = np.linspace(0, 2 * np.pi, n_theta)
    theta_grid, z_grid = np.meshgrid(theta, depth)

    if sector_mode:
        sector_width = np.deg2rad(max(15, min(180, sector_degrees)))
        center = np.deg2rad(40)
        angular_distance = np.angle(np.exp(1j * (theta_grid - center)))
        angular_weight = np.exp(-(angular_distance ** 2) / (2 * (sector_width / 2.355) ** 2))
        local_severity = severity[:, None] * angular_weight
        local_coverage = np.clip(1 - local_severity, 0, 1)
    else:
        local_severity = np.repeat(severity[:, None], n_theta, axis=1)
        local_coverage = np.repeat(coverage[:, None], n_theta, axis=1)

    # Cemento: donde hay buena adherencia se ve más espeso y plomo.
    cement_radius = interface_r + local_coverage * (cement_outer_max - interface_r)
    x_cement = cement_radius * np.cos(theta_grid)
    y_cement = cement_radius * np.sin(theta_grid)

    x_pipe, y_pipe, z_pipe, _ = cylinder_grid(casing_r, depth, n_theta)
    x_interface, y_interface, z_interface, _ = cylinder_grid(interface_r, depth, n_theta)
    x_form, y_form, z_form, _ = cylinder_grid(formation_r, depth, n_theta)

    fig = go.Figure()

    # Formación referencial externa.
    fig.add_trace(go.Surface(
        x=x_form,
        y=y_form,
        z=z_form,
        surfacecolor=np.zeros_like(z_form),
        colorscale=[[0, "#eeeeee"], [1, "#eeeeee"]],
        opacity=0.07,
        showscale=False,
        name="Formación referencial",
        hoverinfo="skip",
    ))

    # Capa VDL externa, pendiente o interpretada si se carga waveform.
    if vdl_interp is None:
        x_vdl, y_vdl, z_vdl, _ = cylinder_grid(vdl_r, depth, n_theta)
        fig.add_trace(go.Surface(
            x=x_vdl,
            y=y_vdl,
            z=z_vdl,
            surfacecolor=np.zeros_like(z_vdl),
            colorscale=[[0, "#c7d8ff"], [1, "#c7d8ff"]],
            opacity=0.10,
            showscale=False,
            name="VDL pendiente",
            hovertemplate="Profundidad: %{z:.2f} ft<br>VDL: pendiente<extra></extra>",
        ))
    else:
        vdl_vals = np.asarray(vdl_interp, dtype=float)
        vdl_vals = np.where(np.isnan(vdl_vals), 0.0, vdl_vals)
        vdl_surface = np.repeat(vdl_vals[:, None], n_theta, axis=1)
        x_vdl, y_vdl, z_vdl, _ = cylinder_grid(vdl_r, depth, n_theta)
        fig.add_trace(go.Surface(
            x=x_vdl,
            y=y_vdl,
            z=z_vdl,
            surfacecolor=vdl_surface,
            cmin=0,
            cmax=1,
            colorscale=[[0, "#eaf0ff"], [0.5, "#5b8def"], [1, "#3f1dcb"]],
            opacity=0.38,
            colorbar=dict(title="VDL índice", len=0.45, y=0.25),
            name="Capa VDL",
            customdata=vdl_surface,
            hovertemplate="Profundidad: %{z:.2f} ft<br>VDL índice: %{customdata:.2f}<extra></extra>",
        ))

    # Línea invisible de referencia casing cemento.
    fig.add_trace(go.Surface(
        x=x_interface,
        y=y_interface,
        z=z_interface,
        surfacecolor=np.zeros_like(z_interface),
        colorscale=[[0, "#ffffff"], [1, "#ffffff"]],
        opacity=0.035,
        showscale=False,
        name="Interfaz casing cemento",
        hoverinfo="skip",
    ))

    # Casing / tubular.
    fig.add_trace(go.Surface(
        x=x_pipe,
        y=y_pipe,
        z=z_pipe,
        surfacecolor=np.zeros_like(z_pipe),
        colorscale=[[0, "#2b2b2b"], [1, "#2b2b2b"]],
        opacity=0.72,
        showscale=False,
        name="Tubular casing",
        hoverinfo="skip",
    ))

    # Cemento plomo. Menor cobertura = radio menor y tono más claro.
    fig.add_trace(go.Surface(
        x=x_cement,
        y=y_cement,
        z=z_grid,
        surfacecolor=local_coverage,
        cmin=0,
        cmax=1,
        colorscale=[[0, "#f7f7f7"], [0.45, "#cfcfcf"], [1, "#8f8f8f"]],
        opacity=0.88,
        colorbar=dict(title="Cobertura cemento", len=0.45, y=0.77),
        name="Cemento interpretado por CBL",
        customdata=np.repeat(cbl[:, None], n_theta, axis=1),
        hovertemplate=(
            "Profundidad: %{z:.2f} ft<br>"
            "CBL: %{customdata:.2f} mV<br>"
            "Cobertura cemento: %{surfacecolor:.2f}<extra></extra>"
        ),
    ))

    # Anillos rojos en los intervalos con CBL crítico o mala adherencia, solo como guía visual.
    bad_depths = depth[cbl >= bad_lim]
    if len(bad_depths) > 0:
        # Reducimos cantidad de anillos para no saturar.
        step = max(1, int(len(bad_depths) / 80))
        for z in bad_depths[::step]:
            fig.add_trace(ring_line(cement_outer_max + 0.015, z, color="rgba(215,25,28,0.40)", width=2, name="bad ring"))

    if show_reference_rings and len(depth) > 0:
        for z in [float(np.nanmin(depth)), float(np.nanmax(depth))]:
            fig.add_trace(ring_line(casing_r, z, color="black", width=4, name="casing ring"))
            fig.add_trace(ring_line(cement_outer_max, z, color="gray", width=2, name="cement ring"))
            fig.add_trace(ring_line(vdl_r, z, color="rgba(91,141,239,0.75)", width=2, name="vdl ring"))

    fig.update_layout(
        height=880,
        scene=dict(
            xaxis=dict(title="X", showbackground=False, visible=True),
            yaxis=dict(title="Y", showbackground=False, visible=True),
            zaxis=dict(title="Profundidad, ft", autorange="reversed"),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=3.6),
            camera=dict(eye=dict(x=1.9, y=1.8, z=1.15)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


# ============================================================
# Tablas
# ============================================================

def make_summary_table(good_lim, regular_lim, bad_lim, critical_lim, min_len_ft):
    return pd.DataFrame([
        {"Rango CBL": f"≤ {good_lim:g} mV", "Condición": "Bueno", "Uso en la app": "Mayor cobertura ploma de cemento alrededor del tubular."},
        {"Rango CBL": f"> {good_lim:g} y ≤ {regular_lim:g} mV", "Condición": "Regular", "Uso en la app": "Cemento visible, pero con menor calidad relativa."},
        {"Rango CBL": f"> {regular_lim:g} y < {bad_lim:g} mV", "Condición": "Alerta", "Uso en la app": "Reduce el espesor visual del cemento."},
        {"Rango CBL": f"≥ {bad_lim:g} y < {critical_lim:g} mV", "Condición": "Mala adherencia", "Uso en la app": "Menos cemento plomo y anillo rojo de advertencia."},
        {"Rango CBL": f"≥ {critical_lim:g} mV", "Condición": "Crítico", "Uso en la app": "Cemento muy reducido visualmente; posible free pipe o pobre adherencia."},
        {"Rango CBL": f"Longitud ≥ {min_len_ft:g} ft", "Condición": "Criterio operativo", "Uso en la app": "Separa eventos puntuales de intervalos relevantes."},
    ])


# ============================================================
# Interfaz Streamlit
# ============================================================

st.title("Visualizador 3D CBL con capa VDL opcional")
st.caption("Carga CBL para ver tubular, línea de interfaz, cemento plomo y espacio VDL. Luego puedes cargar VDL cuando tengas matriz waveform o una curva resumen.")

with st.expander("Interpretación técnica rápida", expanded=True):
    st.markdown(
        """
        **CBL** se interpreta principalmente con la amplitud acústica y se usa para evaluar la adherencia **casing cemento**. En este modelo, amplitud baja representa mejor cobertura de cemento; amplitud alta representa menor cemento plomo alrededor del tubular.

        **VDL** corresponde al tren de onda acústico por profundidad y tiempo. Ayuda a interpretar acoplamiento **cemento formación**, canalización y señal de formación. Para un VDL real se requiere una matriz waveform. Si todavía no tienes esa información, la app deja una capa externa azul tenue como espacio reservado.

        La vista 3D es esquemática. No entrega azimut real del daño porque un CBL convencional no trae orientación radial. El modo sectorial solo sirve para visualizar mejor el concepto.
        """
    )

st.subheader("Constraints operativos")
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

st.divider()

left, right = st.columns(2)
with left:
    uploaded_cbl = st.file_uploader("1. Carga CBL / cement bond log", type=["las", "xlsx", "xls", "csv"], key="cbl_file")
with right:
    uploaded_vdl = st.file_uploader("2. Carga VDL / waveform opcional", type=["las", "xlsx", "xls", "csv"], key="vdl_file")

if uploaded_cbl is None:
    st.info("Primero carga tu archivo CBL. El VDL es opcional y puede cargarse después.")
    st.stop()

try:
    raw_cbl = load_file(uploaded_cbl)
except Exception as exc:
    st.error(f"No se pudo leer el archivo CBL: {exc}")
    st.stop()

if raw_cbl.empty:
    st.error("El archivo CBL se leyó, pero no contiene datos tabulares.")
    st.stop()

st.success(f"CBL cargado: {uploaded_cbl.name}")

with st.expander("Selección de curvas CBL", expanded=True):
    depth_guess = guess_depth_column(raw_cbl)
    cbl_guess = guess_cbl_column(raw_cbl)
    gr_guess = find_optional_column(raw_cbl, ["GR", "GK", "GAMMA", "GK1"])
    ccl_guess = find_optional_column(raw_cbl, ["CCL"])
    tt_guess = find_optional_column(raw_cbl, ["TT", "TT3FT", "TRANSIT", "TRAVEL"])

    cbl_columns = list(raw_cbl.columns)
    depth_col = st.selectbox("Curva de profundidad", cbl_columns, index=cbl_columns.index(depth_guess) if depth_guess in cbl_columns else 0)
    cbl_col = st.selectbox("Curva CBL o amplitud", cbl_columns, index=cbl_columns.index(cbl_guess) if cbl_guess in cbl_columns else 0)

    optional = ["Ninguna"] + cbl_columns
    gr_col = st.selectbox("Gamma Ray, opcional", optional, index=optional.index(gr_guess) if gr_guess in optional else 0)
    tt_col = st.selectbox("Travel Time, opcional", optional, index=optional.index(tt_guess) if tt_guess in optional else 0)
    ccl_col = st.selectbox("CCL, opcional", optional, index=optional.index(ccl_guess) if ccl_guess in optional else 0)

    gr_col = None if gr_col == "Ninguna" else gr_col
    tt_col = None if tt_col == "Ninguna" else tt_col
    ccl_col = None if ccl_col == "Ninguna" else ccl_col

try:
    df = clean_cbl_data(raw_cbl, depth_col, cbl_col, gr_col=gr_col, ccl_col=ccl_col, tt_col=tt_col)
except Exception as exc:
    st.error(f"No se pudo preparar la información CBL: {exc}")
    st.stop()

if len(df) < 3:
    st.error("Hay muy pocos puntos CBL válidos para graficar.")
    st.stop()

df["CLASS"] = df["CBL"].apply(lambda x: classify_cbl(x, good_lim, regular_lim, bad_lim, critical_lim))
df["SEVERITY"] = severity_from_cbl(df["CBL"].to_numpy(), bad_lim, critical_lim)
df["CEMENT_COVERAGE"] = cement_coverage_from_cbl(df["CBL"].to_numpy(), good_lim, critical_lim)

# Preparación del VDL opcional.
vdl_df = None
vdl_interp = None
if uploaded_vdl is not None:
    try:
        raw_vdl = load_file(uploaded_vdl)
        st.success(f"VDL cargado: {uploaded_vdl.name}")
        with st.expander("Selección de curvas VDL", expanded=True):
            vdl_depth_guess = guess_depth_column(raw_vdl)
            vdl_cols = list(raw_vdl.columns)
            vdl_depth_col = st.selectbox("Curva de profundidad VDL", vdl_cols, index=vdl_cols.index(vdl_depth_guess) if vdl_depth_guess in vdl_cols else 0)
            vdl_mode = st.radio("Tipo de información VDL", ["Matriz waveform", "Curva resumen"], horizontal=True)

            if vdl_mode == "Matriz waveform":
                wf_guess = auto_waveform_columns(raw_vdl, vdl_depth_col)
                selected_wf = st.multiselect("Columnas waveform", options=numeric_columns(raw_vdl), default=wf_guess[:120])
                vdl_index_col = None
            else:
                selected_wf = []
                numeric_vdl = [c for c in numeric_columns(raw_vdl) if c != vdl_depth_col]
                default_index = 0 if numeric_vdl else None
                vdl_index_col = st.selectbox("Curva resumen VDL", numeric_vdl, index=default_index if default_index is not None else 0)

            invert_vdl = st.checkbox("Invertir escala VDL", value=False, help="Úsalo si tu curva VDL aumenta cuando la condición es mejor y quieres invertir el color.")

        vdl_df = prepare_vdl_index(raw_vdl, vdl_depth_col, vdl_mode, selected_wf, index_col=vdl_index_col, invert=invert_vdl)
        vdl_interp = interpolate_vdl_to_cbl_depth(df["DEPTH"].to_numpy(), vdl_df)
        if vdl_interp is not None:
            df["VDL_INDEX"] = vdl_interp
    except Exception as exc:
        st.warning(f"Se cargó el CBL, pero no se pudo preparar el VDL: {exc}")
        vdl_df = None
        vdl_interp = None

# Métricas.
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Desde ft", f"{df['DEPTH'].min():.2f}")
m2.metric("Hasta ft", f"{df['DEPTH'].max():.2f}")
m3.metric("CBL máximo mV", f"{df['CBL'].max():.2f}")
m4.metric("Cobertura cemento prom.", f"{df['CEMENT_COVERAGE'].mean()*100:.1f}%")
m5.metric("Puntos analizados", f"{len(df):,}")

intervals = detect_intervals(df, bad_lim, min_len_ft)
operational = intervals[intervals["Condición"] == "Operativo"] if not intervals.empty else pd.DataFrame()
short_events = intervals[intervals["Condición"] == "Evento corto"] if not intervals.empty else pd.DataFrame()

st.subheader("Resultado interpretativo CBL")
if not operational.empty:
    st.warning("Se encontraron intervalos operativos con posible mala adherencia casing cemento.")
    st.dataframe(operational, use_container_width=True, hide_index=True)
else:
    st.info("No se encontraron intervalos continuos que superen el criterio operativo de mala adherencia.")

if not short_events.empty:
    with st.expander("Eventos cortos o puntuales"):
        st.dataframe(short_events, use_container_width=True, hide_index=True)

st.subheader("Vista tipo log")
st.plotly_chart(make_log_plot(df, good_lim, regular_lim, bad_lim, critical_lim), use_container_width=True)

st.subheader("Esquema 2D casing cemento VDL")
st.caption("El cemento plomo se reduce visualmente donde el CBL indica mayor amplitud. La zona azul es el espacio VDL externo; si cargas waveform se colorea con el índice VDL.")
st.plotly_chart(make_2d_annulus_schematic(df, good_lim, bad_lim, critical_lim, vdl_interp=vdl_interp), use_container_width=True)

st.subheader("Modelo 3D tubular con cemento y capa VDL")
col3a, col3b, col3c, col3d = st.columns(4)
with col3a:
    max_points = st.slider("Puntos máximos 3D", min_value=200, max_value=1500, value=700, step=100)
with col3b:
    sector_mode = st.checkbox("Modo sectorial didáctico", value=True)
with col3c:
    sector_degrees = st.slider("Apertura sectorial", min_value=30, max_value=180, value=95, step=5)
with col3d:
    show_reference_rings = st.checkbox("Anillos de referencia", value=True)

st.plotly_chart(
    make_3d_tubular_model(
        df,
        good_lim=good_lim,
        bad_lim=bad_lim,
        critical_lim=critical_lim,
        vdl_interp=vdl_interp,
        max_points=max_points,
        sector_mode=sector_mode,
        sector_degrees=sector_degrees,
        show_reference_rings=show_reference_rings,
    ),
    use_container_width=True,
)

if vdl_df is None:
    st.info("VDL pendiente: cuando cargues un archivo con matriz waveform o una curva resumen, la capa externa azul se completará automáticamente.")
else:
    st.subheader("VDL cargado")
    st.caption("El índice VDL mostrado es visual y normalizado. Para interpretación final se debe revisar el waveform original, escala de tiempo y respuesta acústica completa.")
    fig_vdl = go.Figure()
    fig_vdl.add_trace(go.Scatter(x=vdl_df["VDL_INDEX"], y=vdl_df["DEPTH"], mode="lines", name="VDL índice"))
    fig_vdl.update_yaxes(autorange="reversed", title_text="Profundidad, ft")
    fig_vdl.update_xaxes(title_text="Índice VDL normalizado")
    fig_vdl.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_vdl, use_container_width=True)

st.subheader("Datos procesados")
with st.expander("Ver tabla completa"):
    st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Descargar datos procesados CSV", csv, file_name="cbl_vdl_datos_procesados.csv", mime="text/csv")

if not intervals.empty:
    csv_int = intervals.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar intervalos interpretados CSV", csv_int, file_name="cbl_intervalos_interpretados.csv", mime="text/csv")
