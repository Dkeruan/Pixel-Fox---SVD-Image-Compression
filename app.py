"""Pixel Fox: an interactive image-compression explorer using truncated SVD."""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from svd_core import (
    MAX_ANALYSIS_SIDE,
    MAX_SOURCE_PIXELS,
    MAX_UPLOAD_BYTES,
    ImageInputError,
    ImageSVD,
    decompose_image,
    encode_reconstruction,
    estimated_factor_bytes,
    reconstruct_image,
    retained_energy_percent,
)


st.set_page_config(page_title="Pixel Fox · Image compression", page_icon="🦊", layout="wide")
FOX_ASSET = Path(__file__).parent / "assets" / "fox.png"
FOX_DATA_URI = "data:image/png;base64," + base64.b64encode(FOX_ASSET.read_bytes()).decode("ascii")


st.markdown(
    """
    <style>
      :root {
      --canvas: #f1f6f2;
        --panel: #fbfdfb;
        --ink: #142a22;
        --muted: #566960;
        --line: #d2dfd8;
        --green: #1e5745;
        --green-soft: #e3eee8;
        --orange: #b95d2a;
      }
      .stApp { background: var(--canvas); color: var(--ink); }
      .block-container { max-width: 1500px; padding-top: 2.2rem; padding-bottom: 2.4rem; }
      h1, h2, h3, p, label { color: var(--ink); }
      .hero { display: flex; align-items: center; gap: 1.2rem; min-height: 116px; }
      .hero-fox { width: 6rem; height: auto; image-rendering: pixelated; }
      .eyebrow { color: var(--green); font: 600 .9rem/1.3 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .2em; margin-bottom: .35rem; }
      .hero h1 { font-size: clamp(2.2rem, 4vw, 3.2rem); line-height: 1.05; letter-spacing: -.045em; margin: 0; }
      .hero p { color: var(--muted); font-size: 1.13rem; margin: .55rem 0 0; }
      .panel-title { font-size: 1.15rem; font-weight: 700; margin: 0 0 .25rem; }
      .panel-subtitle { color: var(--muted); margin: 0 0 .8rem; }
      .empty-state { min-height: 470px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 2rem; background: repeating-conic-gradient(#e9f0ec 0% 25%, #f5f8f6 0% 50%) 50% / 22px 22px; border: 1px dashed #cad8d1; border-radius: 20px; }
      .empty-state .fox { width: 6rem; height: auto; image-rendering: pixelated; margin-bottom: .7rem; }
      .empty-state h3 { margin: 0 0 .4rem; }
      .empty-state p { color: var(--muted); max-width: 500px; }
      .callout { padding: .75rem 1rem; border-radius: 12px; background: var(--green-soft); color: #294a3d; font-size: .92rem; }
      [data-testid="stFileUploaderDropzone"] { border-color: #b8cec3; border-radius: 14px; background: #f8fbf9; min-height: 70px; height: 70px; padding: .35rem .65rem; display: flex; flex-direction: row; align-items: center; justify-content: center; }
      [data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
      [data-testid="stFileUploaderDropzone"] button { width: min(100%, 240px); }
      [data-testid="stFileUploaderDropzone"] button, .stDownloadButton button { background: var(--green); color: #fff; border: 0; border-radius: 10px; font-weight: 650; }
      [data-testid="stFileUploaderDropzone"] button:hover, .stDownloadButton button:hover { background: #164534; color: #fff; border-color: #164534; }
      [data-testid="stFileUploaderDropzone"] button:focus-visible, .stDownloadButton button:focus-visible, button:focus-visible { outline: 3px solid #e18448 !important; outline-offset: 2px; }
      [data-testid="stSlider"] [role="slider"] { accent-color: var(--green); }
      [data-testid="stMetric"] { background: #f4f8f5; padding: .85rem 1rem; border-radius: 12px; border: 1px solid var(--line); }
      [data-testid="stMetricLabel"] { color: var(--muted); }
      div[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--line); background: var(--panel); border-radius: 22px; }
      .stCaption { color: var(--muted); }
      @media (max-width: 760px) {
        .block-container { padding: 1rem 1rem 2rem; }
        .hero { min-height: 0; gap: .7rem; }
        .hero-fox { width: 4rem; }
        .hero p { font-size: 1rem; }
        .empty-state { min-height: 320px; }
      }
      @media (max-width: 960px) {
        .hero h1 { font-size: 2rem; }
        .hero p { font-size: .95rem; }
        .hero-fox { width: 3.8rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, max_entries=3)
def cached_decompose(image_bytes: bytes) -> ImageSVD:
    """Cache the expensive factorization so moving k only rebuilds the preview."""
    return decompose_image(image_bytes)


def mark_active_upload(widget_key: str) -> None:
    uploaded_file = st.session_state.get(widget_key)
    if uploaded_file is None:
        if st.session_state.get("active_upload_widget") == widget_key:
            st.session_state.pop("uploaded_image_bytes", None)
            st.session_state.pop("uploaded_image_name", None)
            st.session_state.pop("active_upload_widget", None)
        return
    st.session_state["active_upload_widget"] = widget_key
    st.session_state["uploaded_image_bytes"] = uploaded_file.getvalue()
    st.session_state["uploaded_image_name"] = uploaded_file.name


def toggle_original() -> None:
    st.session_state["show_original"] = not st.session_state.get("show_original", False)


def build_spectrum_figure(decomposition: ImageSVD, k: int) -> go.Figure:
    figure = go.Figure()
    if decomposition.grayscale:
        channel_specs = [("Grayscale", decomposition.singular_values[0], "#1e5745", "circle", "solid")]
    else:
        channel_specs = [
            ("Red channel", decomposition.singular_values[0], "#b4493d", "circle", "solid"),
            ("Green channel", decomposition.singular_values[1], "#267151", "square", "dash"),
            ("Blue channel", decomposition.singular_values[2], "#356da5", "diamond", "dot"),
        ]

    for name, values, color, marker, dash in channel_specs:
        indices = np.arange(1, len(values) + 1)
        log_values = log_spectrum(values, decomposition)
        figure.add_trace(
            go.Scatter(
                x=indices,
                y=log_values,
                mode="lines",
                name=name,
                line={"color": color, "width": 2, "dash": dash},
                customdata=values,
                hovertemplate="Index i=%{x}<br>log₁₀(σᵢ)=%{y:.3f}<br>σᵢ=%{customdata:.3f}<extra>%{fullData.name}</extra>",
            )
        )
        if np.isfinite(log_values[k - 1]):
            figure.add_trace(
                go.Scatter(
                    x=[k],
                    y=[log_values[k - 1]],
                    mode="markers",
                    name=f"{name} at k",
                    showlegend=False,
                    marker={"color": color, "size": 9, "symbol": marker, "line": {"color": "#ffffff", "width": 1}},
                    hovertemplate=f"Retained at k={k}<br>log₁₀(σᵢ)=%{{y:.3f}}<extra>{name}</extra>",
                )
            )

    figure.update_layout(
        height=390,
        margin={"l": 54, "r": 18, "t": 18, "b": 58},
        paper_bgcolor="#fbfdfb",
        plot_bgcolor="#fbfdfb",
        font={"family": "Inter, system-ui, sans-serif", "color": "#263c32", "size": 12},
        legend={"orientation": "h", "y": 1.08, "x": 0, "title_text": ""},
        hoverlabel={"bgcolor": "#ffffff", "font_color": "#142a22"},
        xaxis={
            "title": "Singular-value index, i",
            "range": [1, max(2, decomposition.max_rank)],
            "gridcolor": "#e5ede8",
            "zeroline": False,
            "showline": True,
            "linecolor": "#b9c9c0",
        },
        yaxis={
            "title": "log₁₀(σᵢ)",
            "gridcolor": "#e5ede8",
            "zeroline": False,
            "showline": True,
            "linecolor": "#b9c9c0",
        },
        shapes=[
            {
                "type": "line",
                "x0": k,
                "x1": k,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#b95d2a", "width": 2, "dash": "dot"},
            }
        ],
    )
    return figure


def log_spectrum(values: np.ndarray, decomposition: ImageSVD) -> np.ndarray:
    """Log positive values above the float32 numerical-rank tolerance; leave the rest as gaps."""
    tolerance = np.finfo(np.float32).eps * max(decomposition.height, decomposition.width) * values[0]
    result = np.full(values.shape, np.nan, dtype=np.float64)
    resolved = values > tolerance
    result[resolved] = np.log10(values[resolved])
    return result


def format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.2f} MB"


def safe_stem(filename: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", filename.rsplit(".", 1)[0]).strip("-_")
    return (stem or "image")[:60]


header_left, header_right = st.columns([1.65, 0.75])
with header_left:
    st.markdown(
        f"""
        <div class="hero">
          <img class="hero-fox" src="{FOX_DATA_URI}" alt="Pixel Fox mascot" />
          <div>
            <div class="eyebrow">PIXEL FOX</div>
            <h1>Image compression</h1>
            <p>Compress photos and see how image detail changes with k singular values.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "webp", "gif", "bmp", "tif", "tiff"],
        key="top_upload",
        on_change=mark_active_upload,
        args=("top_upload",),
        help="Up to 20 MB. PNG, JPEG, WebP, GIF, BMP, and TIFF are supported.",
    )


st.write("")
upload_bytes: bytes | None = st.session_state.get("uploaded_image_bytes")
filename = st.session_state.get("uploaded_image_name", "image")
decomposition: ImageSVD | None = None
input_error: str | None = None
if upload_bytes is not None:
    try:
        with st.spinner("Preparing the image and calculating its SVD…"):
            decomposition = cached_decompose(upload_bytes)
    except ImageInputError as exc:
        input_error = str(exc)
    except (np.linalg.LinAlgError, MemoryError) as exc:
        input_error = "The SVD could not finish for this image. Try a smaller or less detailed image."
    except Exception as exc:  # Keep unexpected image decoder failures inside the app UI.
        input_error = "We couldn't process this image. Try saving a fresh PNG or JPEG copy and upload it again."
        st.exception(exc) if st.query_params.get("debug") == "1" else None


if input_error:
    st.error(input_error, icon="⚠️")


with st.container(border=True):
    if decomposition is None:
        st.markdown('<p class="panel-title">Compression controls</p>', unsafe_allow_html=True)
        st.markdown('<p class="panel-subtitle">Upload an image to choose its reconstruction rank.</p>', unsafe_allow_html=True)
        st.info("Choose k singular values after loading an image. Higher k keeps more image detail.", icon="ℹ️")
    else:
        image_token = hashlib.sha1(upload_bytes or b"").hexdigest()[:12]
        default_k = min(decomposition.max_rank, max(1, int(round(decomposition.max_rank * 0.2))))
        controls = st.columns([2.8, 0.85, 0.9])
        with controls[0]:
            k = st.slider(
                "Singular values retained (k)",
                min_value=1,
                max_value=decomposition.max_rank,
                value=default_k,
                key=f"rank_{image_token}",
                help="Each grayscale channel or RGB channel uses the first k singular values.",
            )
        with controls[1]:
            output_format = st.selectbox("Download format", ["PNG", "JPEG", "WebP"], key=f"format_{image_token}")
        current_pixels = reconstruct_image(decomposition, k)
        export_bytes = encode_reconstruction(current_pixels, output_format)
        suffix = {"PNG": "png", "JPEG": "jpg", "WebP": "webp"}[output_format]
        with controls[2]:
            st.download_button(
                "↓  Download",
                data=export_bytes,
                file_name=f"{safe_stem(filename)}-svd-k{k}.{suffix}",
                mime={"PNG": "image/png", "JPEG": "image/jpeg", "WebP": "image/webp"}[output_format],
                key=f"download_{image_token}",
                use_container_width=True,
                help="Download the current reconstruction at the analyzed dimensions.",
            )


left_panel, right_panel = st.columns([1.02, 1.0], gap="large")
with left_panel:
    with st.container(border=True):
        if decomposition is None:
            st.markdown('<p class="panel-title">Image comparison</p>', unsafe_allow_html=True)
            st.markdown('<p class="panel-subtitle">Original and rank-k reconstruction</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="empty-state"><img class="fox" src="{FOX_DATA_URI}" alt="" />'
                '<h3>Feed the fox an image</h3>'
                '<p>Drop a file here or choose one below. PNG, JPEG, WebP, GIF, BMP, and TIFF are supported. '
                'Images stay on this app instance.</p></div>',
                unsafe_allow_html=True,
            )
            st.file_uploader(
                "Choose or drop an image",
                type=["png", "jpg", "jpeg", "webp", "gif", "bmp", "tif", "tiff"],
                key="panel_upload",
                on_change=mark_active_upload,
                args=("panel_upload",),
                label_visibility="collapsed",
            )
        else:
            show_original = st.session_state.get("show_original", False)
            heading, compare = st.columns([3.2, 1.0])
            with heading:
                st.markdown(
                    '<p class="panel-title">Original image</p>' if show_original
                    else f'<p class="panel-title">Compressed image · k = {k}</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<p class="panel-subtitle">Uploaded image at analyzed dimensions</p>' if show_original
                    else '<p class="panel-subtitle">Rank-k reconstruction</p>',
                    unsafe_allow_html=True,
                )
            with compare:
                st.button(
                    "Show reconstruction" if show_original else "Compare original",
                    on_click=toggle_original,
                    key=f"compare_{hashlib.sha1(upload_bytes or b'').hexdigest()[:12]}",
                    use_container_width=True,
                )
            displayed_pixels = decomposition.original if show_original else current_pixels
            st.image(displayed_pixels, caption=f"{decomposition.width} × {decomposition.height} px", use_column_width=True)

            uploaded_size = len(upload_bytes or b"")
            output_size = len(export_bytes)
            saved_pct = 100 * (uploaded_size - output_size) / uploaded_size if uploaded_size else 0.0
            stats = st.columns(3)
            stats[0].metric("Source file", format_bytes(uploaded_size))
            stats[1].metric(f"Current {output_format} export", format_bytes(output_size))
            direction = "↓" if output_size <= uploaded_size else "↑"
            stats[2].metric("Size change", f"{direction} {abs(saved_pct):.1f}%")
            if decomposition.was_resized:
                st.warning(
                    f"Analysis resized the image from {decomposition.source_width:,} × {decomposition.source_height:,} "
                    f"to {decomposition.width:,} × {decomposition.height:,} pixels. The download uses the analyzed size.",
                    icon="↘️",
                )
            if decomposition.grayscale:
                channel_label = "single grayscale channel"
            else:
                channel_label = "three independent RGB channels"
            st.caption(f"SVD applied to {channel_label}. Energy is calculated from the squared singular values before pixel clipping and rounding.")
            pixel_bytes = decomposition.height * decomposition.width * len(decomposition.singular_values)
            st.caption(
                f"Estimated low-rank factors: {format_bytes(estimated_factor_bytes(decomposition, k))} "
                f"vs {format_bytes(pixel_bytes)} for the analyzed uint8 pixels (float32 factors; excludes metadata)."
            )

with right_panel:
    with st.container(border=True):
        st.markdown('<p class="panel-title">Quality by singular value</p>', unsafe_allow_html=True)
        st.markdown('<p class="panel-subtitle">log₁₀(singular value) by singular-value index</p>', unsafe_allow_html=True)
        if decomposition is None:
            st.markdown(
                '<div class="empty-state"><h3>No data yet</h3>'
                '<p>Upload an image to inspect its singular-value spectrum and the selected rank.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            figure = build_spectrum_figure(decomposition, k)
            st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False, "responsive": True})
            st.caption(
                f"The orange dotted line marks k = {k}. Markers and line styles distinguish color channels; "
                "hover for the singular value and its base-10 logarithm. Values at or below the float32 "
                "numerical-rank tolerance are omitted because log(0) is undefined."
            )
            with st.expander("View spectrum data"):
                table = {"Index i": list(range(1, decomposition.max_rank + 1))}
                channel_names = ["Grayscale"] if decomposition.grayscale else ["Red", "Green", "Blue"]
                for name, values in zip(channel_names, decomposition.singular_values):
                    logs = log_spectrum(values, decomposition)
                    table[f"log₁₀(σᵢ) · {name}"] = [float(value) if np.isfinite(value) else None for value in logs]
                st.dataframe(table, hide_index=True, use_container_width=True, height=240)


st.write("")
st.markdown(
    f'<div class="callout"><strong>How it works:</strong> each image channel is approximated using its first k SVD terms. '
    f'For responsive processing, images are analyzed at up to {MAX_ANALYSIS_SIDE} pixels on the longest side; '
    f'uploads are limited to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB and {MAX_SOURCE_PIXELS // 1_000_000} million source pixels. '
    'Large images are downsampled before decomposition. Transparent areas are flattened onto white, and animated files use their first frame. '
    'The SVD factors are cached while k changes.</div>',
    unsafe_allow_html=True,
)
