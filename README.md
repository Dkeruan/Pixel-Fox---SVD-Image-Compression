# Pixel Fox — SVD image compression

A small Streamlit app for exploring low-rank image approximations. Upload a grayscale or color image, choose the number of singular values `k`, compare the reconstruction with the original, inspect the log singular-value spectrum, and download the result.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Method and limits

- Grayscale images use one matrix SVD. Color images use independent SVDs for the red, green, and blue channels, then recombine them.
- The default rank is one fifth of the available rank; the slider spans the valid range from 1 to `min(height, width)`. The spectrum plots `log10(σᵢ)` against the one-based singular-value index, with the selected `k` marked.
- Images are resized with aspect ratio preserved when their longest side exceeds 512 pixels. The download uses the analyzed dimensions. Uploads are limited to 20 MB and 40 million source pixels to bound decode memory and SVD work. NumPy's dense SVD runs once per channel, and Streamlit caches the result while `k` changes.
- PNG, JPEG, WebP, GIF, BMP, and TIFF uploads are accepted. Animated files use their first frame. Transparency is composited onto white. Downloads are PNG, JPEG, or WebP.
- “Retained energy” is the fraction of squared singular-value energy retained across channels, before clipping and rounding the reconstructed pixels. Export size also depends on the selected image format and encoder, so it can be larger or smaller than the uploaded file.
- The spectrum omits singular values at or below the float32 numerical-rank tolerance because their base-10 logarithm is undefined or unresolved at this precision. The table shows the same spectrum values.

The interactive spectrum uses Plotly. The app also provides a table of the plotted values for users who prefer a data view.
