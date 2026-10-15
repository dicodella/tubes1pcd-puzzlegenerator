
import streamlit as st

from processing import (
    load_image,
    create_frame,
    create_pieces,
    process_pieces,
    create_puzzle_preview,
    create_zip,
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Puzzle World — Generator",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==================================================
# LIGHT THEME
# ==================================================

st.markdown("""
<style>
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #FFFCF5 !important;
    color: #172B45 !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    display: none !important;
    height: 0 !important;
}

[data-testid="stMainBlockContainer"],
.block-container {
    max-width: 1500px;
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

h1, h2, h3, h4, p, label,
[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"] {
    color: #172B45 !important;
}

.hero {
    background: linear-gradient(110deg, #FFF5D8, #FFFFFF);
    border: 1px solid #E8D49B;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
}

.hero p {
    color: #46556B !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border-color: #E1DCCF !important;
    border-radius: 15px;
}

[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #C49A16;
    border-radius: 12px;
    padding: 10px;
}

[data-testid="stFileUploader"] * {
    color: #172B45 !important;
}

[data-testid="stCaptionContainer"] {
    color: #526177 !important;
}

div[data-baseweb="tab-list"] {
    background: #F1EFE8;
    border-radius: 10px;
    padding: 4px;
    gap: 5px;
}

button[data-baseweb="tab"] {
    color: #34445A !important;
    border-radius: 8px;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: #FFE28A !important;
    color: #172B45 !important;
}

div.stButton > button,
div.stDownloadButton > button {
    min-height: 43px;
    border-radius: 9px;
    font-weight: 600;
    border: 1px solid #C49A16;
    color: #172B45 !important;
    background: #FFFFFF !important;
}

div.stDownloadButton > button[kind="primary"] {
    background: #FFCB35 !important;
    color: #172B45 !important;
    border-color: #D9A900 !important;
}

hr {
    border-color: #E8E0CF !important;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="hero">
    <h1>🧩 Puzzle World</h1>
    <h3>Create Your Own Puzzle</h3>
    <p>
        Unggah gambar, pilih pengaturan, dan lihat
        preview puzzle yang diperbarui secara langsung.
    </p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# IMAGE UPLOAD
# ==================================================

uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"],
    help="Format JPG dan PNG.",
)

if uploaded is None:
    st.info("Unggah gambar untuk mulai membuat puzzle.")
    st.stop()


try:
    image = load_image(uploaded)

except Exception as error:
    st.error(f"Gambar tidak dapat dibaca: {error}")
    st.stop()


# ==================================================
# SETTINGS
# ==================================================

left, middle, right = st.columns(
    [1.05, 1.8, 1.05],
    gap="large",
)

with left:
    with st.container(border=True):
        st.subheader("1. Grid Size")

        m = st.select_slider(
            "Jumlah grid (M × M)",
            options=[3, 4, 5, 6, 7, 8, 9, 10],
            value=5,
        )

        frame_count = m * m - (m - 2) ** 2
        piece_count = (m - 2) ** 2

        st.caption(
            f"Grid {m} × {m} · "
            f"{frame_count} sel frame · "
            f"{piece_count} pieces"
        )

        st.divider()
        st.subheader("2. Puzzle Options")

        rotate = st.checkbox("Rotasi acak")
        augment = st.checkbox("Augmentasi gambar")
        jigsaw = st.checkbox("Pola jigsaw")

        st.divider()
        st.subheader("3. Frame Style")

        st.markdown("**Image Edge Frame**")

        st.caption(
            "Sel terluar mempertahankan gambar asli. "
            "Bagian tengah menjadi pieces."
        )


# ==================================================
# REALTIME PROCESSING
# ==================================================

if image.width < m or image.height < m:
    st.error(
        "Resolusi gambar terlalu kecil untuk grid yang dipilih."
    )
    st.stop()


try:
    with st.spinner("Memperbarui preview..."):

        frame = create_frame(
            image,
            m,
            jigsaw=jigsaw,
        )

        pieces = create_pieces(
            image,
            m,
            jigsaw=jigsaw,
        )

        pieces = process_pieces(
            pieces,
            rotate=rotate,
            augment=augment,
            jigsaw=False,
        )

        preview = create_puzzle_preview(
            image,
            frame,
            pieces,
            m,
        )

        zip_data = create_zip(
            frame,
            pieces,
        )

except Exception as error:
    st.error(f"Gagal memproses puzzle: {error}")
    st.stop()


# ==================================================
# MIDDLE: PREVIEW
# ==================================================

with middle:
    with st.container(border=True):
        st.subheader("Puzzle Preview")

        preview_tab, original_tab = st.tabs(
            ["🧩 Puzzle Preview", "🖼️ Original Image"]
        )

        with preview_tab:
            st.image(
                preview,
                use_container_width=True,
            )

            if jigsaw:
                st.caption(
                    "Mode jigsaw aktif: batas pieces mengikuti "
                    "bentuk pasak dan celah."
                )
            else:
                st.caption(
                    "Mode kotak aktif: pieces mengikuti grid lurus."
                )

        with original_tab:
            st.image(
                image,
                caption=(
                    f"Gambar asli · {image.width} × "
                    f"{image.height} px"
                ),
                use_container_width=True,
            )


# ==================================================
# RIGHT: SUMMARY & DOWNLOAD
# ==================================================

with right:
    with st.container(border=True):
        st.subheader("Your Puzzle")

        st.image(
            image,
            use_container_width=True,
        )

        st.markdown(f"**Grid {m} × {m}**")
        st.caption(f"{piece_count} pieces")
        st.caption(
            "Jigsaw" if jigsaw else "Square Grid"
        )

        st.divider()

        st.download_button(
            "⬇ Download Puzzle (ZIP)",
            data=zip_data,
            file_name="puzzle_world.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )

        st.caption(
            "Berisi frame.png dan folder pieces/."
        )


# ==================================================
# PIECES GALLERY
# ==================================================

st.divider()
st.subheader(f"Puzzle Pieces · {len(pieces)} potongan")

gallery_columns = st.columns(5)

for index, item in enumerate(pieces):
    with gallery_columns[index % 5]:
        st.image(
            item["image"],
            caption=item["name"],
            use_container_width=True,
        )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Puzzle World · Pengolahan Citra Digital · "
    "Puzzle Generator"
)
