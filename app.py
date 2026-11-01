import streamlit as st

from processing import (
    load_image,
    create_frame,
    create_pieces,
    process_pieces,
    create_puzzle_preview,
    create_zip,
)

from solver import solve_puzzle

import streamlit as st

# ==================================================
# KONFIGURASI
# ==================================================

st.set_page_config(
    page_title="Puzzle World — Generator & Solver",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #FFFCF5 !important; color: #172B45 !important; }
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; height: 0 !important; }
[data-testid="stMainBlockContainer"], .block-container { max-width: 1500px; padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] { color: #172B45 !important; }
.hero { background: linear-gradient(110deg, #FFF5D8, #FFFFFF); border: 1px solid #E8D49B; border-radius: 16px; padding: 24px 28px; margin-bottom: 20px; }
.hero p { color: #46556B !important; }
[data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF !important; border-color: #E1DCCF !important; border-radius: 15px; }
[data-testid="stFileUploader"] { background: #FFFFFF; border: 1px dashed #C49A16; border-radius: 12px; padding: 10px; }
[data-testid="stFileUploader"] * { color: #172B45 !important; }
[data-testid="stCaptionContainer"] { color: #526177 !important; }
div[data-baseweb="tab-list"] { background: #F1EFE8; border-radius: 10px; padding: 4px; gap: 5px; margin-bottom: 15px; }
button[data-baseweb="tab"] { color: #34445A !important; border-radius: 8px; }
button[data-baseweb="tab"][aria-selected="true"] { background: #FFE28A !important; color: #172B45 !important; }
div.stButton > button, div.stDownloadButton > button { min-height: 43px; border-radius: 9px; font-weight: 600; border: 1px solid #C49A16; color: #172B45 !important; background: #FFFFFF !important; }
div.stDownloadButton > button[kind="primary"], div.stButton > button[kind="primary"] { background: #FFCB35 !important; color: #172B45 !important; border-color: #D9A900 !important; }
hr { border-color: #E8E0CF !important; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="hero">
    <h1>🧩 Puzzle World</h1>
    <h3>Generator & Solver</h3>
    <p>
        Buat potongan teka-teki dari gambar Anda, atau susun kembali potongan menjadi gambar utuh.
    </p>
</div>
""", unsafe_allow_html=True)

tab_gen, tab_sol = st.tabs(["🧩 Puzzle Generator", "🔍 Puzzle Solver"])

# ==================================================
# TAB 1: PUZZLE GENERATOR
# ==================================================
with tab_gen:
    uploaded = st.file_uploader("Upload Image (Generator)", type=["jpg", "jpeg", "png"], help="Format JPG dan PNG.")

    if uploaded is not None:
        try:
            image = load_image(uploaded)
            
            left, middle, right = st.columns([1.05, 1.8, 1.05], gap="large")

            with left:
                with st.container(border=True):
                    st.subheader("1. Grid Size")
                    m = st.select_slider("Jumlah grid (M × M)", options=[3, 4, 5, 6, 7, 8, 9, 10], value=5)
                    frame_count = m * m - (m - 2) ** 2
                    piece_count = (m - 2) ** 2
                    st.caption(f"Grid {m} × {m} · {frame_count} sel frame · {piece_count} pieces")

                    st.divider()
                    st.subheader("2. Puzzle Options")
                    rotate = st.checkbox("Rotasi acak")
                    augment = st.checkbox("Augmentasi gambar")
                    jigsaw = st.checkbox("Pola jigsaw")

                    st.divider()
                    st.subheader("3. Frame Style")
                    st.markdown("**Image Edge Frame**")
                    st.caption("Sel terluar mempertahankan gambar asli. Bagian tengah menjadi pieces.")

            if image.width < m or image.height < m:
                st.error("Resolusi gambar terlalu kecil untuk grid yang dipilih.")
            else:
                with st.spinner("Memperbarui preview..."):
                    frame = create_frame(image, m, jigsaw=jigsaw)
                    pieces = create_pieces(image, m, jigsaw=jigsaw)
                    pieces = process_pieces(pieces, rotate=rotate, augment=augment, jigsaw=jigsaw)
                    preview = create_puzzle_preview(image, frame, pieces, m)
                    zip_data = create_zip(frame, pieces)

                with middle:
                    with st.container(border=True):
                        st.subheader("Puzzle Preview")
                        preview_tab, original_tab = st.tabs(["🧩 Puzzle Preview", "🖼️ Original Image"])

                        with preview_tab:
                            st.image(preview, use_container_width=True)
                            st.caption("Mode jigsaw aktif." if jigsaw else "Mode kotak aktif.")

                        with original_tab:
                            st.image(image, caption=f"Gambar asli · {image.width} × {image.height} px", use_container_width=True)

                with right:
                    with st.container(border=True):
                        st.subheader("Your Puzzle")
                        st.image(image, use_container_width=True)
                        st.markdown(f"**Grid {m} × {m}**")
                        st.caption(f"{piece_count} pieces")
                        st.caption("Jigsaw" if jigsaw else "Square Grid")
                        st.divider()
                        st.download_button(
                            "⬇ Download Puzzle (ZIP)",
                            data=zip_data,
                            file_name="puzzle_world.zip",
                            mime="application/zip",
                            type="primary",
                            use_container_width=True,
                        )
                        st.caption("Berisi frame.png dan folder pieces/.")

                st.divider()
                st.subheader(f"Puzzle Pieces · {len(pieces)} potongan")
                gallery_columns = st.columns(5)
                for index, item in enumerate(pieces):
                    with gallery_columns[index % 5]:
                        st.image(item["image"], caption=item["name"], use_container_width=True)

        except Exception as error:
            st.error(f"Gagal memproses puzzle: {error}")
    else:
        st.info("Unggah gambar untuk mulai membuat puzzle.")


# ==================================================
# TAB 2: PUZZLE SOLVER
# ==================================================
with tab_sol:
    st.markdown("### 🧩 Rekonstruksi Puzzle")
    st.caption("Unggah file bingkai (`frame.png`) beserta seluruh potongan gambarnya untuk menyusun ulang gambar utuh.")
    
    col_upload1, col_upload2 = st.columns([1, 2])
    
    with col_upload1:
        frame_file = st.file_uploader("1. Upload Frame", type=["png", "jpg", "jpeg"], key="frame_upload")
    with col_upload2:
        piece_files = st.file_uploader("2. Upload Pieces (Pilih semua file sekaligus)", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="pieces_upload")

    if frame_file and piece_files:
        if st.button("🔍 Solve Puzzle", type="primary", use_container_width=True):
            with st.spinner("Menerapkan algoritma pengolahan citra untuk menyusun puzzle..."):
                try:
                    solved_image = solve_puzzle(frame_file, piece_files)
                    
                    st.success("Tugas 2 Selesai! Puzzle berhasil dirangkai menjadi citra utuh.")
                    
                    st.image(
                        solved_image, 
                        caption=f"Gambar Utuh Tersusun ({len(piece_files)} Potongan)", 
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Gagal menyusun puzzle. Pastikan jumlah potongan dan bingkai saling sesuai. Error: {e}")
    else:
        st.info("Silakan unggah 1 citra bingkai dan N citra potongan untuk melihat hasilnya.")

# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption("Puzzle World · Pengolahan Citra Digital · Generator & Solver")