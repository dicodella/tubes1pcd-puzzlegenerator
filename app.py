import streamlit as st
from processing import (
    load_image,
    get_grid_boundaries,
    create_frame,
    create_pieces,
    create_puzzle_preview,
    create_zip,
)
from solver import solve_puzzle

# ==================================================
# KONFIGURASI HALAMAN & TEMA
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

st.markdown("""
<div class="hero">
    <h1>🧩 Puzzle World</h1>
    <p>Sistem Pengolahan Citra Digital: Pemecah Citra (Generator) & Penyusun Citra Otomatis (Solver).</p>
</div>
""", unsafe_allow_html=True)

tab_generator, tab_solver = st.tabs(["🧩 Puzzle Generator", "🔍 Puzzle Solver"])

# ==================================================
# TAB 1: PUZZLE GENERATOR
# ==================================================
with tab_generator:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], help="Format JPG dan PNG.")

    if uploaded_file is not None:
        try:
            original_image = load_image(uploaded_file)

            left_col, middle_col, right_col = st.columns([1.05, 1.8, 1.05], gap="large")

            with left_col:
                with st.container(border=True):
                    st.subheader("Parameter Grid")
                    grid_size = st.select_slider("Jumlah grid (M × M)", options=[4, 5, 6, 7, 8, 9, 10], value=4)

                    total_cells = grid_size * grid_size
                    piece_count = (grid_size - 2) ** 2
                    frame_count = total_cells - piece_count
                    x_points, y_points = get_grid_boundaries(original_image.width, original_image.height, grid_size)

                    st.markdown(f"""
                    - **Dimensi Citra:** `{original_image.width} × {original_image.height}` px
                    - **Total Grid:** `{grid_size} × {grid_size}` ({total_cells} sel)
                    - **Sel Bingkai (Frame):** `{frame_count}` sel terluar
                    - **Jumlah Potongan (N):** `{piece_count}` pieces
                    - **Rata-rata Ukuran Sel:** `~{x_points[1]} × {y_points[1]}` px
                    """)

            if original_image.width < grid_size * 10 or original_image.height < grid_size * 10:
                st.error("Resolusi gambar terlalu kecil untuk grid yang dipilih.")
            else:
                frame_image = create_frame(original_image, grid_size)
                pieces = create_pieces(original_image, grid_size)
                preview_image = create_puzzle_preview(frame_image, pieces, grid_size)
                zip_data = create_zip(frame_image, pieces)

                with middle_col:
                    with st.container(border=True):
                        st.subheader("Visualisasi Hasil")
                        tab_preview, tab_frame, tab_original = st.tabs(["🧩 Preview Grid", "🖼️ Bingkai (Frame)", "📷 Citra Asli"])

                        with tab_preview:
                            st.image(preview_image, use_container_width=True)
                            st.caption("Visualisasi garis pembagian grid M × M pada citra utuh.")

                        with tab_frame:
                            st.image(frame_image, use_container_width=True)
                            st.caption("Output 1: Berkas citra bingkai (frame.png) sebagai acuan penyusunan.")

                        with tab_original:
                            st.image(original_image, use_container_width=True)
                            st.caption(f"Citra masukan asli ({original_image.width} × {original_image.height} px).")

                with right_col:
                    with st.container(border=True):
                        st.subheader("Output Berkas")
                        st.markdown(f"""
                        **1. Citra Bingkai:** `frame.png`  
                        **2. Citra Potongan:** `{piece_count} berkas` (`piece_001.png` s.d. `piece_{piece_count:03d}.png`)
                        """)
                        st.divider()
                        st.download_button(
                            "⬇ Download Output (ZIP)",
                            data=zip_data,
                            file_name="puzzle_generator_output.zip",
                            mime="application/zip",
                            type="primary",
                            use_container_width=True,
                        )

                st.divider()
                st.subheader(f"Daftar Potongan Citra ({len(pieces)} Pieces)")

                gallery_cols = st.columns(5)
                for index, item in enumerate(pieces):
                    with gallery_cols[index % 5]:
                        st.image(item["image"], caption=item["name"], use_container_width=True)

        except Exception as error:
            st.error(f"Gagal memproses gambar: {error}")
    else:
        st.info("Unggah gambar untuk mulai membuat puzzle.")

# ==================================================
# TAB 2: PUZZLE SOLVER
# ==================================================
with tab_solver:
    st.subheader("Penyusunan Otomatis (Edge Matching)")
    st.caption("Unggah 1 berkas bingkai (frame.png) dan seluruh N berkas potongan untuk disusun kembali menjadi citra utuh.")

    upload_col1, upload_col2 = st.columns([1, 2])
    with upload_col1:
        frame_upload = st.file_uploader("1. Upload Bingkai (frame.png)", type=["png", "jpg", "jpeg"], key="solver_frame")
    with upload_col2:
        pieces_upload = st.file_uploader("2. Upload Potongan (Pilih semua file pieces sekaligus)", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="solver_pieces")

    if frame_upload and pieces_upload:
        if st.button("🔍 Susun Puzzle (Solve)", type="primary", use_container_width=True):
            with st.spinner("Menghitung kecocokan piksel tepi (MSE)..."):
                try:
                    solved_image = solve_puzzle(frame_upload, pieces_upload)
                    st.success("Puzzle berhasil disusun kembali menjadi citra utuh!")
                    st.image(solved_image, caption=f"Hasil Rekonstruksi Citra ({len(pieces_upload)} potongan)", use_container_width=True)
                except Exception as error:
                    st.error(f"Gagal menyusun puzzle: {error}")
    else:
        st.info("Silakan unggah 1 berkas bingkai dan N berkas potongan terlebih dahulu.")