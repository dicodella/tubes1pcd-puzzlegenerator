from PIL import Image, ImageOps
import numpy as np
import cv2
import io
import zipfile


# ==================================================
# 1. MEMBACA & STANDARISASI CITRA
# ==================================================
def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


# ==================================================
# 2. MENGHITUNG KOORDINAT BATAS GRID (M x M)
# ==================================================
def get_grid_boundaries(width, height, grid_size):
    x_points = [round(i * width / grid_size) for i in range(grid_size + 1)]
    y_points = [round(i * height / grid_size) for i in range(grid_size + 1)]
    return x_points, y_points


# ==================================================
# 3. MEMBUAT BINGKAI (FRAME)
# ==================================================
def create_frame(image, grid_size):
    original_array = np.array(image.convert("RGB"))
    height, width = original_array.shape[:2]
    x_points, y_points = get_grid_boundaries(width, height, grid_size)

    # Buat kanvas kosong berwarna krem terang (RGB: 250, 248, 240)
    frame_array = np.full((height, width, 3), (250, 248, 240), dtype=np.uint8)

    # Salin piksel citra asli HANYA pada sel terluar
    for row in range(grid_size):
        for col in range(grid_size):
            is_outer_cell = (
                row == 0 or
                row == grid_size - 1 or
                col == 0 or
                col == grid_size - 1
            )

            if is_outer_cell:
                y_start, y_end = y_points[row], y_points[row + 1]
                x_start, x_end = x_points[col], x_points[col + 1]
                frame_array[y_start:y_end, x_start:x_end] = original_array[y_start:y_end, x_start:x_end]

    # Gambar garis bantu grid pada bingkai
    for x_pos in x_points:
        cv2.line(frame_array, (x_pos, 0), (x_pos, height - 1), (225, 205, 155), 1)
    for y_pos in y_points:
        cv2.line(frame_array, (0, y_pos), (width - 1, y_pos), (225, 205, 155), 1)

    return Image.fromarray(frame_array)


# ==================================================
# 4. MEMOTONG AREA DALAM MENJADI N PIECES (BERURUTAN)
# ==================================================
def create_pieces(image, grid_size):
    original_array = np.array(image.convert("RGB"))
    height, width = original_array.shape[:2]
    x_points, y_points = get_grid_boundaries(width, height, grid_size)

    pieces = []

    # Potong bagian dalam grid secara berurutan dari kiri-atas ke kanan-bawah
    for row in range(1, grid_size - 1):
        for col in range(1, grid_size - 1):
            x_start, x_end = x_points[col], x_points[col + 1]
            y_start, y_end = y_points[row], y_points[row + 1]

            piece_array = original_array[y_start:y_end, x_start:x_end]
            piece_image = Image.fromarray(piece_array)

            pieces.append({
                "name": f"piece_{len(pieces) + 1:03d}.png",
                "image": piece_image,
                "row": row,
                "col": col,
            })

    return pieces


# ==================================================
# 5. MEMBUAT VISUALISASI PREVIEW (BERURUTAN)
# ==================================================
def create_puzzle_preview(frame_image, pieces, grid_size):
    preview_array = np.array(frame_image.convert("RGB"))
    height, width = preview_array.shape[:2]
    x_points, y_points = get_grid_boundaries(width, height, grid_size)

    # Tempelkan setiap potongan langsung ke koordinat (row, col) aslinya
    for item in pieces:
        row = item["row"]
        col = item["col"]
        piece_array = np.array(item["image"])

        y_start, y_end = y_points[row], y_points[row + 1]
        x_start, x_end = x_points[col], x_points[col + 1]

        preview_array[y_start:y_end, x_start:x_end] = piece_array

    # Gambar garis batas antar potongan agar terlihat jelas di preview
    for x_pos in x_points:
        cv2.line(preview_array, (x_pos, 0), (x_pos, height - 1), (40, 40, 40), 1)
    for y_pos in y_points:
        cv2.line(preview_array, (0, y_pos), (width - 1, y_pos), (40, 40, 40), 1)

    return Image.fromarray(preview_array)


# ==================================================
# 6. MENGEMAS OUTPUT KE DALAM ZIP
# ==================================================
def create_zip(frame_image, pieces):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        frame_buffer = io.BytesIO()
        frame_image.save(frame_buffer, format="PNG")
        zip_file.writestr("frame.png", frame_buffer.getvalue())

        for item in pieces:
            piece_buffer = io.BytesIO()
            item["image"].save(piece_buffer, format="PNG")
            zip_file.writestr(f"pieces/{item['name']}", piece_buffer.getvalue())

    return zip_buffer.getvalue()