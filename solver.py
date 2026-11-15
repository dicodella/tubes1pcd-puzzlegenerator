import math
import cv2
import numpy as np
from PIL import Image

from processing import load_image, get_grid_boundaries


# ==================================================
# 1. FUNGSI HITUNG BIAYA KECOCOKAN TEPI (GRADIENT + COLOR)
# ==================================================
def calculate_edge_error(edge_outer_1, edge_outer_2, piece_edge_1, piece_edge_2):
    """
    Menghitung tingkat ketidakcocokan antara dua tepi potongan gambar.
    Menggabungkan selisih warna langsung dan kelanjutan gradien piksel.
    Semakin kecil nilainya, semakin mulus sambungan gambarnya.
    """
    min_length = min(len(edge_outer_1), len(piece_edge_1))
    if min_length == 0:
        return float("inf")

    a1 = edge_outer_1[:min_length].astype(np.float32)
    a2 = edge_outer_2[:min_length].astype(np.float32)
    b1 = piece_edge_1[:min_length].astype(np.float32)
    b2 = piece_edge_2[:min_length].astype(np.float32)

    # Prediksi piksel tepi berdasarkan kemiringan/gradien warna
    predicted_b1 = a1 + (a1 - a2)
    predicted_a1 = b1 + (b1 - b2)

    # Gabungan selisih warna langsung dan selisih prediksi gradien (L1 + L2)
    direct_diff = np.mean(np.abs(a1 - b1))
    gradient_diff = 0.5 * (
        np.mean(np.abs(predicted_b1 - b1)) + np.mean(np.abs(predicted_a1 - a1))
    )

    return float(direct_diff + gradient_diff)


# ==================================================
# 2. EKSTRAKSI 4 TEPI DARI SETIAP POTONGAN
# ==================================================
def extract_piece_edges(piece_array):
    """Mengambil 2 lapisan piksel terluar dari keempat sisi potongan."""
    return {
        "top": (piece_array[0, :], piece_array[1, :]),
        "bottom": (piece_array[-1, :], piece_array[-2, :]),
        "left": (piece_array[:, 0], piece_array[:, 1]),
        "right": (piece_array[:, -1], piece_array[:, -2]),
    }


# ==================================================
# 3. HITUNG TOTAL ERROR SUATU SLOT TERHADAP TETANGGANYA
# ==================================================
def evaluate_slot_cost(row, col, piece_edges, placed_grid, frame_borders):
    """Menghitung rata-rata error dari sisi-sisi yang bersentuhan dengan tetangga/bingkai."""
    total_error = 0.0
    connected_sides = 0

    # --- Cek Tetangga ATAS ---
    if row == 1:
        f1, f2 = frame_borders["top"][col]
        p1, p2 = piece_edges["top"]
        total_error += calculate_edge_error(f1, f2, p1, p2)
        connected_sides += 1
    elif placed_grid[row - 1][col] is not None:
        n1, n2 = placed_grid[row - 1][col]["edges"]["bottom"]
        p1, p2 = piece_edges["top"]
        total_error += calculate_edge_error(n1, n2, p1, p2)
        connected_sides += 1

    # --- Cek Tetangga BAWAH ---
    max_inner = len(placed_grid) - 2
    if row == max_inner:
        f1, f2 = frame_borders["bottom"][col]
        p1, p2 = piece_edges["bottom"]
        total_error += calculate_edge_error(f1, f2, p1, p2)
        connected_sides += 1
    elif placed_grid[row + 1][col] is not None:
        n1, n2 = placed_grid[row + 1][col]["edges"]["top"]
        p1, p2 = piece_edges["bottom"]
        total_error += calculate_edge_error(n1, n2, p1, p2)
        connected_sides += 1

    # --- Cek Tetangga KIRI ---
    if col == 1:
        f1, f2 = frame_borders["left"][row]
        p1, p2 = piece_edges["left"]
        total_error += calculate_edge_error(f1, f2, p1, p2)
        connected_sides += 1
    elif placed_grid[row][col - 1] is not None:
        n1, n2 = placed_grid[row][col - 1]["edges"]["right"]
        p1, p2 = piece_edges["left"]
        total_error += calculate_edge_error(n1, n2, p1, p2)
        connected_sides += 1

    # --- Cek Tetangga KANAN ---
    if col == max_inner:
        f1, f2 = frame_borders["right"][row]
        p1, p2 = piece_edges["right"]
        total_error += calculate_edge_error(f1, f2, p1, p2)
        connected_sides += 1
    elif placed_grid[row][col + 1] is not None:
        n1, n2 = placed_grid[row][col + 1]["edges"]["left"]
        p1, p2 = piece_edges["right"]
        total_error += calculate_edge_error(n1, n2, p1, p2)
        connected_sides += 1

    if connected_sides == 0:
        return float("inf"), 0

    return total_error / connected_sides, connected_sides


# ==================================================
# 4. FUNGSI UTAMA PUZZLE SOLVER
# ==================================================
def solve_puzzle(frame_file, piece_files):
    # 1. Baca gambar bingkai (frame)
    frame_image = load_image(frame_file)
    canvas_array = np.array(frame_image.convert("RGB"))
    height, width = canvas_array.shape[:2]

    # 2. Baca seluruh potongan gambar
    piece_items = []
    for file in piece_files:
        piece_img = load_image(file)
        piece_arr = np.array(piece_img)
        piece_items.append({
            "array": piece_arr,
            "edges": extract_piece_edges(piece_arr),
        })

    total_pieces = len(piece_items)
    if total_pieces == 0:
        return frame_image

    # 3. Hitung ukuran grid M x M dari jumlah potongan N
    inner_grid = math.isqrt(total_pieces)
    if inner_grid * inner_grid != total_pieces:
        raise ValueError(
            f"Jumlah potongan ({total_pieces}) harus berupa bilangan kuadrat (4, 9, 16, 25, dst)."
        )

    grid_size = inner_grid + 2
    x_points, y_points = get_grid_boundaries(width, height, grid_size)

    # 4. Ekstrak tepi bingkai luar (melewati garis tipis pembatas grid)
    frame_borders = {"top": {}, "bottom": {}, "left": {}, "right": {}}

    for col in range(1, grid_size - 1):
        x_start, x_end = x_points[col], x_points[col + 1]
        # Bingkai atas: baris terakhir yang utuh adalah y_points[1] - 1
        y_top = y_points[1]
        frame_borders["top"][col] = (
            canvas_array[y_top - 1, x_start:x_end],
            canvas_array[max(0, y_top - 2), x_start:x_end],
        )
        # Bingkai bawah: y_points[grid_size - 1] berisi garis grid, jadi gambar dimulai +1
        y_bot = y_points[grid_size - 1]
        frame_borders["bottom"][col] = (
            canvas_array[min(height - 1, y_bot + 1), x_start:x_end],
            canvas_array[min(height - 1, y_bot + 2), x_start:x_end],
        )

    for row in range(1, grid_size - 1):
        y_start, y_end = y_points[row], y_points[row + 1]
        # Bingkai kiri: kolom terakhir yang utuh adalah x_points[1] - 1
        x_left = x_points[1]
        frame_borders["left"][row] = (
            canvas_array[y_start:y_end, x_left - 1],
            canvas_array[y_start:y_end, max(0, x_left - 2)],
        )
        # Bingkai kanan: x_points[grid_size - 1] berisi garis grid, jadi gambar dimulai +1
        x_right = x_points[grid_size - 1]
        frame_borders["right"][row] = (
            canvas_array[y_start:y_end, min(width - 1, x_right + 1)],
            canvas_array[y_start:y_end, min(width - 1, x_right + 2)],
        )

    # 5. Tahap 1: Best-First Placement (Isi slot yang paling yakin terlebih dahulu)
    placed_grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]
    unplaced_indices = list(range(total_pieces))
    empty_slots = [
        (row, col)
        for row in range(1, grid_size - 1)
        for col in range(1, grid_size - 1)
    ]

    while empty_slots and unplaced_indices:
        best_choice = None
        best_priority_score = float("inf")

        for row, col in empty_slots:
            scores = []
            for idx in unplaced_indices:
                avg_error, sides = evaluate_slot_cost(
                    row, col, piece_items[idx]["edges"], placed_grid, frame_borders
                )
                if sides > 0:
                    scores.append((avg_error, sides, idx))

            if not scores:
                continue

            scores.sort(key=lambda item: item[0])
            best_err, sides, best_idx = scores[0]
            second_err = scores[1][0] if len(scores) > 1 else best_err + 50.0

            # Prioritaskan slot dengan error rendah & perbedaan jelas dibanding kandidat ke-2
            confidence_gap = second_err - best_err
            priority_score = best_err - (0.25 * confidence_gap) - (sides * 2.0)

            if priority_score < best_priority_score:
                best_priority_score = priority_score
                best_choice = (row, col, best_idx)

        if best_choice is None:
            # Fallback jika tidak ada slot yang terhubung
            row, col = empty_slots[0]
            best_idx = unplaced_indices[0]
        else:
            row, col, best_idx = best_choice

        placed_grid[row][col] = piece_items[best_idx]
        empty_slots.remove((row, col))
        unplaced_indices.remove(best_idx)

    # 6. Tahap 2: Swap Refinement (Tukar dua potongan jika total error menjadi lebih kecil)
    all_slots = [
        (row, col)
        for row in range(1, grid_size - 1)
        for col in range(1, grid_size - 1)
    ]

    for _ in range(5):
        swapped = False
        for i in range(len(all_slots)):
            for j in range(i + 1, len(all_slots)):
                r1, c1 = all_slots[i]
                r2, c2 = all_slots[j]

                piece_1 = placed_grid[r1][c1]
                piece_2 = placed_grid[r2][c2]

                # Hitung error sebelum ditukar
                err1_before, _ = evaluate_slot_cost(r1, c1, piece_1["edges"], placed_grid, frame_borders)
                err2_before, _ = evaluate_slot_cost(r2, c2, piece_2["edges"], placed_grid, frame_borders)
                cost_before = err1_before + err2_before

                # Coba tukar sementara
                placed_grid[r1][c1] = piece_2
                placed_grid[r2][c2] = piece_1

                err1_after, _ = evaluate_slot_cost(r1, c1, piece_2["edges"], placed_grid, frame_borders)
                err2_after, _ = evaluate_slot_cost(r2, c2, piece_1["edges"], placed_grid, frame_borders)
                cost_after = err1_after + err2_after

                if cost_after < cost_before - 0.5:
                    swapped = True
                else:
                    # Kembalikan jika tidak lebih baik
                    placed_grid[r1][c1] = piece_1
                    placed_grid[r2][c2] = piece_2

        if not swapped:
            break

    # 7. Tempelkan seluruh potongan yang telah tersusun ke kanvas
    for row in range(1, grid_size - 1):
        for col in range(1, grid_size - 1):
            y_start, y_end = y_points[row], y_points[row + 1]
            x_start, x_end = x_points[col], x_points[col + 1]

            target_height = y_end - y_start
            target_width = x_end - x_start

            piece_arr = placed_grid[row][col]["array"]
            if piece_arr.shape[:2] != (target_height, target_width):
                piece_arr = cv2.resize(piece_arr, (target_width, target_height))

            canvas_array[y_start:y_end, x_start:x_end] = piece_arr

    return Image.fromarray(canvas_array)