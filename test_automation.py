import os
import io
import csv
import time
import random
import numpy as np

from processing import load_image, get_grid_boundaries, create_frame, create_pieces
from solver import solve_puzzle


def image_to_memory_file(pil_image, filename="temp.png"):
    """Mengubah objek PIL Image menjadi file virtual di memori untuk input Solver."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    buffer.name = filename
    return buffer


def evaluate_reconstruction(original_image, solved_image, grid_size):
    """
    Membandingkan area dalam (area potongan puzzle) antara gambar asli dan hasil solver.
    MSE = 0.0 berarti seluruh piksel tersusun kembali secara identik.
    """
    orig_array = np.array(original_image.convert("RGB"), dtype=np.float32)
    solv_array = np.array(solved_image.convert("RGB"), dtype=np.float32)

    height, width = orig_array.shape[:2]
    x_points, y_points = get_grid_boundaries(width, height, grid_size)

    y_start, y_end = y_points[1], y_points[grid_size - 1]
    x_start, x_end = x_points[1], x_points[grid_size - 1]

    orig_inner = orig_array[y_start:y_end, x_start:x_end]
    solv_inner = solv_array[y_start:y_end, x_start:x_end]

    mse = float(np.mean((orig_inner - solv_inner) ** 2))
    exact_matches = np.all(orig_inner == solv_inner, axis=-1)
    pixel_accuracy = float(np.mean(exact_matches) * 100.0)

    return mse, pixel_accuracy


def run_full_automation_test(dataset_folder="dataset_uji", output_csv="hasil_pengujian_lengkap.csv"):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = sorted([
        f for f in os.listdir(dataset_folder)
        if f.lower().endswith(valid_extensions)
    ])

    total_images = len(image_files)
    if total_images == 0:
        print(f"Folder '{dataset_folder}' kosong! Masukkan minimal 100 gambar terlebih dahulu.")
        return

    grid_sizes = [4, 5, 6, 7, 8, 9, 10]
    all_results = []
    grid_summary = {}

    print(f"Memulai Automation Testing untuk {total_images} gambar pada Grid 4x4 s.d. 10x10...")
    print(f"Total skenario pengujian: {total_images * len(grid_sizes)} kasus\n")

    for grid_size in grid_sizes:
        passed_count = 0
        failed_count = 0
        mse_list = []
        time_list = []
        piece_count = (grid_size - 2) ** 2

        print(f"--- Menguji Grid {grid_size}x{grid_size} ({piece_count} potongan per gambar) ---")

        for index, filename in enumerate(image_files, start=1):
            file_path = os.path.join(dataset_folder, filename)
            start_time = time.time()

            try:
                original_image = load_image(file_path)

                # Cek resolusi minimum agar setiap sel minimal berukuran 10x10 piksel
                if original_image.width < grid_size * 10 or original_image.height < grid_size * 10:
                    raise ValueError("Resolusi gambar terlalu kecil untuk grid ini")

                # 1. Generator
                frame_image = create_frame(original_image, grid_size)
                pieces = create_pieces(original_image, grid_size)

                # 2. Acak urutan potongan di memori sebelum masuk Solver
                shuffled_pieces = pieces.copy()
                random.shuffle(shuffled_pieces)

                frame_virtual = image_to_memory_file(frame_image, "frame.png")
                pieces_virtual = [
                    image_to_memory_file(item["image"], f"piece_{i}.png")
                    for i, item in enumerate(shuffled_pieces)
                ]

                # 3. Solver
                solved_image = solve_puzzle(frame_virtual, pieces_virtual)

                # 4. Komparasi dengan Gambar Asli
                mse, accuracy = evaluate_reconstruction(original_image, solved_image, grid_size)
                elapsed_time = round(time.time() - start_time, 4)

                is_pass = (mse < 1.0) and (accuracy == 100.0)
                status = "PASS" if is_pass else "FAIL"

                if is_pass:
                    passed_count += 1
                else:
                    failed_count += 1

                mse_list.append(mse)
                time_list.append(elapsed_time)

                all_results.append([
                    f"{grid_size}x{grid_size}", piece_count, index, filename,
                    f"{original_image.width}x{original_image.height}",
                    round(mse, 4), round(accuracy, 2), elapsed_time, status
                ])

            except Exception as error:
                failed_count += 1
                all_results.append([
                    f"{grid_size}x{grid_size}", piece_count, index, filename,
                    "-", "-", 0, 0, f"ERROR: {error}"
                ])

        pass_rate = (passed_count / total_images) * 100
        avg_mse = float(np.mean(mse_list)) if mse_list else 0.0
        avg_time = float(np.mean(time_list)) if time_list else 0.0

        grid_summary[grid_size] = {
            "pieces": piece_count,
            "pass": passed_count,
            "fail": failed_count,
            "pass_rate": pass_rate,
            "avg_mse": avg_mse,
            "avg_time": avg_time,
        }

        print(f"Selesai Grid {grid_size}x{grid_size} -> PASS: {passed_count}/{total_images} ({pass_rate:.2f}%) | Rata-rata MSE: {avg_mse:.2f} | Rata-rata Waktu: {avg_time:.4f}s\n")

    # Simpan seluruh log detail ke CSV
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "Ukuran_Grid", "Jumlah_Potongan", "No_Gambar", "Nama_File",
            "Resolusi", "MSE", "Akurasi_Piksel(%)", "Waktu_Eksekusi(s)", "Status"
        ])
        writer.writerows(all_results)

    # Cetak Tabel Rekapitulasi Akhir
    print("=" * 80)
    print(f"REKAPITULASI AKHIR AUTOMATION TESTING ({total_images} GAMBAR PER GRID)")
    print("=" * 80)
    print(f"{'Grid':<8} | {'Potongan':<10} | {'PASS':<6} | {'FAIL':<6} | {'Pass Rate':<10} | {'Avg MSE':<10} | {'Avg Waktu':<10}")
    print("-" * 80)

    total_all_pass = 0
    for g in grid_sizes:
        s = grid_summary[g]
        total_all_pass += s["pass"]
        print(f"{g}x{g:<6} | {s['pieces']:<10} | {s['pass']:<6} | {s['fail']:<6} | {s['pass_rate']:>8.2f}% | {s['avg_mse']:>10.2f} | {s['avg_time']:>8.4f}s")

    overall_rate = (total_all_pass / (total_images * len(grid_sizes))) * 100
    print("=" * 80)
    print(f"TOTAL KESELURUHAN PASS : {total_all_pass} / {total_images * len(grid_sizes)} skenario ({overall_rate:.2f}%)")
    print(f"File laporan lengkap   : {output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    if not os.path.exists("dataset_uji"):
        os.makedirs("dataset_uji")
        print("Folder 'dataset_uji' telah dibuat. Masukkan minimal 100 gambar ke dalamnya lalu jalankan ulang.")
    else:
        run_full_automation_test(dataset_folder="dataset_uji", output_csv="hasil_pengujian_lengkap.csv")