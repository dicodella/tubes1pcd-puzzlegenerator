
from PIL import Image, ImageOps, ImageDraw, ImageEnhance
import numpy as np
import cv2
import random
import io
import zipfile


# ==================================================
# 1. MEMBACA GAMBAR
# ==================================================

def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


# ==================================================
# 2. MEMBUAT BATAS GRID
# ==================================================

def get_grid_boundaries(width, height, m):
    x = [round(i * width / m) for i in range(m + 1)]
    y = [round(i * height / m) for i in range(m + 1)]
    return x, y


# ==================================================
# 3. MEMBUAT FRAME
# ==================================================

def create_frame(image, m, jigsaw=False):
    image = image.convert("RGB")
    original = np.array(image)

    height, width = original.shape[:2]
    x, y = get_grid_boundaries(width, height, m)

    frame = np.full(
        (height, width, 3),
        (250, 248, 240),
        dtype=np.uint8
    )

    for row in range(m):
        for col in range(m):
            outer = (
                row == 0
                or row == m - 1
                or col == 0
                or col == m - 1
            )

            if outer:
                frame[
                    y[row]:y[row + 1],
                    x[col]:x[col + 1]
                ] = original[
                    y[row]:y[row + 1],
                    x[col]:x[col + 1]
                ]

    if not jigsaw:
        for xi in x:
            cv2.line(
                frame,
                (xi, 0),
                (xi, height - 1),
                (225, 205, 155),
                1
            )

        for yi in y:
            cv2.line(
                frame,
                (0, yi),
                (width - 1, yi),
                (225, 205, 155),
                1
            )

    return Image.fromarray(frame)


# ==================================================
# 4. MEMBUAT MASKER JIGSAW
# ==================================================

def make_mask(width, height, tabs, radius):
    mask = Image.new("L", (width, height), 0)

    pad = radius
    x0 = pad
    y0 = pad
    x1 = width - pad
    y1 = height - pad

    def make_side(start, end, side, tab):
        points = []
        steps = 32

        for i in range(steps + 1):
            t = i / steps

            if 0.30 <= t <= 0.70 and tab != 0:
                u = (t - 0.30) / 0.40
                offset = (
                    radius
                    * np.sin(np.pi * u)
                    * tab
                )
            else:
                offset = 0

            px = start[0] + (end[0] - start[0]) * t
            py = start[1] + (end[1] - start[1]) * t

            if side == "top":
                py -= offset
            elif side == "right":
                px += offset
            elif side == "bottom":
                py += offset
            elif side == "left":
                px -= offset

            points.append((round(px), round(py)))

        return points

    points = []
    points += make_side(
        (x0, y0), (x1, y0), "top", tabs["top"]
    )
    points += make_side(
        (x1, y0), (x1, y1), "right", tabs["right"]
    )
    points += make_side(
        (x1, y1), (x0, y1), "bottom", tabs["bottom"]
    )
    points += make_side(
        (x0, y1), (x0, y0), "left", tabs["left"]
    )

    draw = ImageDraw.Draw(mask)
    draw.polygon(points, fill=255)

    return mask


# ==================================================
# 5. MEMOTONG GAMBAR MENJADI PIECES
# ==================================================

def create_pieces(image, m, jigsaw=False):
    width, height = image.size
    x, y = get_grid_boundaries(width, height, m)

    pieces = []

    cell_w = width / m
    cell_h = height / m
    radius = max(2, int(min(cell_w, cell_h) * 0.15))

    # Pola batas horizontal dan vertikal.
    horizontal_edges = {
        (row, col): random.choice([-1, 1])
        for row in range(m - 1)
        for col in range(m)
    }

    vertical_edges = {
        (row, col): random.choice([-1, 1])
        for row in range(m)
        for col in range(m - 1)
    }

    for row in range(1, m - 1):
        for col in range(1, m - 1):
            x1, x2 = x[col], x[col + 1]
            y1, y2 = y[row], y[row + 1]

            cell_width = x2 - x1
            cell_height = y2 - y1

            if not jigsaw:
                piece = image.crop((x1, y1, x2, y2))
                paste_x, paste_y = x1, y1

            else:
                tabs = {
                    "top": (
                        -horizontal_edges[(row - 1, col)]
                        if row > 1 else 0
                    ),
                    "bottom": (
                        horizontal_edges[(row, col)]
                        if row < m - 2 else 0
                    ),
                    "left": (
                        -vertical_edges[(row, col - 1)]
                        if col > 1 else 0
                    ),
                    "right": (
                        vertical_edges[(row, col)]
                        if col < m - 2 else 0
                    ),
                }

                pad = radius
                
                crop = image.crop((
                    x1 - pad,
                    y1 - pad,
                    x2 + pad,
                    y2 + pad
                )).convert("RGBA")

                mask = make_mask(
                    cell_width + 2 * pad,
                    cell_height + 2 * pad,
                    tabs,
                    radius
                )

                crop.putalpha(mask)

                piece = crop
                paste_x = x1 - pad
                paste_y = y1 - pad

            pieces.append({
                "name": f"piece_{len(pieces) + 1:03d}.png",
                "image": piece,
                "row": row,
                "col": col,
                "paste_x": paste_x,
                "paste_y": paste_y,
                "jigsaw": jigsaw
            })

    return pieces


# ==================================================
# 6. ROTASI DAN AUGMENTASI
# ==================================================

def rotate_piece(piece):
    angle = random.choice([90, 180, 270])
    
    return piece.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False
    )


def augment_piece(piece):
    piece = ImageEnhance.Brightness(piece).enhance(
        random.uniform(0.85, 1.15)
    )

    piece = ImageEnhance.Contrast(piece).enhance(
        random.uniform(0.85, 1.15)
    )

    if piece.mode == "RGBA":
        rgb = piece.convert("RGB")
        rgb = ImageEnhance.Color(rgb).enhance(
            random.uniform(0.85, 1.15)
        )
        rgb.putalpha(piece.getchannel("A"))
        piece = rgb
    else:
        piece = ImageEnhance.Color(piece).enhance(
            random.uniform(0.85, 1.15)
        )

    return piece


def process_pieces(
    pieces,
    rotate=False,
    augment=False,
    jigsaw=False
):
    results = []

    for item in pieces:
        piece = item["image"].copy()

        if augment:
            piece = augment_piece(piece)

        if rotate:
            piece = rotate_piece(piece)

        results.append({
            **item,
            "image": piece
        })

    return results


# ==================================================
# 7. MEMBUAT PREVIEW
# ==================================================


def create_puzzle_preview(image, frame, pieces, m):
    from PIL import Image, ImageDraw, ImageFilter, ImageChops

    width, height = image.size
    preview = frame.convert("RGBA")

    is_jigsaw = any(
        item.get("jigsaw", False)
        for item in pieces
    )

    for item in pieces:
        piece = item["image"].convert("RGBA")
        px = item["paste_x"]
        py = item["paste_y"]

        preview.alpha_composite(piece, (px, py))

        if is_jigsaw:
            alpha = piece.getchannel("A")

            eroded = alpha.filter(ImageFilter.MinFilter(3))
            edge = ImageChops.subtract(alpha, eroded)

            edge = edge.filter(
                ImageFilter.MaxFilter(3)
            )

            outline = Image.new(
                "RGBA",
                piece.size,
                (255, 193, 7, 255)
            )
            outline.putalpha(edge)

            preview.alpha_composite(
                outline,
                (px, py)
            )

    if not is_jigsaw:
        x, y = get_grid_boundaries(width, height, m)
        draw = ImageDraw.Draw(preview)
        grid_color = (30, 30, 30, 255)

        for xi in x:
            draw.line(
                (xi, 0, xi, height - 1),
                fill=grid_color,
                width=1
            )

        for yi in y:
            draw.line(
                (0, yi, width - 1, yi),
                fill=grid_color,
                width=1
            )

    return preview.convert("RGB")


# ==================================================
# 8. DOWNLOAD ZIP
# ==================================================

def create_zip(frame, pieces):
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zip_file:

        frame_buffer = io.BytesIO()
        frame.save(frame_buffer, format="PNG")

        zip_file.writestr(
            "frame.png",
            frame_buffer.getvalue()
        )

        for item in pieces:
            piece_buffer = io.BytesIO()

            item["image"].save(
                piece_buffer,
                format="PNG"
            )

            zip_file.writestr(
                f"pieces/{item['name']}",
                piece_buffer.getvalue()
            )

    return buffer.getvalue()
