import math
import cv2
import numpy as np
from PIL import Image

from processing import load_image, get_grid_boundaries

INF = 1e5                   
SHAPE_PENALTY = 1000.0    
SIZE_PENALTY = 300.0        
DC_WEIGHT = 0.5        
ALPHA_T = 128
GAIN_RIDGE = 16.0          
GAIN_LO, GAIN_HI = 0.70, 1.43


# ==================================================
# 1. UTILITAS KECIL
# ==================================================

def _resample(arr, n):
    L = arr.shape[0]
    if L == n:
        return arr
    idx = np.clip(np.round(np.linspace(0, L - 1, n)).astype(int), 0, L - 1)
    return arr[idx]


def _is_jigsaw(images):
    for im in images:
        if np.array(im.getchannel("A")).min() < 250:
            return True
    return False

# ==================================================
# 2. DETEKSI BENTUK PASAK / CELAH (jigsaw)
# ==================================================

def _tab_signature(alpha, pad):
    if pad == 0:
        return (0, 0, 0, 0)

    H, W = alpha.shape
    cw, ch = W - 2 * pad, H - 2 * pad
    op = alpha > ALPHA_T

    xs = slice(pad + int(cw * 0.40), pad + int(cw * 0.60) + 1)
    ys = slice(pad + int(ch * 0.40), pad + int(ch * 0.60) + 1)
    d = max(1, pad // 2)

    def classify(outer_fill, inner_fill):
        if outer_fill > 0.5:
            return 1
        if inner_fill < 0.5:
            return -1
        return 0

    top = classify(op[pad - d:pad, xs].mean(), op[pad:pad + d, xs].mean())
    bottom = classify(op[pad + ch:pad + ch + d, xs].mean(),
                      op[pad + ch - d:pad + ch, xs].mean())
    left = classify(op[ys, pad - d:pad].mean(), op[ys, pad:pad + d].mean())
    right = classify(op[ys, pad + cw:pad + cw + d].mean(),
                     op[ys, pad + cw - d:pad + cw].mean())
    return (top, bottom, left, right)


# ==================================================
# 3. EKSTRAKSI TEPI
# ==================================================

class Variant:

    def __init__(self, rgba, pad, Lw, Lh):
        self.rgba = rgba
        H, W = rgba.shape[:2]
        self.cw, self.ch = W - 2 * pad, H - 2 * pad
        rgb = rgba[..., :3].astype(np.float32)
        ok = rgba[..., 3] > ALPHA_T
        p, cw, ch = pad, self.cw, self.ch

        def hz(row1, row2):
            r1, r2 = rgb[row1, p:p + cw], rgb[row2, p:p + cw]
            v = ok[row1, p:p + cw] & ok[row2, p:p + cw]
            return _resample(r1, Lw), _resample(r2, Lw), _resample(v, Lw)

        def vt(col1, col2):
            r1, r2 = rgb[p:p + ch, col1], rgb[p:p + ch, col2]
            v = ok[p:p + ch, col1] & ok[p:p + ch, col2]
            return _resample(r1, Lh), _resample(r2, Lh), _resample(v, Lh)

        self.top = hz(p, p + 1)
        self.bottom = hz(p + ch - 1, p + ch - 2)
        self.left = vt(p, p + 1)
        self.right = vt(p + cw - 1, p + cw - 2)
        self.tabs = _tab_signature(rgba[..., 3], pad)


def _edge_cost(a1, a2, va, b1, b2, vb):
    u = b1 - 0.5 * (b2 - b1)                               
    w = a1 + 0.5 * (a1 - a2)                             
    v = (va & vb)[..., None].astype(np.float32)            
    cnt = v.sum(axis=-2)                                  
    safe = np.maximum(cnt, 1.0)
    mu = (u * v).sum(axis=-2) / safe                       
    mw = (w * v).sum(axis=-2) / safe
    uc, wc = u - mu[..., None, :], w - mw[..., None, :]
    num = (uc * wc * v).sum(axis=-2) + GAIN_RIDGE * safe
    den = (uc * uc * v).sum(axis=-2) + GAIN_RIDGE * safe
    g = np.clip(num / den, GAIN_LO, GAIN_HI)       
    dev = (np.abs(wc - g[..., None, :] * uc) * v).sum(axis=-2) / safe
    cost = dev.mean(-1) + DC_WEIGHT * np.abs(mu - mw).mean(-1)
    cost = np.where(cnt[..., 0] < 3, INF, cost)
    return cost


def _stack(variants, side):
    e = [getattr(v, side) for v in variants]
    return (np.stack([x[0] for x in e]), np.stack([x[1] for x in e]),
            np.stack([x[2] for x in e]))


# ==================================================
# 4. BEAM SEARCH + LOCAL SEARCH
# ==================================================

def _spiral_order(n):
    order, top, bot, left, right = [], 0, n - 1, 0, n - 1
    while top <= bot and left <= right:
        order += [top * n + c for c in range(left, right + 1)]
        order += [r * n + right for r in range(top + 1, bot + 1)]
        if top < bot:
            order += [bot * n + c for c in range(right - 1, left - 1, -1)]
        if left < right:
            order += [r * n + left for r in range(bot - 1, top, -1)]
        top, bot, left, right = top + 1, bot - 1, left + 1, right - 1
    return order


def _beam_search(n, V, TB, LR, ftop, fleft, fbottom, fright, SP, width, expand):
    N, S = V // 4, n * n
    piece_of = np.arange(V) // 4
    TBt, LRt = np.ascontiguousarray(TB.T), np.ascontiguousarray(LR.T)
    order = _spiral_order(n)

    frontier, done = [], set()
    for s in order:
        done.add(s)
        fr = []
        for q in done:
            r, c = divmod(q, n)
            nb = [q - n if r > 0 else None, q + n if r < n - 1 else None,
                  q - 1 if c > 0 else None, q + 1 if c < n - 1 else None]
            if any(t is not None and t not in done for t in nb):
                fr.append(q)
        frontier.append(sorted(fr))

    beam = [(0.0, (-1,) * S, 0, np.zeros(N, dtype=bool))]
    for step, s in enumerate(order):
        r, c = divmod(s, n)
        base = SP[s].copy()
        if r == 0:
            base += ftop[c]
        if c == 0:
            base += fleft[r]
        if r == n - 1:
            base += fbottom[c]
        if c == n - 1:
            base += fright[r]

        best = {}
        k = min(expand, V - 1)
        for cost, placed, bits, used in beam:
            vec = base.copy()
            if r > 0 and placed[s - n] >= 0:
                vec += TB[placed[s - n]]
            if r < n - 1 and placed[s + n] >= 0:
                vec += TBt[placed[s + n]]
            if c > 0 and placed[s - 1] >= 0:
                vec += LR[placed[s - 1]]
            if c < n - 1 and placed[s + 1] >= 0:
                vec += LRt[placed[s + 1]]
            vec[used[piece_of]] = np.inf
            for v in np.argpartition(vec, k)[:k]:
                if not np.isfinite(vec[v]):
                    continue
                lst = list(placed)
                lst[s] = int(v)
                npl = tuple(lst)
                nbits = bits | (1 << int(v // 4))
                key = (nbits, tuple(npl[q] for q in frontier[step]))
                nc = cost + float(vec[v])
                if key not in best or nc < best[key][0]:
                    best[key] = (nc, npl, nbits)

        ranked = sorted(best.values(), key=lambda t: t[0])[:width]
        beam = [(nc, npl, nbits,
                 np.array([(nbits >> i) & 1 for i in range(N)], dtype=bool))
                for nc, npl, nbits in ranked]
        if not beam:
            raise RuntimeError("Beam search gagal menemukan susunan.")

    return beam[0][1], beam[0][0]


def _local_cost(place, s, n, TB, LR, ftop, fleft, fbottom, fright, SP):
    r, c = divmod(s, n)
    v = place[s]
    t = SP[s][v]
    t += TB[place[s - n]][v] if r > 0 else ftop[c][v]
    t += LR[place[s - 1]][v] if c > 0 else fleft[r][v]
    t += TB[v][place[s + n]] if r < n - 1 else fbottom[c][v]
    t += LR[v][place[s + 1]] if c < n - 1 else fright[r][v]
    return t


def _refine(place, n, mats, max_sweeps=6):
    place = list(place)
    S = n * n
    for _ in range(max_sweeps):
        improved = False

        for s in range(S):                  
            base = place[s] - place[s] % 4
            cur = _local_cost(place, s, n, *mats)
            for k in range(4):
                v = base + k
                if v == place[s]:
                    continue
                old = place[s]
                place[s] = v
                new = _local_cost(place, s, n, *mats)
                if new < cur - 1e-6:
                    cur, improved = new, True
                else:
                    place[s] = old

        for a in range(S):                     
            for b in range(a + 1, S):
                pa, pb = place[a], place[b]
                ba, bb = pa - pa % 4, pb - pb % 4
                before = _local_cost(place, a, n, *mats) + _local_cost(place, b, n, *mats)
                best_gain, best_pair = 1e-6, None
                for ka in range(4):
                    for kb in range(4):
                        place[a], place[b] = bb + kb, ba + ka
                        after = _local_cost(place, a, n, *mats) + _local_cost(place, b, n, *mats)
                        if before - after > best_gain:
                            best_gain, best_pair = before - after, (place[a], place[b])
                if best_pair:
                    place[a], place[b] = best_pair
                    improved = True
                else:
                    place[a], place[b] = pa, pb
        if not improved:
            break
    return tuple(place)


# ==================================================
# 5. SOLVER UTAMA
# ==================================================

def _solve(frame_file, piece_files, beam_width=200, expand=8, refine=True):
    frame_img = load_image(frame_file)
    frame_rgb = np.array(frame_img).astype(np.float32)
    W, H = frame_img.size

    imgs = [Image.open(pf).convert("RGBA") for pf in piece_files]
    N = len(imgs)
    if N == 0:
        return frame_img, ()

    n = math.isqrt(N)
    if n * n != N:
        raise ValueError(
            f"Jumlah potongan ({N}) bukan bilangan kuadrat sempurna. "
            "Pastikan semua potongan diunggah."
        )
    m = n + 2
    x, y = get_grid_boundaries(W, H, m)

    jigsaw = _is_jigsaw(imgs)
    pad = max(2, int(min(W / m, H / m) * 0.15)) if jigsaw else 0
    skip = 0 if jigsaw else 1    

    Lw, Lh = int(round(W / m)), int(round(H / m))
    nom_w, nom_h = W / m, H / m
    tol = 2

    # ---------- varian rotasi ----------
    variants = []
    valid_variant = np.zeros(N * 4, dtype=bool)
    for i, im in enumerate(imgs):
        arr = np.array(im)
        rots = [np.ascontiguousarray(np.rot90(arr, k)) for k in range(4)]
        ok = [abs((r.shape[1] - 2 * pad) - nom_w) <= tol and
              abs((r.shape[0] - 2 * pad) - nom_h) <= tol for r in rots]
        if not any(ok):  # kalau ukuran tidak cocok sama sekali
            nw, nh = int(round(nom_w)) + 2 * pad, int(round(nom_h)) + 2 * pad
            rots = [cv2.resize(r, (nw, nh), interpolation=cv2.INTER_AREA) for r in rots]
            ok = [True] * 4
        for k in range(4):
            variants.append(Variant(rots[k], pad, Lw, Lh))
            valid_variant[i * 4 + k] = ok[k]
    V = len(variants)

    # ---------- tepi bingkai ----------
    def fr_hz(row1, row2, c):
        sl = slice(x[c], x[c + 1])
        a1, a2 = _resample(frame_rgb[row1, sl], Lw), _resample(frame_rgb[row2, sl], Lw)
        v = np.ones(a1.shape[0], dtype=bool)
        if skip:
            v[0] = False
        return a1, a2, v

    def fr_vt(col1, col2, r):
        sl = slice(y[r], y[r + 1])
        a1, a2 = _resample(frame_rgb[sl, col1], Lh), _resample(frame_rgb[sl, col2], Lh)
        v = np.ones(a1.shape[0], dtype=bool)
        if skip:
            v[0] = False
        return a1, a2, v

    # ---------- matriks biaya ----------
    T_top, T_bot, T_left, T_right = (_stack(variants, s) for s in ("top", "bottom", "left", "right"))

    TB = np.empty((V, V), dtype=np.float32) 
    LR = np.empty((V, V), dtype=np.float32)   
    for i in range(V):
        TB[i] = _edge_cost(T_bot[0][i], T_bot[1][i], T_bot[2][i], T_top[0], T_top[1], T_top[2])
        LR[i] = _edge_cost(T_right[0][i], T_right[1][i], T_right[2][i], T_left[0], T_left[1], T_left[2])

    ftop = np.empty((n, V), dtype=np.float32)
    fbottom = np.empty((n, V), dtype=np.float32)
    fleft = np.empty((n, V), dtype=np.float32)
    fright = np.empty((n, V), dtype=np.float32)
    for j in range(n):
        c = r = j + 1
        a1, a2, va = fr_hz(y[1] - 1, y[1] - 2, c)                       # bingkai atas
        ftop[j] = _edge_cost(a1, a2, va, T_top[0], T_top[1], T_top[2])
        b1, b2, vb = fr_hz(y[m - 1] + skip, y[m - 1] + skip + 1, c)     # bingkai bawah
        fbottom[j] = _edge_cost(T_bot[0], T_bot[1], T_bot[2], b1, b2, vb)
        a1, a2, va = fr_vt(x[1] - 1, x[1] - 2, r)                       # bingkai kiri
        fleft[j] = _edge_cost(a1, a2, va, T_left[0], T_left[1], T_left[2])
        b1, b2, vb = fr_vt(x[m - 1] + skip, x[m - 1] + skip + 1, r)     # bingkai kanan
        fright[j] = _edge_cost(T_right[0], T_right[1], T_right[2], b1, b2, vb)

    # ---------- penalti bentuk jigsaw ----------
    if jigsaw:
        T = np.array([v.tabs for v in variants])      
        TB += np.where(T[:, 1][:, None] * T[:, 0][None, :] == -1, 0.0, SHAPE_PENALTY)
        LR += np.where(T[:, 3][:, None] * T[:, 2][None, :] == -1, 0.0, SHAPE_PENALTY)
        ftop += np.where(T[:, 0] == 0, 0.0, SHAPE_PENALTY)[None, :]
        fbottom += np.where(T[:, 1] == 0, 0.0, SHAPE_PENALTY)[None, :]
        fleft += np.where(T[:, 2] == 0, 0.0, SHAPE_PENALTY)[None, :]
        fright += np.where(T[:, 3] == 0, 0.0, SHAPE_PENALTY)[None, :]

    # ---------- penalti ukuran slot & varian tidak valid ----------
    vcw = np.array([v.cw for v in variants])
    vch = np.array([v.ch for v in variants])
    SP = np.zeros((n * n, V), dtype=np.float32)
    for s in range(n * n):
        r, c = divmod(s, n)
        sw, sh = x[c + 2] - x[c + 1], y[r + 2] - y[r + 1]
        SP[s] = np.where((vcw == sw) & (vch == sh), 0.0, SIZE_PENALTY)
        SP[s][~valid_variant] = INF

    # ---------- cari susunan ----------
    mats = (TB, LR, ftop, fleft, fbottom, fright, SP)
    place, _ = _beam_search(n, V, *mats[:6], SP, beam_width, expand)
    if refine and n > 1:
        place = _refine(place, n, mats)

    # ---------- render ----------
    canvas = frame_img.convert("RGBA")
    for s, v in enumerate(place):
        r, c = divmod(s, n)
        r, c = r + 1, c + 1
        sw, sh = x[c + 1] - x[c], y[r + 1] - y[r]
        rgba = variants[v].rgba.copy()
        tw, th = sw + 2 * pad, sh + 2 * pad
        if rgba.shape[1] != tw or rgba.shape[0] != th:
            rgba = cv2.resize(rgba, (tw, th), interpolation=cv2.INTER_LINEAR)
        rgba[..., 3] = np.where(rgba[..., 3] > ALPHA_T, 255, 0)
        canvas.alpha_composite(Image.fromarray(rgba, "RGBA"), (x[c] - pad, y[r] - pad))

    out = np.array(canvas.convert("RGB"))

    # penghilang garis
    if skip:
        mask = np.zeros((H, W), dtype=np.uint8)
        for xi in x[1:-1]:
            mask[:y[1], xi] = 255
            mask[y[m - 1]:, xi] = 255
        for yi in y[1:-1]:
            mask[yi, :x[1]] = 255
            mask[yi, x[m - 1]:] = 255
        mask[:, x[m - 1]] = 255
        mask[y[m - 1], :] = 255
        mask[y[1]:y[m - 1], x[1]:x[m - 1]] = 0 
        out = cv2.inpaint(out, mask, 2, cv2.INPAINT_TELEA)

    return Image.fromarray(out), tuple(int(v) for v in place)


def solve_puzzle(frame_file, piece_files):
    image, _ = _solve(frame_file, piece_files)
    return image