#!/usr/bin/env python3
"""Rebuild terminal-boot.gif with the Tesla hedgehog in place of Mona ASCII.

The boot sequence, login, and fetch-panel copy are preserved from
assets/terminal-boot-mona.gif (the x0rzavi/gifos animation already on main).
Only the left-hand fetch art is replaced: the white background of
assets/tesla-hedgehog.png is knocked out and the sprite is composited onto
the navy terminal (#0d1117), aligned with the stats panel.

Requires Pillow, NumPy, and gifsicle.
"""

from __future__ import annotations

import subprocess
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
BASE_GIF = ROOT / "assets" / "terminal-boot-mona.gif"
HEDGEHOG_PNG = ROOT / "assets" / "tesla-hedgehog.png"
OUT_GIF = ROOT / "terminal-boot.gif"

NAVY = (13, 17, 23)
# Mona slot relative to the orange stats header. Measured on the source GIF:
# header y=51 with art y=115..335, and the pair shifts together later.
SLOT_TOP_OFFSET = 64
SLOT_HEIGHT = 221
MONA_X0, MONA_X1 = 55, 245
SLOT_CENTER_X = (MONA_X0 + MONA_X1) // 2
SPRITE_COLORS = 48


def _flood_background(near_white: np.ndarray) -> np.ndarray:
    height, width = near_white.shape
    background = np.zeros((height, width), dtype=bool)
    seen = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        queue.append((0, x))
        queue.append((height - 1, x))
    for y in range(height):
        queue.append((y, 0))
        queue.append((y, width - 1))
    while queue:
        y, x = queue.popleft()
        if y < 0 or x < 0 or y >= height or x >= width or seen[y, x]:
            continue
        seen[y, x] = True
        if not near_white[y, x]:
            continue
        background[y, x] = True
        queue.append((y - 1, x))
        queue.append((y + 1, x))
        queue.append((y, x - 1))
        queue.append((y, x + 1))
    return background


def knockout_hedgehog(path: Path) -> Image.Image:
    """Remove the white backdrop, keeping the hedgehog's white fur."""
    rgb = np.array(Image.open(path).convert("RGB"))
    height, width = rgb.shape[:2]
    pixels = rgb.astype(np.float32)
    distance = np.sqrt(((pixels - 255.0) ** 2).sum(axis=2))
    chroma = pixels.max(axis=2) - pixels.min(axis=2)
    near_white = (distance <= 16.0) & (chroma <= 14.0)
    background = _flood_background(near_white)

    alpha = np.full((height, width), 255, np.uint8)
    alpha[background] = 0

    background_img = Image.fromarray(background.astype(np.uint8) * 255, "L")
    fringe = (np.array(background_img.filter(ImageFilter.MaxFilter(3))) > 0) & ~background
    outer = (np.array(background_img.filter(ImageFilter.MaxFilter(5))) > 0) & ~background & ~fringe
    for mask, low, high in ((fringe, 8.0, 28.0), (outer, 18.0, 42.0)):
        fade = np.clip((distance - low) / (high - low), 0.0, 1.0)
        pale = distance < high
        softened = (fade * 255.0).astype(np.uint8)
        selected = mask & pale
        alpha[selected] = np.minimum(alpha[selected], softened[selected])

    sprite = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    ys, xs = np.where(alpha > 8)
    pad = 1
    box = (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(width, int(xs.max()) + 1 + pad),
        min(height, int(ys.max()) + 1 + pad),
    )
    return sprite.crop(box)


def fit_sprite(sprite: Image.Image, colors: int) -> Image.Image:
    """Scale the sprite to the Mona slot, then quantize so every frame shares a palette."""
    target_h = SLOT_HEIGHT - 6
    target_w = max(1, round(sprite.width * (target_h / sprite.height)))
    fitted = sprite.resize((target_w, target_h), Image.Resampling.LANCZOS)
    alpha = fitted.getchannel("A").point(lambda value: 255 if value >= 128 else 0)
    flat = Image.new("RGB", fitted.size, NAVY)
    flat.paste(fitted, mask=alpha)
    reduced = flat.quantize(
        colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    quantized = reduced.convert("RGB")
    out = Image.new("RGBA", fitted.size, NAVY + (0,))
    out.paste(quantized, mask=alpha)
    return out


def _longest_run(rows: np.ndarray) -> tuple[int, int, int] | None:
    best: tuple[int, int, int] | None = None
    y = 0
    height = len(rows)
    while y < height:
        if not rows[y]:
            y += 1
            continue
        start = y
        while y < height and rows[y]:
            y += 1
        run = (y - start, start, y - 1)
        if best is None or run[0] > best[0]:
            best = run
    return best


def mona_bbox(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box of the Mona ASCII block, if this frame has it."""
    navy = np.array(NAVY, dtype=np.int16)
    diff = np.abs(frame[:, 40:300].astype(np.int16) - navy).sum(axis=2) > 30
    run = _longest_run(diff.any(axis=1))
    if run is None or run[0] < 150:
        return None
    _, y0, y1 = run
    band = diff[y0 : y1 + 1]
    if int(band.sum()) < 8000:
        return None
    xs = np.where(band.any(axis=0))[0]
    return int(xs.min()) + 40, y0, int(xs.max()) + 40, y1


def orange_header_y(frame: np.ndarray) -> int | None:
    red = frame[:, 320:, 0].astype(np.int16)
    green = frame[:, 320:, 1].astype(np.int16)
    blue = frame[:, 320:, 2].astype(np.int16)
    orange = (np.abs(red - 228) < 30) & (green < 150) & (blue < 100)
    rows = np.where(orange.sum(axis=1) > 8)[0]
    if len(rows) == 0:
        return None
    return int(rows[0])


def collect_colors(sprite: Image.Image, frames: list[Image.Image]) -> list[tuple[int, int, int]]:
    """Keep every color from the original animation, then add the sprite."""
    colors: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for frame in frames:
        pixels = np.unique(np.array(frame).reshape(-1, 3), axis=0)
        for pixel in pixels:
            color = (int(pixel[0]), int(pixel[1]), int(pixel[2]))
            if color not in seen:
                seen.add(color)
                colors.append(color)
    opaque = np.array(sprite)
    mask = opaque[:, :, 3] > 0
    for pixel in opaque[mask][:, :3]:
        color = (int(pixel[0]), int(pixel[1]), int(pixel[2]))
        if color not in seen:
            seen.add(color)
            colors.append(color)
    if len(colors) > 256:
        raise SystemExit(f"palette overflow: {len(colors)} colors")
    return colors


def apply_exact_palette(frame: Image.Image, colors: list[tuple[int, int, int]]) -> Image.Image:
    """Map RGB pixels onto palette indexes without nearest-color snapping."""
    array = np.array(frame.convert("RGB"))
    packed = (
        array[:, :, 0].astype(np.uint32) << 16
        | array[:, :, 1].astype(np.uint32) << 8
        | array[:, :, 2].astype(np.uint32)
    )
    indexes = np.full(packed.shape, 255, np.uint8)
    for index, (red, green, blue) in enumerate(colors):
        key = (red << 16) | (green << 8) | blue
        indexes[packed == key] = index
    if int((indexes == 255).sum()):
        missing = np.unique(array[indexes == 255].reshape(-1, 3), axis=0)
        raise SystemExit(f"unmapped colors: {missing[:5].tolist()}")
    image = Image.fromarray(indexes, "P")
    flat: list[int] = []
    for color in colors:
        flat.extend(color)
    flat.extend([0, 0, 0] * (256 - len(colors)))
    image.putpalette(flat)
    return image


def composite_frame(frame: Image.Image, sprite: Image.Image) -> Image.Image:
    array = np.array(frame.convert("RGB"))
    box = mona_bbox(array)
    if box is None:
        return frame.convert("RGB")
    x0, y0, x1, y1 = box
    array[y0 : y1 + 1, x0 : x1 + 1] = NAVY
    base = Image.fromarray(array, "RGB")

    header_y = orange_header_y(array)
    if header_y is None:
        top = y0 + max(0, ((y1 - y0 + 1) - sprite.height) // 2)
    else:
        slot_top = header_y + SLOT_TOP_OFFSET
        top = slot_top + max(0, (SLOT_HEIGHT - sprite.height) // 2)
    left = SLOT_CENTER_X - sprite.width // 2
    base.paste(sprite, (left, top), sprite.getchannel("A"))
    return base


def load_base_frames(path: Path) -> tuple[list[Image.Image], list[int]]:
    gif = Image.open(path)
    frames: list[Image.Image] = []
    durations: list[int] = []
    for index in range(gif.n_frames):
        gif.seek(index)
        frames.append(gif.convert("RGB"))
        durations.append(int(gif.info.get("duration", 70)))
    return frames, durations


def write_gif(
    frames: list[Image.Image],
    durations: list[int],
    colors: list[tuple[int, int, int]],
    dest: Path,
) -> None:
    quantized = []
    for frame in frames:
        reduced = apply_exact_palette(frame, colors)
        reduced.info.clear()
        quantized.append(reduced)
    quantized[0].save(
        dest,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    subprocess.run(
        ["gifsicle", "-O3", str(dest), "-o", str(dest)],
        check=True,
    )


def main() -> None:
    sprite = fit_sprite(knockout_hedgehog(HEDGEHOG_PNG), SPRITE_COLORS)
    frames, durations = load_base_frames(BASE_GIF)
    colors = collect_colors(sprite, frames)
    composited = [composite_frame(frame, sprite) for frame in frames]
    write_gif(composited, durations, colors, OUT_GIF)
    with Image.open(OUT_GIF) as gif:
        print(
            f"INFO: Wrote {OUT_GIF} size={gif.size} frames={gif.n_frames} "
            f"bytes={OUT_GIF.stat().st_size}"
        )


if __name__ == "__main__":
    main()
