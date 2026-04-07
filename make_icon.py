#!/usr/bin/env python3
"""
Generate the Doza Assist app icon as a PNG at multiple sizes
for creating a macOS .icns file.
"""

import struct
import zlib
import os
import math

def create_png(width, height, pixels):
    """Create a PNG file from raw RGBA pixel data."""
    def chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack('>I', len(data)) + c + crc

    header = b'\x89PNG\r\n\x1a\n'
    ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0))

    raw = b''
    for y in range(height):
        raw += b'\x00'  # filter byte
        for x in range(width):
            idx = (y * width + x) * 4
            raw += bytes(pixels[idx:idx+4])

    idat = chunk(b'IDAT', zlib.compress(raw, 9))
    iend = chunk(b'IEND', b'')

    return header + ihdr + idat + iend


def draw_icon(size):
    """Draw the Doza Assist icon at the given size."""
    pixels = [0] * (size * size * 4)

    # Colors
    bg_r, bg_g, bg_b = 0x0a, 0x0a, 0x0c       # dark background
    accent_r, accent_g, accent_b = 0x4a, 0x9e, 0xff  # blue accent
    letter_r, letter_g, letter_b = 0xe8, 0xe8, 0xec   # light text

    corner_radius = size * 0.22
    margin = size * 0.02

    def in_rounded_rect(x, y, x0, y0, x1, y1, r):
        """Check if point is inside a rounded rectangle."""
        if x < x0 or x > x1 or y < y0 or y > y1:
            return False, 0.0
        # Corner checks
        corners = [
            (x0 + r, y0 + r),
            (x1 - r, y0 + r),
            (x0 + r, y1 - r),
            (x1 - r, y1 - r),
        ]
        for cx, cy in corners:
            dx = abs(x - cx)
            dy = abs(y - cy)
            if (x < x0 + r or x > x1 - r) and (y < y0 + r or y > y1 - r):
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > r + 0.5:
                    return False, 0.0
                elif dist > r - 0.5:
                    return True, max(0.0, min(1.0, r + 0.5 - dist))
                break
        return True, 1.0

    def set_pixel(x, y, r, g, b, a=255):
        if 0 <= x < size and 0 <= y < size:
            idx = (y * size + x) * 4
            if a < 255:
                # Alpha blend
                existing_r = pixels[idx]
                existing_g = pixels[idx+1]
                existing_b = pixels[idx+2]
                af = a / 255.0
                r = int(existing_r * (1 - af) + r * af)
                g = int(existing_g * (1 - af) + g * af)
                b = int(existing_b * (1 - af) + b * af)
                a = max(pixels[idx+3], a)
            pixels[idx] = r
            pixels[idx+1] = g
            pixels[idx+2] = b
            pixels[idx+3] = a

    # Draw background rounded rectangle
    for y in range(size):
        for x in range(size):
            inside, alpha = in_rounded_rect(x, y, margin, margin, size - margin - 1, size - margin - 1, corner_radius)
            if inside:
                set_pixel(x, y, bg_r, bg_g, bg_b, int(alpha * 255))

    # Draw the "D" logo - blue accent square with D shape
    # Position the logo centered
    logo_size = size * 0.65
    logo_x = (size - logo_size) / 2
    logo_y = (size - logo_size) / 2
    logo_radius = logo_size * 0.18

    # Blue background square for logo
    for y in range(size):
        for x in range(size):
            inside, alpha = in_rounded_rect(x, y, logo_x, logo_y, logo_x + logo_size, logo_y + logo_size, logo_radius)
            if inside:
                set_pixel(x, y, accent_r, accent_g, accent_b, int(alpha * 255))

    # Draw the "D" letter shape (dark on blue)
    # Left vertical bar of D
    bar_x = logo_x + logo_size * 0.18
    bar_y = logo_y + logo_size * 0.18
    bar_w = logo_size * 0.15
    bar_h = logo_size * 0.64

    for y in range(int(bar_y), int(bar_y + bar_h)):
        for x in range(int(bar_x), int(bar_x + bar_w)):
            set_pixel(x, y, 0, 0, 0, 230)

    # Curved part of D (semicircle on right side)
    d_center_x = bar_x + bar_w
    d_center_y = bar_y + bar_h / 2
    d_outer_r = bar_h / 2
    d_inner_r = d_outer_r - logo_size * 0.14

    for y in range(int(logo_y), int(logo_y + logo_size)):
        for x in range(int(d_center_x), int(logo_x + logo_size)):
            dx = x - d_center_x
            dy = y - d_center_y
            dist = math.sqrt(dx*dx + dy*dy)
            if d_inner_r < dist < d_outer_r:
                # Anti-aliasing at edges
                alpha = 230
                if dist > d_outer_r - 1:
                    alpha = int(230 * (d_outer_r - dist))
                elif dist < d_inner_r + 1:
                    alpha = int(230 * (dist - d_inner_r))
                if alpha > 0:
                    set_pixel(x, y, 0, 0, 0, min(230, max(0, alpha)))

    # Small waveform lines (audio visualization accent) below the D
    wave_y = logo_y + logo_size + size * 0.08
    wave_center_x = size / 2
    wave_heights = [0.03, 0.06, 0.09, 0.07, 0.04, 0.08, 0.05, 0.03]
    wave_spacing = size * 0.035
    start_x = wave_center_x - (len(wave_heights) - 1) * wave_spacing / 2

    for i, h in enumerate(wave_heights):
        wx = int(start_x + i * wave_spacing)
        wh = int(size * h)
        wy = int(wave_y - wh / 2)
        bar_width = max(2, int(size * 0.012))
        for dy in range(wh):
            for dx in range(bar_width):
                set_pixel(wx + dx, wy + dy, accent_r, accent_g, accent_b, 180)

    return pixels


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'icon_build')
    os.makedirs(output_dir, exist_ok=True)

    # Generate at standard macOS icon sizes
    sizes = [16, 32, 64, 128, 256, 512, 1024]

    for s in sizes:
        print(f"Generating {s}x{s} icon...")
        pixels = draw_icon(s)
        png_data = create_png(s, s, pixels)
        out_path = os.path.join(output_dir, f'icon_{s}x{s}.png')
        with open(out_path, 'wb') as f:
            f.write(png_data)
        print(f"  Saved: {out_path}")

    print(f"\nAll icons generated in {output_dir}")


if __name__ == '__main__':
    main()
