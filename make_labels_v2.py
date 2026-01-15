#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Labelprinter 9X – CSV → 89x36mm labels @203DPI
GEEN overbodige witruimte — compacte layout, exact 4 regels:
1) Bedrijfnaam
2) Straat + huisnummer
3) Postcode + stad
4) Land

Gebruik:
    python make_labels_v2.py "tilburg_firecrawl_max.csv"

Vereist:
    pip install pillow pandas
    
Voor 2D Data Matrix Code (PostNL standaard):
    pip install pylibdmtx
    
    Op Linux/Mac kan libdmtx systeemlibs vereisen:
    - Ubuntu/Debian: sudo apt-get install libdmtx0a libdmtx-dev
    - macOS: brew install libdmtx
    
(Windows print optioneel) pip install pywin32
"""

import os
import sys
import re
import platform
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io

# Try to import Data Matrix libraries
try:
    from pylibdmtx import pylibdmtx as dmtx
    HAS_DMTX = True
except ImportError:
    HAS_DMTX = False
    
HAS_SEGNO = False
HAS_QR = False

# ======================= Config =======================
DPI = 203
WIDTH_PX  = 716    # ≈ 89 mm
HEIGHT_PX = 290    # ≈ 36 mm

# Compacte marges en strakkere tussenruimte voor 89x36mm labels
MARGIN_LEFT   = 20
MARGIN_TOP    = 25
LINE_GAP      = 8       # extra ruimte tussen regels
LETTER_SPACING = 2      # extra spatie tussen letters in pixels

# Doel-tekengroottes (worden automatisch verkleind indien nodig) - aangepast voor 89x36mm
NAME_SIZE  = 45
ADDR_SIZE  = 32
CITY_SIZE  = 32
MIN_SIZE   = 16         # niet kleiner dan dit

OUTPUT_DIR = Path("label_output")
OUTPUT_PDF = OUTPUT_DIR / "labels.pdf"

AUTO_PRINT = False
PRINTER_NAME_HINTS = ["Labelprinter 9X", "LabelPrinter 9X", "9X"]

# CSV-veld mapping (niet meer nodig - we gebruiken direct de kolomnamen)

# =================== Lettertypes ======================
def draw_text_with_spacing(draw, xy, text, font, fill, letter_spacing=0):
    """Tekst tekenen met extra spatie tussen letters"""
    x, y = xy
    for char in text:
        draw.text((x, y), char, font=font, fill=fill)
        # Bereken breedte van het karakter
        try:
            bbox = draw.textbbox((0, 0), char, font=font)
            char_width = bbox[2] - bbox[0]
        except AttributeError:
            char_width, _ = draw.textsize(char, font=font)
        x += char_width + letter_spacing

def load_font(sz):
    # Probeer verschillende fonts die altijd beschikbaar zijn
    font_paths = [
        # Windows fonts
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf", 
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/verdana.ttf",
        # Linux fonts
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # Mac fonts
        "/System/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        # Fallback
        "DejaVuSans.ttf"
    ]
    
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, sz)
        except Exception:
            continue
    
    # Laatste fallback - gebruik default font
    return ImageFont.load_default()

def fit_text(draw, text, target_size, max_width, letter_spacing=0, min_size=MIN_SIZE):
    """Schaal lettertype omlaag totdat text in max_width past, rekening houdend met letterafstand."""
    text = str(text or "")
    size = max(target_size, min_size)
    while size >= min_size:
        f = load_font(size)
        # Gebruik textbbox voor nieuwere Pillow versies
        try:
            bbox = draw.textbbox((0, 0), text, font=f)
            w = bbox[2] - bbox[0]
        except AttributeError:
            # Fallback voor oudere versies
            w, _ = draw.textsize(text, font=f)
        
        # Voeg letterafstand toe aan de berekende breedte
        effective_width = w + (len(text) - 1) * letter_spacing if len(text) > 0 else 0
        
        if effective_width <= max_width:
            return f
        size -= 2
    return load_font(min_size)

# ================== NL-adres parsing ==================
ADDR_RE = re.compile(
    r"""
    ^\s*
    (?P<street>[^\d,]+?)          # straat
    \s*
    (?P<number>\d+\w?)?           # nummer (+toevoeging)
    [,\s]*
    (?P<postcode>\d{4}\s?[A-Za-z]{2})?   # 1234AB of 1234 AB
    [,\s]*
    (?P<city>[A-Za-zÀ-ÿ\-\s']+)?  # stad
    \s*$
    """,
    re.VERBOSE
)

def parse_nl_address(raw):
    """Verbeterde adres parsing - zorgt ervoor dat postcodes altijd op aparte regel staan."""
    raw = str(raw or "").strip()
    if not raw:
        return "", ""
    
    # Verbeterde regex voor complexe Nederlandse adressen
    # Dit pattern zoekt specifiek naar postcode patroon en splitst daar
    improved_pattern = re.compile(
        r"""
        ^\s*
        (.+?)                    # Alles voor de postcode (straat + huisnummer)
        \s+                      # Spatie tussen straat en postcode
        (\d{4}\s?[A-Z]{2})       # Nederlandse postcode (1234 AB of 1234AB)
        \s+                      # Spatie tussen postcode en stad
        (.+)                     # Stad/plaats
        \s*$
        """, re.VERBOSE | re.IGNORECASE
    )
    
    # Probeer eerst de verbeterde regex
    match = improved_pattern.match(raw)
    if match:
        street_part = match.group(1).strip()
        postcode = match.group(2).strip()
        city = match.group(3).strip()
        
        # Validatie: straat deel moet een huisnummer bevatten
        if re.search(r'\d', street_part):
            return street_part, f"{postcode} {city}".strip()
    
    # Fallback naar originele regex
    m = ADDR_RE.match(raw)
    if m:
        street  = (m.group("street")  or "").strip()
        number  = (m.group("number")  or "").strip()
        pc      = (m.group("postcode") or "").replace(" ", "").upper()
        city    = (m.group("city")    or "").strip()
        line2 = f"{street} {number}".strip()
        line3 = f"{pc} {city}".strip()
        return line2, line3
    
    # Laatste fallback - zoek handmatig naar postcode en splits daar
    postcode_match = re.search(r'(\d{4}\s?[A-Z]{2})', raw, re.IGNORECASE)
    if postcode_match:
        street_part = raw[:postcode_match.start()].strip()
        postcode_and_city = raw[postcode_match.start():].strip()
        return street_part, postcode_and_city
    
    return raw, ""

# ============= Data Matrix Code Generation ==============
MATRIX_CODE_SIZE = 60  # Size in pixels for the matrix code (verkleind voor rechtsonder)
MATRIX_MARGIN_RIGHT = 10
MATRIX_MARGIN_BOTTOM = 10

# Vaste afmetingen voor alle codes (zodat ze allemaal dezelfde grootte hebben)
MATRIX_WIDTH = 60   # Breedte in pixels
MATRIX_HEIGHT = 60  # Hoogte in pixels

def generate_matrix_code(address_text):
    """
    Genereer 2D Data Matrix ECC200 van het adres.
    Geen QR-fallback: als pylibdmtx ontbreekt, return None.
    """
    if not address_text or not address_text.strip():
        return None

    if not HAS_DMTX:
        return None

    try:
        # Encode naar Data Matrix (ECC200)
        enc = dmtx.encode(address_text.encode("utf-8"))
        # pylibdmtx levert RGB-bytes terug
        img = Image.frombytes("RGB", (enc.width, enc.height), enc.pixels)
        # Voor strak zwart-wit werken we in L en drempelen
        img = img.convert("L")
        img = img.point(lambda p: 0 if p < 128 else 255, mode="1")  # binaire bitmap
        # Voeg klein kader (quiet zone) van 1 module rondom toe
        module_qz = 2  # pixels, vergroot mee bij scaling
        w, h = img.size
        qz = Image.new("1", (w + 2*module_qz, h + 2*module_qz), 1)  # 1 = wit
        qz.paste(img, (module_qz, module_qz))
        return qz.convert("L")  # terug naar 'L' voor consistentie met labelcanvas
    except Exception:
        return None

# ================== Printer detectie ==================
def detect_printer():
    os_name = platform.system().lower()

    if "darwin" in os_name or "linux" in os_name:
        try:
            out = subprocess.check_output(["lpstat", "-p"], text=True, stderr=subprocess.STDOUT)
            candidates = []
            for line in out.splitlines():
                if line.startswith("printer "):
                    pname = line.split()[1]
                    candidates.append(pname)
            for p in candidates:
                if any(h.lower() in p.lower() for h in PRINTER_NAME_HINTS):
                    return p
            try:
                default_p = subprocess.check_output(["lpstat", "-d"], text=True)
                if ":" in default_p:
                    return default_p.split(":")[1].strip()
            except Exception:
                pass
            return candidates[0] if candidates else None
        except Exception:
            return None

    if "windows" in os_name:
        try:
            import win32print
            printers = [p[2] for p in win32print.EnumPrinters(
                win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS)]
            for p in printers:
                if any(h.lower() in p.lower() for h in PRINTER_NAME_HINTS):
                    return p
            try:
                return win32print.GetDefaultPrinter()
            except Exception:
                return printers[0] if printers else None
        except Exception:
            return None

    return None

# ====================== Rendering ======================
def render_label_compact(name, address_raw):
    """
    Zeer compacte layout linksboven, minimale witruimte.
    3 regels: Naam / Straat+nr / Postcode Stad
    """
    img = Image.new("L", (WIDTH_PX, HEIGHT_PX), color=255)
    d = ImageDraw.Draw(img)

    max_width = WIDTH_PX - (MARGIN_LEFT * 2)

    # 1) Bedrijfnaam
    name_txt = (str(name or "").strip()) or "—"
    f_name = fit_text(d, name_txt, NAME_SIZE, max_width, LETTER_SPACING)
    y = MARGIN_TOP
    draw_text_with_spacing(d, (MARGIN_LEFT, y), name_txt, f_name, 0, LETTER_SPACING)
    y += f_name.size + LINE_GAP

    # 2) Straat + huisnummer
    line2, line3 = parse_nl_address(address_raw)

    f_addr = fit_text(d, line2, ADDR_SIZE, max_width, LETTER_SPACING)
    draw_text_with_spacing(d, (MARGIN_LEFT, y), line2, f_addr, 0, LETTER_SPACING)
    y += f_addr.size + LINE_GAP

    # 3) Postcode + stad
    f_city = fit_text(d, line3, CITY_SIZE, max_width, LETTER_SPACING)
    draw_text_with_spacing(d, (MARGIN_LEFT, y), line3, f_city, 0, LETTER_SPACING)
    y += f_city.size + LINE_GAP

    # 4) Land
    f_country = fit_text(d, "Nederland", CITY_SIZE, max_width, LETTER_SPACING)
    draw_text_with_spacing(d, (MARGIN_LEFT, y), "Nederland", f_country, 0, LETTER_SPACING)
    
    # 5) 2D Matrix Code rechtsonder
    matrix_img = generate_matrix_code(address_raw)
    if matrix_img:
        # Forceer exacte vaste afmetingen voor alle codes (elk label hetzelfde)
        matrix_img = matrix_img.resize((MATRIX_WIDTH, MATRIX_HEIGHT), Image.NEAREST)  # GEEN LANCZOS! Dit maakt modules onscherp
        
        # Position: right bottom corner (vaste afmetingen)
        x_pos = WIDTH_PX - MATRIX_WIDTH - MATRIX_MARGIN_RIGHT
        y_pos = HEIGHT_PX - MATRIX_HEIGHT - MATRIX_MARGIN_BOTTOM
        
        # Paste the matrix code on the label
        img.paste(matrix_img, (x_pos, y_pos))

    return img

# ======================== Print ========================
def print_file(filepath, printer_name):
    os_name = platform.system().lower()
    if not printer_name:
        print("Printer niet gevonden – bestand is gegenereerd, maar niet geprint.")
        return

    if "darwin" in os_name or "linux" in os_name:
        try:
            subprocess.check_call(["lpr", "-P", printer_name, "-o", "media=Custom.102x150mm", str(filepath)])
            print(f"Afgedrukt naar: {printer_name}")
        except Exception as e:
            print(f"Printen via lpr mislukt: {e}")
        return

    if "windows" in os_name:
        try:
            import win32api
            win32api.ShellExecute(0, "printto", str(filepath), f'"{printer_name}"', ".", 0)
            print(f"Afdruktaak gestuurd naar: {printer_name}")
        except Exception as e:
            print(f"Windows-print mislukt: {e}")

# ========================= Main ========================
def main(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"CSV niet gevonden: {csv_path}")
        sys.exit(1)

    printer = detect_printer()
    print(f"Printer gedetecteerd: {printer if printer else '— geen gevonden —'}")

    try:
        df = pd.read_csv(csv_path)
        print(f"CSV geladen: {len(df)} bedrijven gevonden")
    except Exception as e:
        print(f"CSV kon niet worden gelezen: {e}")
        sys.exit(2)

    # Check of benodigde kolommen aanwezig zijn
    required_columns = ["name", "address"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Ontbrekende kolommen in CSV: {missing_columns}")
        print(f"Beschikbare kolommen: {list(df.columns)}")
        sys.exit(3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = []
    processed_count = 0
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        address_raw = str(row.get("address", "")).strip()
        
        # Skip rijen zonder naam of adres
        if not name or not address_raw:
            continue
            
        images.append(render_label_compact(name, address_raw))
        processed_count += 1

    if not images:
        print("Geen geldige data om te verwerken (geen naam of adres).")
        sys.exit(4)

    print(f"Verwerkt: {processed_count} labels")
    rgb = [im.convert("RGB") for im in images]
    rgb[0].save(OUTPUT_PDF, save_all=True, append_images=rgb[1:])
    print(f"PDF klaar: {OUTPUT_PDF.resolve()}")

    if AUTO_PRINT:
        print_file(OUTPUT_PDF, printer)

def test_address_parsing():
    """Test functie voor adres parsing"""
    test_addresses = [
        "Jellinghausstraat 26-05 5048AZ Tilburg",
        "Daniël Josephus Jittastraat 8 5042MX Tilburg", 
        "Wagnerplein 52 5011LR Tilburg",
        "Waalstraat 10 5046AP Tilburg",
        "Ringbaan-Noord 177a 5046AA Tilburg"
    ]
    
    print("=== ADRES PARSING TEST ===")
    for addr in test_addresses:
        line2, line3 = parse_nl_address(addr)
        print(f"Origineel: {addr}")
        print(f"Regel 2:   {line2}")
        print(f"Regel 3:   {line3}")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Gebruik: python make_labels_v2.py <pad/naar/bestand.csv>")
        print("Voorbeeld: python make_labels_v2.py tilburg_firecrawl_max.csv")
        print("Test:      python make_labels_v2.py --test")
        sys.exit(2)
    
    if sys.argv[1] == "--test":
        test_address_parsing()
    else:
        main(sys.argv[1])
