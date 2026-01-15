# 🚀 Firecrawl Scraper Cheatsheet

## 🔑 **API KEY SETUP**
```bash
# Jouw Firecrawl API Key
export FIRECRAWL_API_KEY="fc-8ac363e5a6e441a7b264e05b1a34727d"

# Of direct in commando's gebruiken met --firecrawl-api-key parameter
```

## 🚀 **COPY & PASTE COMMANDS - Kant & Klaar!**

### **📍 Tilburg Commands:**
```bash
    # Basis Tilburg
    python local_business_scraper_firecrawl.py --loc "Tilburg, NL" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"

    # Tilburg met pause support
    python local_business_scraper_firecrawl.py --loc "Tilburg, NL" --pause-file "pause.flag" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"

    # Tilburg snelle test (klein gebied)
    python local_business_scraper_firecrawl.py --loc "Tilburg, NL" --radius-km 5 --max-sites 50 --no-progress --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"

    # Tilburg productie run (AANBEVOLEN)
    python local_business_scraper_firecrawl.py --loc "Tilburg, NL" --radius-km 40 --max-sites 100000 --pause-file "pause.flag" --out "tilburg_bedrijven.csv" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"
```

### **🏙️ Andere Steden - Copy & Paste:**
```bash
    # Amsterdam
    python local_business_scraper_firecrawl.py --loc "Amsterdam, NL" --radius-km 20 --max-sites 500 --pause-file "pause.flag" --out "amsterdam_bedrijven.csv" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"

    # Rotterdam
    python local_business_scraper_firecrawl.py --loc "Rotterdam, NL" --radius-km 15 --max-sites 300 --pause-file "pause.flag" --out "rotterdam_bedrijven.csv" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"

    # Utrecht
    python local_business_scraper_firecrawl.py --loc "Utrecht, NL" --radius-km 10 --max-sites 200 --pause-file "pause.flag" --out "utrecht_bedrijven.csv" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"

    # Den Haag
    python local_business_scraper_firecrawl.py --loc "Den Haag, NL" --radius-km 12 --max-sites 250 --pause-file "pause.flag" --out "den_haag_bedrijven.csv" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"

    # Eindhoven
    python local_business_scraper_firecrawl.py --loc "Eindhoven, NL" --radius-km 10 --max-sites 150 --pause-file "pause.flag" --out "eindhoven_bedrijven.csv" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"

    # Groningen
    python local_business_scraper_firecrawl.py --loc "Groningen, NL" --radius-km 8 --max-sites 100 --pause-file "pause.flag" --out "groningen_bedrijven.csv" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"
```

### **🎯 Met Coördinaten:**
```bash
    # Exacte locatie (bijvoorbeeld centrum Amsterdam)
    python local_business_scraper_firecrawl.py --coords "52.3676,4.9041" --radius-km 10 --max-sites 200 --pause-file "pause.flag" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"

    # Met Firecrawl API
    export FIRECRAWL_API_KEY="fc-8ac363e5a6e441a7b264e05b1a34727d"
    python local_business_scraper_firecrawl.py --loc "Amsterdam, NL" --radius-km 15 --max-sites 300 --pause-file "pause.flag"
```

---

## ⏸️ **PAUSE & RESUME COMMANDS**

### **🛑 Pauzeren:**
```bash
# Methode 1: Terminal
echo. > pause.flag

# Methode 2: VSCode - maak bestand aan met naam "pause.flag"
```

### **▶️ Doorgaan:**
```bash
# Verwijder pause bestand
del pause.flag

# Of op Linux/Mac:
rm pause.flag
```

### **🔄 Resume na Crash:**
```bash
    # Resume Tilburg
    python local_business_scraper_firecrawl.py --loc "Tilburg, NL" --resume --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"

    # Resume Amsterdam
    python local_business_scraper_firecrawl.py --loc "Amsterdam, NL" --resume --out "amsterdam_bedrijven.csv" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"

    # Resume met specifiek bestand
    python local_business_scraper_firecrawl.py --loc "Rotterdam, NL" --resume --out "rotterdam_bedrijven.csv" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"
```

### **🔄 Resume met Skip Bestaande Data:**
```bash
    # Skip bedrijven uit bestaand CSV
    python local_business_scraper_firecrawl.py --loc "Utrecht, NL" --skip-csv bestaande_bedrijven.csv --out "nieuwe_bedrijven.csv" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"
```

### **💻 VSCode Specifiek:**
```bash
# In VSCode Terminal:
    # 1. Start scraper
    python local_business_scraper_firecrawl.py --loc "Tilburg, NL" --pause-file "pause.flag" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"

# 2. Pauzeren - maak bestand aan in VSCode file explorer
# Rechtsklik → New File → naam: "pause.flag"

# 3. Doorgaan - verwijder het bestand in VSCode file explorer
# Rechtsklik op "pause.flag" → Delete

    # 4. Resume na crash - nieuwe terminal
    python local_business_scraper_firecrawl.py --loc "Tilburg, NL" --resume --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"
```

---

## 📋 **Snelle Start (Origineel)**

### **Basis Gebruik:**
```bash
# Eenvoudigste commando
python local_business_scraper_firecrawl.py --loc "Amsterdam, NL"

# Met radius
python local_business_scraper_firecrawl.py --loc "Rotterdam, NL" --radius-km 10

# Met coördinaten
python local_business_scraper_firecrawl.py --coords "52.3676,4.9041" --radius-km 5
```

---

## 🎯 **Vereiste Parameters**

| Parameter | Beschrijving | Voorbeeld |
|-----------|--------------|-----------|
| `--loc` | Locatie naam | `"Amsterdam, NL"` |
| `--coords` | LAT,LON coördinaten | `"52.3676,4.9041"` |

**⚠️ Je moet EEN van beide opgeven!**

---

## 🔧 **Basis Configuratie**

### **Locatie & Gebied:**
```bash
--loc "Utrecht, NL"                    # Stad/plaats
--coords "52.0907,5.1214"             # Exacte coördinaten
--countrycodes "nl,de,be"             # Landcodes (optioneel)
--radius-km 15                        # Straal in kilometers (default: 30)
```

### **Output:**
```bash
--out "mijn_bedrijven.csv"            # Output bestand (default: bedrijven_firecrawl.csv)
```

---

## ⚡ **Performance Tuning**

### **Scraping Limieten:**
```bash
--max-sites 100                       # Max aantal bedrijven (default: onbeperkt)
--max-pages 3                         # Max pagina's per website (default: 3)
--tile-km 2.0                         # Tegelgrootte in km (default: 2.0)
```

### **Timeouts & Retries:**
```bash
--timeout 15                          # HTTP timeout in seconden (default: 12)
--retries 2                           # Aantal retries (default: 2)
--crawl-delay 1.0                     # Delay tussen requests in seconden (default: 0.7)
```

---

## 🔥 **Firecrawl Optimalisatie**

### **Met API Key (Aanbevolen):**
```bash
# Environment variable
export FIRECRAWL_API_KEY="your_api_key_here"
python local_business_scraper_firecrawl.py --loc "Amsterdam, NL"

# Of direct in commando
python local_business_scraper_firecrawl.py --loc "Amsterdam, NL" --firecrawl-api-key "your_api_key"
```

### **Overpass API:**
```bash
--overpass "https://overpass-api.de/api/interpreter"  # Custom endpoint
--overpass-limit 500                   # Max resultaten per tile (default: 400)
--overpass-query-timeout 30            # Query timeout (default: 25)
--overpass-budget 120                  # Totaal time budget in seconden (default: 75)
```

---

## 🛠️ **Geavanceerde Features**

### **Pause/Resume:**
```bash
--pause-file "pause.flag"             # Maak dit bestand aan om te pauzeren
--resume                              # Ga verder met bestaand .part bestand
```

### **Autosave:**
```bash
--autosave-every 50                   # Flush elke N bedrijven (default: 200)
--autosave-percent 10                 # Flush bij elke N% voortgang (default: 0)
--hard-fsync                          # Forceer disk sync (trager maar veiliger)
```

### **Skip Bestaande Data:**
```bash
--skip-csv bestaande.csv              # Skip bedrijven uit dit bestand
--skip-csv file1.csv file2.csv        # Meerdere bestanden
```

### **Progress & Logging:**
```bash
--no-progress                         # Geen progress bars
--progress-interval 10                # Heartbeat interval in seconden (default: 5)
```

---

## 📊 **Praktische Voorbeelden**

### **1. Snelle Test:**
```bash
python local_business_scraper_firecrawl.py \
  --loc "Tilburg, NL" \
  --radius-km 3 \
  --max-sites 10 \
  --no-progress
```

### **2. Productie Run:**
```bash
python local_business_scraper_firecrawl.py \
  --loc "Amsterdam, NL" \
  --radius-km 20 \
  --max-sites 500 \
  --out "amsterdam_bedrijven.csv" \
  --autosave-every 100 \
  --autosave-percent 5
```

### **3. Met Firecrawl API:**
```bash
export FIRECRAWL_API_KEY="fc-your_key_here"
python local_business_scraper_firecrawl.py \
  --loc "Den Haag, NL" \
  --radius-km 15 \
  --max-pages 5 \
  --timeout 20 \
  --crawl-delay 1.0
```

### **4. Resume Na Crash:**
```bash
python local_business_scraper_firecrawl.py \
  --loc "Rotterdam, NL" \
  --radius-km 25 \
  --resume \
  --out "rotterdam_bedrijven.csv"
```

### **5. Skip Bestaande Data:**
```bash
python local_business_scraper_firecrawl.py \
  --loc "Utrecht, NL" \
  --radius-km 10 \
  --skip-csv bestaande_bedrijven.csv \
  --out "nieuwe_bedrijven.csv"
```

---

## 🎛️ **Alle Parameters Overzicht**

| Categorie | Parameter | Default | Beschrijving |
|-----------|-----------|---------|--------------|
| **Locatie** | `--loc` | - | Locatie naam |
| | `--coords` | - | LAT,LON coördinaten |
| | `--countrycodes` | - | Landcodes filter |
| | `--radius-km` | 30.0 | Straal in kilometers |
| | `--tile-km` | 2.0 | Tegelgrootte in kilometers |
| **Limieten** | `--max-sites` | - | Max aantal bedrijven |
| | `--max-pages` | 3 | Max pagina's per website |
| **Performance** | `--timeout` | 12.0 | HTTP timeout |
| | `--retries` | 2 | Aantal retries |
| | `--crawl-delay` | 0.7 | Delay tussen requests |
| **Overpass** | `--overpass` | auto | Overpass endpoint |
| | `--overpass-limit` | 400 | Max resultaten per tile |
| | `--overpass-query-timeout` | 25 | Query timeout |
| | `--overpass-budget` | 75.0 | Totaal time budget |
| **Firecrawl** | `--firecrawl-api-key` | env | Firecrawl API key |
| **Output** | `--out` | bedrijven_firecrawl.csv | Output bestand |
| **Autosave** | `--autosave-every` | 200 | Flush elke N bedrijven |
| | `--autosave-percent` | 0.0 | Flush bij elke N% |
| | `--hard-fsync` | False | Forceer disk sync |
| **Control** | `--pause-file` | pause.flag | Pause bestand |
| | `--resume` | False | Resume van .part |
| | `--skip-csv` | - | Skip bestaande CSV's |
| **UI** | `--no-progress` | False | Geen progress bars |
| | `--progress-interval` | 5.0 | Heartbeat interval |
| **Robots** | `--robots-fail-closed` | False | Strict robots.txt |

---

## 🚨 **Troubleshooting**

### **Veelvoorkomende Fouten:**

**❌ "Geef --loc of --coords op"**
```bash
# Oplossing: Voeg locatie toe
python local_business_scraper_firecrawl.py --loc "Amsterdam, NL"
```

**❌ "Geen bruikbaar Overpass endpoint"**
```bash
# Oplossing: Probeer andere endpoint
python local_business_scraper_firecrawl.py --loc "Amsterdam, NL" --overpass "https://overpass.kumi.systems/api/interpreter"
```

**❌ "Rate limited"**
```bash
# Oplossing: Verhoog delays
python local_business_scraper_firecrawl.py --loc "Amsterdam, NL" --crawl-delay 2.0 --retries 3
```

### **Performance Tips:**

**🐌 Te langzaam?**
```bash
# Verklein radius en tiles
--radius-km 10 --tile-km 1.0 --max-sites 100
```

**💾 Te veel geheugen?**
```bash
# Meer autosave
--autosave-every 50 --autosave-percent 5
```

**🌐 Te veel websites?**
```bash
# Beperk scraping
--max-pages 2 --timeout 10 --crawl-delay 1.5
```

---

## 📈 **Output Format**

De scraper produceert CSV met deze kolommen:
```csv
name,category,lat,lon,address,website,email,phone,source_location
```

**Voorbeeld:**
```csv
Pondres,kantoor:company,51.580895,5.0640772,Kraaivenstraat 19 5048AB Tilburg,https://pondres.nl,info@pondres.nl,+31 88 9494100,"Tilburg, NL"
```

---

## 🎯 **Quick Commands - Copy & Paste**

### **🚀 Start Commands:**
```bash
    # Tilburg (AANBEVOLEN)
    python local_business_scraper_firecrawl.py --loc "Tilburg, NL" --radius-km 15 --max-sites 200 --pause-file "pause.flag" --out "tilburg_bedrijven.csv" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"

    # Amsterdam (groot)
    python local_business_scraper_firecrawl.py --loc "Amsterdam, NL" --radius-km 20 --max-sites 500 --pause-file "pause.flag" --out "amsterdam_bedrijven.csv" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"

    # Test run (klein)
    python local_business_scraper_firecrawl.py --loc "Tilburg, NL" --radius-km 3 --max-sites 10 --no-progress --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"

    # Met Firecrawl API
    export FIRECRAWL_API_KEY="fc-8ac363e5a6e441a7b264e05b1a34727d"
    python local_business_scraper_firecrawl.py --loc "Rotterdam, NL" --radius-km 15 --max-sites 300 --pause-file "pause.flag" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"
```

### **⏸️ Pause/Resume Commands:**
```bash
# Pauzeren
echo. > pause.flag

# Doorgaan
del pause.flag

# Resume na crash
python local_business_scraper_firecrawl.py --loc "Tilburg, NL" --resume --out "tilburg_bedrijven.csv"

# Resume Amsterdam
python local_business_scraper_firecrawl.py --loc "Amsterdam, NL" --resume --out "amsterdam_bedrijven.csv"
```

### **🔄 Advanced Commands:**
```bash
    # Skip bestaande data
    python local_business_scraper_firecrawl.py --loc "Utrecht, NL" --skip-csv bestaande_bedrijven.csv --out "nieuwe_bedrijven.csv" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"

# Met autosave
python local_business_scraper_firecrawl.py --loc "Den Haag, NL" --autosave-every 50 --autosave-percent 5 --pause-file "pause.flag"

# Snelle test met beperkingen
python local_business_scraper_firecrawl.py --loc "Eindhoven, NL" --radius-km 5 --max-sites 20 --max-pages 2 --timeout 10 --no-progress
```

### **🎯 MAXIMUM BEZOEKEN - Meer Sites Scrapen:**
```bash
    # Tilburg - MAXIMUM bezochte sites (AANBEVOLEN voor veel data)
    python local_business_scraper_firecrawl.py --loc "Tilburg, NL" --radius-km 25 --max-sites 1000 --max-pages 5 --timeout 20 --retries 3 --crawl-delay 1.0 --pause-file "pause.flag" --out "tilburg_max_sites.csv" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"

    # Amsterdam - MAXIMUM bezochte sites
    python local_business_scraper_firecrawl.py --loc "Amsterdam, NL" --radius-km 30 --max-sites 2000 --max-pages 5 --timeout 25 --retries 3 --crawl-delay 1.2 --pause-file "pause.flag" --out "amsterdam_max_sites.csv" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"

    # Rotterdam - MAXIMUM bezochte sites
    python local_business_scraper_firecrawl.py --loc "Rotterdam, NL" --radius-km 25 --max-sites 1500 --max-pages 5 --timeout 20 --retries 3 --crawl-delay 1.0 --pause-file "pause.flag" --out "rotterdam_max_sites.csv" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"

    # Utrecht - MAXIMUM bezochte sites
    python local_business_scraper_firecrawl.py --loc "Utrecht, NL" --radius-km 20 --max-sites 1200 --max-pages 5 --timeout 20 --retries 3 --crawl-delay 1.0 --pause-file "pause.flag" --out "utrecht_max_sites.csv" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"
```

### **🔥 ULTRA MAXIMUM - Alle Steden:**
```bash
    # Den Haag - ULTRA MAXIMUM
    python local_business_scraper_firecrawl.py --loc "Den Haag, NL" --radius-km 30 --max-sites 2500 --max-pages 7 --timeout 30 --retries 4 --crawl-delay 1.5 --autosave-every 100 --autosave-percent 2 --pause-file "pause.flag" --out "den_haag_ultra_max.csv" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"

    # Eindhoven - ULTRA MAXIMUM
    python local_business_scraper_firecrawl.py --loc "Eindhoven, NL" --radius-km 25 --max-sites 2000 --max-pages 7 --timeout 30 --retries 4 --crawl-delay 1.5 --autosave-every 100 --autosave-percent 2 --pause-file "pause.flag" --out "eindhoven_ultra_max.csv" --firecrawl-api-key "fc-8ac363e5a6e441a7b264e05b1a34727d"
```

---

## 📚 **Meer Info**

- **Logs**: Bekijk `scraper.log` voor gedetailleerde informatie
- **Checkpoints**: `.part` bestanden worden automatisch aangemaakt
- **Metrics**: Aan het einde zie je een overzicht van alle resultaten
- **Pause**: Maak `pause.flag` aan om te pauzeren, verwijder om door te gaan

**🎉 Veel succes met scraper!**
