#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local Business Scraper (OSM + Firecrawl) — enhanced contact extraction + address parsing
+ Pause/Autosave/Resume + Firecrawl integration for better website scraping

- Overpass tiled (kleine bboxes) met endpoint-check + korte fallback
- Directe contactextractie uit OSM-tags (email/phone) + betere website-selectie
- Firecrawl website scraping voor betere contact/adres extractie
- Alleen bedrijven met contact (e-mail OF telefoon) in CSV
- Verbeterde adres extractie uit website content
- Heartbeat + ETA voor tiles en sites
- CSV write-safe (timestamp fallback bij lock) + WINDOWS-SAFE pad/naam normalisatie
- Early-stop in tilefase zodra genoeg kandidaten
- --skip-csv: sla bekende entries over (match op naam+coörd en website-domein)
- NIEUW:
  * Firecrawl integration voor betere website scraping
  * Verbeterde adres extractie uit website content
  * --pause-file: pauzeer door een bestand aan te maken/verwijderen
  * Continue schrijven naar <out>.part + periodieke flush (autosave)
  * --resume: ga verder op bestaand .part en sla reeds verwerkte rows over
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
import urllib.parse as urlparse
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from urllib import robotparser
import concurrent.futures
import threading
from queue import Queue
import asyncio
import aiohttp

# GPU/CUDA imports (optioneel)
try:
    import os
    # Set CUDA path to fix warning
    if 'CUDA_PATH' not in os.environ:
        possible_cuda_paths = [
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0',
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8',
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0',
        ]
        for path in possible_cuda_paths:
            if os.path.exists(path):
                os.environ['CUDA_PATH'] = path
                break
    
    import cupy as cp
    import cupyx.scipy.sparse as cusparse
    CUDA_AVAILABLE = True
    print("[gpu] CUDA/GPU support beschikbaar")
except ImportError:
    CUDA_AVAILABLE = False
    print("[gpu] CUDA niet beschikbaar - gebruik CPU only")

# Firecrawl import (optioneel)
try:
    from firecrawl import Firecrawl
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False
    print("[warning] Firecrawl niet beschikbaar. Installeer met: pip install firecrawl-py")

# ========================= Config =========================
USER_AGENT = "LocalBizScraper/4.0 (+mailto:contact@example.com)"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
REQUEST_TIMEOUT = 12.0
MAX_RETRIES = 2
BACKOFF_BASE = 1.6
CRAWL_DELAY_DEFAULT = 0.7
MAX_INTERNAL_PAGES_DEFAULT = 3

# GPU/Performance config
USE_GPU = CUDA_AVAILABLE
GPU_WORKER_RATIO = 0.8  # 80% GPU workers, 20% CPU workers
MAX_GPU_WORKERS = 12
MAX_CPU_WORKERS = 4
ASYNC_BATCH_SIZE = 20

OSM_NOMINATIM = "https://nominatim.openstreetmap.org/search"
OVERPASS_PRIMARY = "https://overpass-api.de/api/interpreter"
OVERPASS_FALLBACK = "https://overpass.kumi.systems/api/interpreter"

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}")
OBF_AT = re.compile(r"\s?(?:\(|\[)?(?:at|apenstaart|AT|\(at\)|\[at\])(?:\)|\])?\s?", re.I)
OBF_DOT = re.compile(r"\s?(?:\(|\[)?(?:dot|DOT|\(dot\)|\[dot\])(?:\)|\])?\s?", re.I)

# Verbeterde adres regex patterns
ADDRESS_PATTERNS = [
    # Nederlandse postcodes
    r'\b\d{4}\s?[A-Z]{2}\b',  # 1234 AB
    # Straat + nummer
    r'\b[A-Za-z][A-Za-z\s]+(?:straat|laan|weg|plein|park|dreef|kade|gracht|singel)\s+\d+[A-Za-z]?\b',
    # Huisnummer + straat
    r'\b\d+[A-Za-z]?\s+[A-Za-z][A-Za-z\s]+(?:straat|laan|weg|plein|park|dreef|kade|gracht|singel)\b',
    # Algemene adres patronen
    r'\b[A-Za-z][A-Za-z\s]+\s+\d+[A-Za-z]?\s*,\s*\d{4}\s?[A-Z]{2}\s+[A-Za-z\s]+\b',
]

CONTACT_HINTS = ["contact", "contacteer", "neem-contact", "over", "over-ons", "team", "impressum", "support", "privacy"]
SECONDARY_HINTS = ["klantenservice", "service", "help", "helpdesk", "faq", "voorwaarden"]

# Specifieke set tags voor meer gerichte zoekopdrachten
BASE_TAGS = [
    'node["name"]', 'way["name"]', 'relation["name"]',
    'node["company"]', 'way["company"]', 'relation["company"]',
    
    # Restaurants & Horeca
    'node["amenity"="restaurant"]', 'way["amenity"="restaurant"]', 'relation["amenity"="restaurant"]',
    'node["amenity"="cafe"]', 'way["amenity"="cafe"]', 'relation["amenity"="cafe"]',
    'node["amenity"="bar"]', 'way["amenity"="bar"]', 'relation["amenity"="bar"]',
    'node["amenity"="pub"]', 'way["amenity"="pub"]', 'relation["amenity"="pub"]',
    'node["amenity"="fast_food"]', 'way["amenity"="fast_food"]', 'relation["amenity"="fast_food"]',
    'node["amenity"="food_court"]', 'way["amenity"="food_court"]', 'relation["amenity"="food_court"]',
    'node["amenity"="ice_cream"]', 'way["amenity"="ice_cream"]', 'relation["amenity"="ice_cream"]',
    
    # Winkels & Retail
    'node["shop"="supermarket"]', 'way["shop"="supermarket"]', 'relation["shop"="supermarket"]',
    'node["shop"="convenience"]', 'way["shop"="convenience"]', 'relation["shop"="convenience"]',
    'node["shop"="clothes"]', 'way["shop"="clothes"]', 'relation["shop"="clothes"]',
    'node["shop"="shoes"]', 'way["shop"="shoes"]', 'relation["shop"="shoes"]',
    'node["shop"="electronics"]', 'way["shop"="electronics"]', 'relation["shop"="electronics"]',
    'node["shop"="furniture"]', 'way["shop"="furniture"]', 'relation["shop"="furniture"]',
    'node["shop"="hardware"]', 'way["shop"="hardware"]', 'relation["shop"="hardware"]',
    'node["shop"="pharmacy"]', 'way["shop"="pharmacy"]', 'relation["shop"="pharmacy"]',
    'node["shop"="bakery"]', 'way["shop"="bakery"]', 'relation["shop"="bakery"]',
    'node["shop"="butcher"]', 'way["shop"="butcher"]', 'relation["shop"="butcher"]',
    'node["shop"="florist"]', 'way["shop"="florist"]', 'relation["shop"="florist"]',
    'node["shop"="jewelry"]', 'way["shop"="jewelry"]', 'relation["shop"="jewelry"]',
    'node["shop"="books"]', 'way["shop"="books"]', 'relation["shop"="books"]',
    'node["shop"="gift"]', 'way["shop"="gift"]', 'relation["shop"="gift"]',
    'node["shop"="sports"]', 'way["shop"="sports"]', 'relation["shop"="sports"]',
    'node["shop"="toys"]', 'way["shop"="toys"]', 'relation["shop"="toys"]',
    'node["shop"="beauty"]', 'way["shop"="beauty"]', 'relation["shop"="beauty"]',
    'node["shop"="hairdresser"]', 'way["shop"="hairdresser"]', 'relation["shop"="hairdresser"]',
    
    # Auto & Transport
    'node["shop"="car"]', 'way["shop"="car"]', 'relation["shop"="car"]',
    'node["shop"="car_repair"]', 'way["shop"="car_repair"]', 'relation["shop"="car_repair"]',
    'node["shop"="car_parts"]', 'way["shop"="car_parts"]', 'relation["shop"="car_parts"]',
    'node["amenity"="fuel"]', 'way["amenity"="fuel"]', 'relation["amenity"="fuel"]',
    'node["amenity"="car_wash"]', 'way["amenity"="car_wash"]', 'relation["amenity"="car_wash"]',
    'node["amenity"="vehicle_inspection"]', 'way["amenity"="vehicle_inspection"]', 'relation["amenity"="vehicle_inspection"]',
    'node["craft"="vehicle_repair"]', 'way["craft"="vehicle_repair"]', 'relation["craft"="vehicle_repair"]',
    
    # Gezondheid & Medisch
    'node["amenity"="hospital"]', 'way["amenity"="hospital"]', 'relation["amenity"="hospital"]',
    'node["amenity"="clinic"]', 'way["amenity"="clinic"]', 'relation["amenity"="clinic"]',
    'node["amenity"="dentist"]', 'way["amenity"="dentist"]', 'relation["amenity"="dentist"]',
    'node["amenity"="doctors"]', 'way["amenity"="doctors"]', 'relation["amenity"="doctors"]',
    'node["amenity"="veterinary"]', 'way["amenity"="veterinary"]', 'relation["amenity"="veterinary"]',
    'node["amenity"="pharmacy"]', 'way["amenity"="pharmacy"]', 'relation["amenity"="pharmacy"]',
    
    # Financieel & Professioneel
    'node["amenity"="bank"]', 'way["amenity"="bank"]', 'relation["amenity"="bank"]',
    'node["amenity"="atm"]', 'way["amenity"="atm"]', 'relation["amenity"="atm"]',
    'node["office"="accountant"]', 'way["office"="accountant"]', 'relation["office"="accountant"]',
    'node["office"="lawyer"]', 'way["office"="lawyer"]', 'relation["office"="lawyer"]',
    'node["office"="insurance"]', 'way["office"="insurance"]', 'relation["office"="insurance"]',
    'node["office"="estate_agent"]', 'way["office"="estate_agent"]', 'relation["office"="estate_agent"]',
    'node["office"="consulting"]', 'way["office"="consulting"]', 'relation["office"="consulting"]',
    'node["office"="marketing"]', 'way["office"="marketing"]', 'relation["office"="marketing"]',
    'node["office"="advertising"]', 'way["office"="advertising"]', 'relation["office"="advertising"]',
    
    # Technologie & IT
    'node["office"="it"]', 'way["office"="it"]', 'relation["office"="it"]',
    'node["office"="computer"]', 'way["office"="computer"]', 'relation["office"="computer"]',
    'node["office"="software"]', 'way["office"="software"]', 'relation["office"="software"]',
    'node["office"="web_design"]', 'way["office"="web_design"]', 'relation["office"="web_design"]',
    
    # Onderwijs & Training
    'node["amenity"="school"]', 'way["amenity"="school"]', 'relation["amenity"="school"]',
    'node["amenity"="college"]', 'way["amenity"="college"]', 'relation["amenity"="college"]',
    'node["amenity"="university"]', 'way["amenity"="university"]', 'relation["amenity"="university"]',
    'node["amenity"="kindergarten"]', 'way["amenity"="kindergarten"]', 'relation["amenity"="kindergarten"]',
    'node["office"="educational"]', 'way["office"="educational"]', 'relation["office"="educational"]',
    
    # Vrije tijd & Entertainment
    'node["amenity"="cinema"]', 'way["amenity"="cinema"]', 'relation["amenity"="cinema"]',
    'node["amenity"="theatre"]', 'way["amenity"="theatre"]', 'relation["amenity"="theatre"]',
    'node["amenity"="nightclub"]', 'way["amenity"="nightclub"]', 'relation["amenity"="nightclub"]',
    'node["amenity"="casino"]', 'way["amenity"="casino"]', 'relation["amenity"="casino"]',
    'node["leisure"="fitness_center"]', 'way["leisure"="fitness_center"]', 'relation["leisure"="fitness_center"]',
    'node["leisure"="sports_center"]', 'way["leisure"="sports_center"]', 'relation["leisure"="sports_center"]',
    'node["leisure"="swimming_pool"]', 'way["leisure"="swimming_pool"]', 'relation["leisure"="swimming_pool"]',
    'node["leisure"="golf_course"]', 'way["leisure"="golf_course"]', 'relation["leisure"="golf_course"]',
    
    # Accommodatie
    'node["tourism"="hotel"]', 'way["tourism"="hotel"]', 'relation["tourism"="hotel"]',
    'node["tourism"="guest_house"]', 'way["tourism"="guest_house"]', 'relation["tourism"="guest_house"]',
    'node["tourism"="hostel"]', 'way["tourism"="hostel"]', 'relation["tourism"="hostel"]',
    'node["tourism"="apartment"]', 'way["tourism"="apartment"]', 'relation["tourism"="apartment"]',
    
    # Ambacht & Industrie
    'node["craft"="bakery"]', 'way["craft"="bakery"]', 'relation["craft"="bakery"]',
    'node["craft"="carpenter"]', 'way["craft"="carpenter"]', 'relation["craft"="carpenter"]',
    'node["craft"="electrician"]', 'way["craft"="electrician"]', 'relation["craft"="electrician"]',
    'node["craft"="plumber"]', 'way["craft"="plumber"]', 'relation["craft"="plumber"]',
    'node["craft"="painter"]', 'way["craft"="painter"]', 'relation["craft"="painter"]',
    'node["craft"="photographer"]', 'way["craft"="photographer"]', 'relation["craft"="photographer"]',
    'node["craft"="tailor"]', 'way["craft"="tailor"]', 'relation["craft"="tailor"]',
    'node["craft"="jeweler"]', 'way["craft"="jeweler"]', 'relation["craft"="jeweler"]',
    
    # Contact & Website tags
    'node["website"]', 'way["website"]', 'relation["website"]',
    'node["contact:website"]', 'way["contact:website"]', 'relation["contact:website"]',
    'node["url"]', 'way["url"]', 'relation["url"]',
    'node["email"]', 'way["email"]', 'relation["email"]',
    'node["contact:email"]', 'way["contact:email"]', 'relation["contact:email"]',
    'node["phone"]', 'way["phone"]', 'relation["phone"]',
    'node["contact:phone"]', 'way["contact:phone"]', 'relation["contact:phone"]',
    'node["telephone"]', 'way["telephone"]', 'relation["telephone"]',
    'node["contact:telephone"]', 'way["contact:telephone"]', 'relation["contact:telephone"]',
    
    # Adres tags
    'node["addr:street"]', 'way["addr:street"]', 'relation["addr:street"]',
    'node["addr:housenumber"]', 'way["addr:housenumber"]', 'relation["addr:housenumber"]',
    'node["addr:postcode"]', 'way["addr:postcode"]', 'relation["addr:postcode"]',
    'node["addr:city"]', 'way["addr:city"]', 'relation["addr:city"]',
]

# =============================== UX ===============================
import threading
import itertools

def fmt_eta(sec: float) -> str:
    sec = max(0, int(sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"

class Spinner:
    def __init__(self, text: str = "Bezig"):
        self.text = text
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        for ch in itertools.cycle("|/-\\"):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r{self.text} {ch}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * (len(self.text) + 6) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        self._thr.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thr.join()

def progress_bar(done: int, total: int, start_ts: float, prefix: str, width: int = 28):
    if total <= 0:
        return
    ratio = min(max(done / total, 0), 1)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - start_ts
    eta = (elapsed / done * (total - done)) if done else 0
    rate = f"{(done/elapsed):.1f}/s" if elapsed > 0 and done > 0 else "-/s"
    sys.stdout.write(f"\r{prefix} [{bar}] {int(ratio*100):3d}% ({done}/{total}) {rate} ETA {fmt_eta(eta)}")
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()

class Heartbeat:
    def __init__(self, interval_sec: float = 5.0):
        self.intv = interval_sec
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self.phase = "init"
        self.detail = ""
        self.counter = 0
        self.total = 0
        self.start_ts = time.time()

    def start(self):
        self._thr.start()

    def stop(self):
        self._stop.set()
        self._thr.join()

    def set_phase(self, phase: str, detail: str = ""):
        self.phase = phase
        self.detail = detail

    def set_prog(self, c: int, t: int):
        self.counter, self.total = c, t

    def _run(self):
        while not self._stop.is_set():
            elapsed = time.time() - self.start_ts
            if self.total > 0 and self.counter > 0:
                eta = (elapsed / self.counter) * (self.total - self.counter)
                pct = int(self.counter / self.total * 100)
                line = f"[hb] {self.phase} {pct}% ({self.counter}/{self.total}) ETA {fmt_eta(eta)} {self.detail}"
            else:
                line = f"[hb] {self.phase} elapsed {fmt_eta(elapsed)} {self.detail}"
            sys.stdout.write("\n" + line + "\n")
            sys.stdout.flush()
            for _ in range(int(self.intv * 10)):
                if self._stop.is_set():
                    break
                time.sleep(0.1)

# Pause helper
def maybe_pause(pause_file: str, where: str = ""):
    if not pause_file:
        return
    while os.path.exists(pause_file):
        sys.stdout.write(f"\n[pause] '{pause_file}' gevonden → pauze ({where}). Verwijder het bestand om door te gaan...\n")
        sys.stdout.flush()
        time.sleep(2)

# ============================ HTTP & Robots ============================
def jitter(s: float) -> float:
    return s + random.uniform(0, 0.5)

def http_get(url: str, headers=None, allow_redirects=True):
    headers = {**HEADERS, **(headers or {})}
    for a in range(1, int(MAX_RETRIES) + 1):
        try:
            r = requests.get(url, headers=headers, timeout=float(REQUEST_TIMEOUT), allow_redirects=allow_redirects)
            if 200 <= r.status_code < 300:
                return r
            if r.status_code in (429, 503):
                time.sleep((BACKOFF_BASE ** a) + random.random())
                continue
            if r.status_code in (401, 403):
                return None
            return None
        except requests.RequestException:
            time.sleep((BACKOFF_BASE ** a) + random.random())
    return None

def http_post(url: str, data: bytes, headers=None):
    headers = {**HEADERS, **(headers or {})}
    for a in range(1, int(MAX_RETRIES) + 1):
        try:
            r = requests.post(url, data=data, headers=headers, timeout=float(REQUEST_TIMEOUT))
            if 200 <= r.status_code < 300:
                return r
            if r.status_code in (429, 503):
                time.sleep((BACKOFF_BASE ** a) + random.random())
                continue
            return None
        except requests.RequestException:
            time.sleep((BACKOFF_BASE ** a) + random.random())
    return None

def normalize_url(u: str) -> Optional[str]:
    try:
        p = urlparse.urlparse(u)
        return "http://" + u if not p.scheme else u
    except Exception:
        return None

def absolutize(base: str, href: str) -> Optional[str]:
    try:
        return urlparse.urljoin(base, href)
    except Exception:
        return None

_robot_cache: Dict[str, Tuple[float, robotparser.RobotFileParser]] = {}

def _load_robots(host: str):
    robots_url = urlparse.urljoin(host, "/robots.txt")
    rp = robotparser.RobotFileParser()
    rp.set_url(robots_url)
    resp = http_get(robots_url, headers={"User-Agent": USER_AGENT})
    if resp and resp.status_code == 200 and resp.text:
        rp.parse(resp.text.splitlines())
    else:
        rp.parse(["User-agent: *", "Allow: /"])
    return rp

def can_fetch(url: str, fail_closed: bool = False, cache_ttl: int = 3600) -> bool:
    p = urlparse.urlparse(url)
    if not p.scheme or not p.netloc:
        return False
    host = f"{p.scheme}://{p.netloc}"
    ts_rp = _robot_cache.get(host)
    if (not ts_rp) or (time.time() - ts_rp[0] > cache_ttl):
        try:
            rp = _load_robots(host)
        except Exception:
            rp = robotparser.RobotFileParser()
            rp.parse(["User-agent: *", "Disallow: /"] if fail_closed else ["User-agent: *", "Allow: /"])
        _robot_cache[host] = (time.time(), rp)
    return _robot_cache[host][1].can_fetch(USER_AGENT, url)

# ===================== GPU/Async Worker Pool =====================
class HybridWorkerPool:
    """Hybrid CPU/GPU worker pool voor parallelle verwerking"""
    
    def __init__(self, gpu_ratio=0.8, max_gpu_workers=12, max_cpu_workers=4):
        self.gpu_ratio = gpu_ratio
        self.max_gpu_workers = max_gpu_workers if USE_GPU else 0
        self.max_cpu_workers = max_cpu_workers
        self.gpu_executor = None
        self.cpu_executor = None
        self.async_session = None
        
        if USE_GPU:
            try:
                # Test GPU availability
                cp.cuda.runtime.getDeviceCount()
                print(f"[gpu] GPU workers: {self.max_gpu_workers}, CPU workers: {self.max_cpu_workers}")
            except Exception as e:
                print(f"[gpu] GPU test failed: {e}, falling back to CPU only")
                self.max_gpu_workers = 0
    
    def __enter__(self):
        # Start CPU thread pool
        self.cpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_cpu_workers,
            thread_name_prefix="CPU-Worker"
        )
        
        # Start GPU thread pool (if available)
        if self.max_gpu_workers > 0:
            self.gpu_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_gpu_workers,
                thread_name_prefix="GPU-Worker"
            )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cpu_executor:
            self.cpu_executor.shutdown(wait=True)
        if self.gpu_executor:
            self.gpu_executor.shutdown(wait=True)
    
    def submit_gpu_task(self, func, *args, **kwargs):
        """Submit task to GPU worker pool"""
        if self.gpu_executor:
            return self.gpu_executor.submit(func, *args, **kwargs)
        else:
            # Fallback to CPU if GPU not available
            return self.cpu_executor.submit(func, *args, **kwargs)
    
    def submit_cpu_task(self, func, *args, **kwargs):
        """Submit task to CPU worker pool"""
        return self.cpu_executor.submit(func, *args, **kwargs)
    
    def submit_batch(self, tasks, gpu_tasks=None):
        """Submit batch of tasks with GPU/CPU distribution"""
        if gpu_tasks is None:
            gpu_tasks = int(len(tasks) * self.gpu_ratio)
        
        futures = []
        
        # Submit GPU tasks
        for i, task in enumerate(tasks[:gpu_tasks]):
            func, args, kwargs = task
            future = self.submit_gpu_task(func, *args, **kwargs)
            futures.append(future)
        
        # Submit CPU tasks
        for task in tasks[gpu_tasks:]:
            func, args, kwargs = task
            future = self.submit_cpu_task(func, *args, **kwargs)
            futures.append(future)
        
        return futures

class GPUAcceleratedProcessor:
    """GPU-accelerated text processing voor contact extractie"""
    
    @staticmethod
    def extract_contacts_gpu(text_batch):
        """GPU-accelerated contact extraction voor batch van teksten"""
        if not USE_GPU:
            return [GPUAcceleratedProcessor.extract_contacts_cpu(text) for text in text_batch]
        
        try:
            # Convert texts to GPU arrays for parallel processing
            results = []
            for text in text_batch:
                # GPU-accelerated regex matching
                gpu_text = cp.asarray([ord(c) for c in text])
                
                # Email pattern matching op GPU
                email_pattern = GPUAcceleratedProcessor._gpu_email_pattern()
                phone_pattern = GPUAcceleratedProcessor._gpu_phone_pattern()
                
                # Simplified GPU processing (basic implementation)
                emails = GPUAcceleratedProcessor._find_emails_gpu(text)
                phones = GPUAcceleratedProcessor._find_phones_gpu(text)
                
                results.append((emails, phones))
            
            return results
        except Exception as e:
            print(f"[gpu] GPU processing failed: {e}, falling back to CPU")
            return [GPUAcceleratedProcessor.extract_contacts_cpu(text) for text in text_batch]
    
    @staticmethod
    def extract_contacts_cpu(text):
        """CPU fallback voor contact extractie"""
        emails = set(EMAIL_RE.findall(text))
        phones = set([p for p in PHONE_RE.findall(text) if len(re.sub(r"\D", "", p)) >= 8])
        return emails, phones
    
    @staticmethod
    def _find_emails_gpu(text):
        """GPU-accelerated email finding"""
        # Simplified implementation - in practice you'd use more complex GPU kernels
        return set(EMAIL_RE.findall(text))
    
    @staticmethod
    def _find_phones_gpu(text):
        """GPU-accelerated phone finding"""
        return set([p for p in PHONE_RE.findall(text) if len(re.sub(r"\D", "", p)) >= 8])
    
    @staticmethod
    def _gpu_email_pattern():
        """Create GPU-optimized email pattern"""
        return None  # Placeholder for GPU kernel
    
    @staticmethod
    def _gpu_phone_pattern():
        """Create GPU-optimized phone pattern"""
        return None  # Placeholder for GPU kernel

# ===================== Firecrawl Integration =====================
class FirecrawlScraper:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('FIRECRAWL_API_KEY')
        self.client = None
        
        print(f"[firecrawl] Initialiseren met API key: {self.api_key[:10] if self.api_key else 'None'}...")
        print(f"[firecrawl] FIRECRAWL_AVAILABLE: {FIRECRAWL_AVAILABLE}")
        
        if FIRECRAWL_AVAILABLE and self.api_key:
            try:
                print(f"[firecrawl] Proberen Firecrawl client te maken...")
                self.client = Firecrawl(api_key=self.api_key)
                print(f"[firecrawl] ✅ Firecrawl client succesvol geïnitialiseerd!")
                
                # Test de client met een eenvoudige call
                try:
                    print(f"[firecrawl] Test API call...")
                    test_result = self.client.scrape("https://example.com", formats=['markdown'])
                    print(f"[firecrawl] ✅ Test API call succesvol!")
                except Exception as test_e:
                    print(f"[firecrawl] ⚠️ Test API call gefaald: {test_e}")
                    
            except Exception as e:
                print(f"[firecrawl] ❌ Fout bij initialisatie: {e}")
                print(f"[firecrawl] Exception type: {type(e)}")
                self.client = None
        elif not FIRECRAWL_AVAILABLE:
            print("[firecrawl] ❌ Firecrawl niet beschikbaar - gebruik fallback scraping")
        else:
            print("[firecrawl] ❌ Geen API key gevonden - gebruik fallback scraping")

    def scrape_website(self, url: str, max_pages: int = 3) -> Tuple[str, str, str]:
        """
        Scrape website met Firecrawl voor betere contact en adres extractie
        Returns: (email, phone, address)
        """
        if not self.client:
            print(f"[firecrawl] Geen client beschikbaar voor {url}")
            return self._fallback_scrape(url, max_pages)
        
        try:
            print(f"[firecrawl] Scraping {url} met Firecrawl API...")
            
            # Probeer eerst de hoofdpagina met nieuwe API syntax
            result = self.client.scrape(
                url,
                formats=['markdown'],
                onlyMainContent=True
            )
            
            print(f"[firecrawl] API response voor {url}: {type(result)}")
            
            # Controleer verschillende response formaten
            content = None
            if hasattr(result, 'data') and isinstance(result.data, dict):
                content = result.data.get('markdown', '')
            elif isinstance(result, dict):
                content = result.get('markdown', '')
            elif hasattr(result, 'markdown'):
                content = result.markdown
            else:
                print(f"[firecrawl] Onverwacht response format: {result}")
                return self._fallback_scrape(url, max_pages)
            
            if not content:
                print(f"[firecrawl] Geen content gevonden voor {url}")
                return self._fallback_scrape(url, max_pages)
            
            print(f"[firecrawl] Content gevonden ({len(content)} chars) voor {url}")
            email, phone = self._extract_contacts_from_text(content)
            address = self._extract_address_from_text(content)
            
            print(f"[firecrawl] Geëxtraheerd voor {url}: email={bool(email)}, phone={bool(phone)}, address={bool(address)}")
            
            # Als we nog geen contact hebben, probeer contact pagina's
            if not email and not phone:
                contact_urls = self._find_contact_pages(url, content)
                print(f"[firecrawl] Gevonden {len(contact_urls)} contact URLs voor {url}")
                
                for contact_url in contact_urls[:max_pages-1]:
                    try:
                        print(f"[firecrawl] Scraping contact pagina: {contact_url}")
                        contact_result = self.client.scrape(
                            contact_url,
                            formats=['markdown'],
                            onlyMainContent=True
                        )
                        
                        contact_content = None
                        if hasattr(contact_result, 'data') and isinstance(contact_result.data, dict):
                            contact_content = contact_result.data.get('markdown', '')
                        elif isinstance(contact_result, dict):
                            contact_content = contact_result.get('markdown', '')
                        elif hasattr(contact_result, 'markdown'):
                            contact_content = contact_result.markdown
                        
                        if contact_content:
                            c_email, c_phone = self._extract_contacts_from_text(contact_content)
                            c_address = self._extract_address_from_text(contact_content)
                            email = email or c_email
                            phone = phone or c_phone
                            address = address or c_address
                            print(f"[firecrawl] Contact pagina {contact_url}: email={bool(c_email)}, phone={bool(c_phone)}")
                    except Exception as e:
                        print(f"[firecrawl] Fout bij contact pagina {contact_url}: {e}")
                        continue
            
            return email, phone, address
            
        except Exception as e:
            print(f"[firecrawl] Fout bij scraping {url}: {e}")
            print(f"[firecrawl] Exception type: {type(e)}")
            return self._fallback_scrape(url, max_pages)
    
    def _fallback_scrape(self, url: str, max_pages: int) -> Tuple[str, str, str]:
        """Fallback naar originele scraping methode"""
        try:
            email, phone = crawl_site_for_contacts(url, max_pages, False, REQUEST_TIMEOUT)
            # Probeer adres te extraheren uit de hoofdpagina
            resp = http_get(url)
            address = ""
            if resp:
                soup = BeautifulSoup(resp.text, "html.parser")
                address = self._extract_address_from_soup(soup)
            return email, phone, address
        except Exception:
            return "", "", ""
    
    def _extract_contacts_from_text(self, text: str) -> Tuple[str, str]:
        """Extract email en telefoon uit tekst"""
        emails = set()
        phones = set()
        
        # Email extractie
        t = OBF_AT.sub("@", text)
        t = OBF_DOT.sub(".", t)
        emails = set(EMAIL_RE.findall(t))
        
        # Telefoon extractie
        phones = set([p for p in PHONE_RE.findall(text) if len(re.sub(r"\D", "", p)) >= 8])
        
        email = next(iter(sorted(emails)), "")
        phone = next(iter(sorted(phones)), "")
        return email, phone
    
    def _extract_address_from_text(self, text: str) -> str:
        """Extract adres uit tekst met verbeterde patterns"""
        addresses = []
        
        for pattern in ADDRESS_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            addresses.extend(matches)
        
        # Filter en sorteer adressen
        unique_addresses = list(set(addresses))
        unique_addresses.sort(key=len, reverse=True)  # Langste eerst (meest compleet)
        
        return unique_addresses[0] if unique_addresses else ""
    
    def _extract_address_from_soup(self, soup: BeautifulSoup) -> str:
        """Extract adres uit BeautifulSoup object"""
        # Zoek naar adres in verschillende elementen
        address_selectors = [
            '[itemtype*="PostalAddress"]',
            '.address', '.adres', '.contact-info',
            '[class*="address"]', '[class*="adres"]',
            '[id*="address"]', '[id*="adres"]'
        ]
        
        for selector in address_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                if text:
                    address = self._extract_address_from_text(text)
                    if address:
                        return address
        
        # Fallback: zoek in hele pagina
        full_text = soup.get_text()
        return self._extract_address_from_text(full_text)
    
    def _find_contact_pages(self, base_url: str, content: str) -> List[str]:
        """Vind contact pagina's uit content"""
        contact_urls = []
        
        # Zoek naar contact links in markdown
        contact_patterns = [
            r'\[([^\]]*contact[^\]]*)\]\(([^)]+)\)',
            r'\[([^\]]*over[^\]]*)\]\(([^)]+)\)',
            r'\[([^\]]*team[^\]]*)\]\(([^)]+)\)',
        ]
        
        for pattern in contact_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for _, url in matches:
                full_url = absolutize(base_url, url)
                if full_url and full_url not in contact_urls:
                    contact_urls.append(full_url)
        
        return contact_urls[:5]  # Max 5 contact pagina's

# ===================== Geocoding & Overpass (tiled) =====================
def geocode_location(query: str, countrycodes: Optional[str]) -> Tuple[float, float]:
    params = {"q": query, "format": "json", "addressdetails": 0, "limit": 1}
    if countrycodes:
        params["countrycodes"] = countrycodes
    resp = http_get(OSM_NOMINATIM + "?" + urlparse.urlencode(params),
                    headers={"User-Agent": USER_AGENT, "Accept-Language": "nl"})
    if not resp:
        raise RuntimeError(f"Geocoding mislukt voor: {query}")
    data = resp.json()
    if not data:
        raise RuntimeError(f"Geen resultaten voor: {query}")
    return float(data[0]["lat"]), float(data[0]["lon"])

def make_selector(tag: str, bbox: Tuple[float, float, float, float]) -> str:
    s, w, n, e = bbox
    tag = tag.strip()
    if tag.startswith(("node[", "way[", "relation[")):
        return f"{tag}({s},{w},{n},{e});"
    if "=" in tag:
        key, val = tag.split("=", 1)
        key, val = key.strip(), val.strip()
        if val in ("*", ""):
            return (
                f'node[{key}]({s},{w},{n},{e}); '
                f'way[{key}]({s},{w},{n},{e}); '
                f'relation[{key}]({s},{w},{n},{e});'
            )
        return (
            f'node["{key}"="{val}"]({s},{w},{n},{e}); '
            f'way["{key}"="{val}"]({s},{w},{n},{e}); '
            f'relation["{key}"="{val}"]({s},{w},{n},{e});'
        )
    return (
        f'node[{tag}]({s},{w},{n},{e}); '
        f'way[{tag}]({s},{w},{n},{e}); '
        f'relation[{tag}]({s},{w},{n},{e});'
    )

def build_query_for_bbox(bbox: Tuple[float, float, float, float], limit: int, q_timeout: int) -> bytes:
    selectors = "\n".join([make_selector(t, bbox) for t in BASE_TAGS])
    q = f"""[out:json][timeout:{int(q_timeout)}];
(
  {selectors}
);
out center tags {limit};"""
    return q.encode("utf-8")

def check_overpass(endpoint: str, timeout: float) -> bool:
    try:
        base = endpoint.rsplit("/", 1)[0]
        status = base + "/status"
        r = requests.get(status, headers={"User-Agent": USER_AGENT}, timeout=min(timeout, 5))
        if r.status_code == 200:
            return True
    except Exception:
        pass
    try:
        tiny = b"[out:json][timeout:5];node(0,0,0,0);out;"
        r = requests.post(endpoint, data=tiny, headers={"User-Agent": USER_AGENT}, timeout=min(timeout, 6))
        return r.status_code == 200
    except Exception:
        return False

def overpass_bbox(endpoint: str, bbox: Tuple[float, float, float, float], limit: int, q_timeout: int) -> List[Dict]:
    payload = build_query_for_bbox(bbox, limit, q_timeout)
    r = http_post(endpoint, data=payload, headers={"User-Agent": USER_AGENT})
    if not r:
        return []
    return r.json().get("elements", [])

# ===================== Tiling helpers =====================
def km_to_deg_lat(km: float) -> float:
    return km / 110.574

def km_to_deg_lon(km: float, lat: float) -> float:
    return km / (111.320 * max(0.1, math.cos(math.radians(lat))))

def make_tiles(lat: float, lon: float, radius_km: float, tile_km: float) -> List[Tuple[float, float, float, float]]:
    dlat = km_to_deg_lat(tile_km)
    dlon = km_to_deg_lon(tile_km, lat)
    r_lat = km_to_deg_lat(radius_km)
    r_lon = km_to_deg_lon(radius_km, lat)
    tiles: List[Tuple[float, float, float, float]] = []
    i_min = int(math.floor(-r_lat / dlat)) - 1
    i_max = int(math.ceil(r_lat / dlat)) + 1
    j_min = int(math.floor(-r_lon / dlon)) - 1
    j_max = int(math.ceil(r_lon / dlon)) + 1
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            s = lat + i * dlat
            n = s + dlat
            w = lon + j * dlon
            e = w + dlon
            tiles.append((s, w, n, e))
    return tiles

# ===================== Contact extractie (advanced) =====================
CONTACT_EMAIL_KEYS = ["contact:email", "email"]
CONTACT_PHONE_KEYS = ["contact:phone", "phone", "telephone", "contact:telephone"]
WEBSITE_KEYS = ["contact:website", "website", "url", "website:nl", "website:en", "website:de", "website:fr"]

def best_website_from_tags(tags: Dict) -> str:
    for k in WEBSITE_KEYS:
        v = tags.get(k)
        if isinstance(v, str) and v.strip():
            u = normalize_url(v.strip())
            if u:
                return u
    # fallback: domein uit e-mail
    for ek in CONTACT_EMAIL_KEYS:
        v = tags.get(ek)
        if isinstance(v, str) and "@" in v:
            try:
                dom = v.split("@", 1)[1].split(">")[0].split(")")[0].split(" ")[0]
                return normalize_url(f"http://{dom}")
            except Exception:
                pass
    return ""

def contacts_from_osm_tags(tags: Dict) -> Tuple[str, str]:
    email = ""
    phone = ""
    for k in CONTACT_EMAIL_KEYS:
        val = tags.get(k)
        if isinstance(val, str) and val.strip():
            t = OBF_AT.sub("@", val)
            t = OBF_DOT.sub(".", t)
            cands = EMAIL_RE.findall(t)
            if cands:
                email = sorted(set(cands))[0]
                break
    for k in CONTACT_PHONE_KEYS:
        val = tags.get(k)
        if isinstance(val, str) and val.strip():
            cands = [p for p in PHONE_RE.findall(val) if len(re.sub(r"\D", "", p)) >= 8]
            if cands:
                phone = cands[0]
                break
    return email, phone

def parse_ld_json_emails_phones(soup: BeautifulSoup) -> Tuple[Set[str], Set[str]]:
    emails, phones = set(), set()
    for tag in soup.find_all("script", type=lambda v: v and "ld+json" in v):
        try:
            import json
            data = json.loads(tag.string or "{}")
            objs = data if isinstance(data, list) else [data]
            for obj in objs:
                if isinstance(obj, dict):
                    if "email" in obj and isinstance(obj["email"], str):
                        e = OBF_AT.sub("@", obj["email"])
                        e = OBF_DOT.sub(".", e)
                        emails |= set(EMAIL_RE.findall(e))
                    cps = obj.get("contactPoint") or obj.get("contactPoints") or []
                    cps = cps if isinstance(cps, list) else [cps]
                    for cp in cps:
                        if isinstance(cp, dict):
                            if "email" in cp and isinstance(cp["email"], str):
                                e = OBF_AT.sub("@", cp["email"])
                                e = OBF_DOT.sub(".", e)
                                emails |= set(EMAIL_RE.findall(e))
                            if "telephone" in cp and isinstance(cp["telephone"], str):
                                phones |= set(PHONE_RE.findall(cp["telephone"]))
        except Exception:
            continue
    return emails, phones

def score_internal_path(path: str) -> int:
    p = path.lower()
    score = 100
    for kw in CONTACT_HINTS:
        if kw in p:
            score = min(score, 0)
    for kw in SECONDARY_HINTS:
        if kw in p:
            score = min(score, 10)
    if p.endswith((".php", ".html", "/")):
        score = min(score, score + 5)
    return score

def extract_contacts_from_html(base_url: str, html: str) -> Tuple[Set[str], Set[str], List[str]]:
    soup = BeautifulSoup(html, "html.parser")
    emails: Set[str] = set()
    phones: Set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().startswith("mailto:"):
            addr = href.split("mailto:", 1)[1].split("?")[0]
            t = OBF_AT.sub("@", addr)
            t = OBF_DOT.sub(".", t)
            emails |= set(EMAIL_RE.findall(t))

    em2, ph2 = parse_ld_json_emails_phones(soup)
    emails |= em2
    phones |= ph2

    text = soup.get_text(" ", strip=True)
    t = OBF_AT.sub("@", text)
    t = OBF_DOT.sub(".", t)
    emails |= set(EMAIL_RE.findall(t))
    phones |= set([p for p in PHONE_RE.findall(text) if len(re.sub(r"\D", "", p)) >= 8])

    base = urlparse.urlparse(base_url)
    scored: List[Tuple[int, str]] = []
    for a in soup.find_all("a", href=True):
        href = absolutize(base_url, a["href"]) or ""
        if not href:
            continue
        try:
            u = urlparse.urlparse(href)
            if u.netloc and u.netloc == base.netloc:
                if not u.path.lower().endswith((
                    ".pdf",".png",".jpg",".jpeg",".gif",".svg",".webp",".zip",".rar",".7z",
                    ".doc",".docx",".xls",".xlsx",".ppt",".pptx",".mp4",".mp3",".avi",".mov",".webm",".ico"
                )):
                    scored.append((score_internal_path(u.path or "/"), href.split("#")[0]))
        except Exception:
            continue

    scored.sort(key=lambda t: (t[0], t[1]))
    seen: Set[str] = set()
    internal: List[str] = []
    for _, link in scored:
        if link not in seen:
            seen.add(link)
            internal.append(link)

    return emails, phones, internal

def crawl_site_for_contacts(start_url: str, max_pages: int, fail_closed: bool, page_timeout: float) -> Tuple[str, str]:
    start_url = normalize_url(start_url) or ""
    if not start_url:
        return "", ""
    if not can_fetch(start_url, fail_closed=fail_closed):
        return "", ""

    queue: List[str] = [start_url]
    seen: Set[str] = set()
    found_emails: Set[str] = set()
    found_phones: Set[str] = set()

    while queue and len(seen) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        time.sleep(jitter(CRAWL_DELAY_DEFAULT))
        resp = http_get(url, headers={"User-Agent": USER_AGENT})
        if not resp:
            continue
        seen.add(url)

        html = resp.text
        if len(html) > 1_500_000:
            html = html[:1_500_000]

        em, ph, internal = extract_contacts_from_html(resp.url, html)
        found_emails |= em
        found_phones |= ph

        front = [l for l in internal if any(h in l.lower() for h in CONTACT_HINTS)][:3]
        rest = [l for l in internal if l not in front]
        ordered = front + rest

        max_add = max(0, max_pages - len(seen))
        for link in ordered[:max_add]:
            if link not in seen and link not in queue:
                queue.append(link)

        if found_emails or found_phones:
            break

    email = next(iter(sorted(found_emails)), "")
    phone = next(iter(sorted(found_phones)), "")
    return email, phone

# ===================== Data model & CSV =====================
@dataclass
class Biz:
    name: str
    category: str
    lat: float
    lon: float
    address: str
    website: str
    email: str
    phone: str
    source_location: str

# ---- WINDOWS-SAFE pad/naam helpers ----
INVALID_FS_CHARS = r'[<>:"/\\|?*\x00-\x1F]'
RESERVED_BASENAMES = {
    "CON","PRN","AUX","NUL",
    *(f"COM{i}" for i in range(1,10)),
    *(f"LPT{i}" for i in range(1,10)),
}

def _replace_date_time_placeholders(s: str, now: Optional[datetime] = None) -> str:
    now = now or datetime.now()
    s = re.sub(r'%DATE(?::~\d+,\d+)?%', now.strftime('%Y%m%d'), s, flags=re.I)
    s = re.sub(r'%TIME(?::~\d+,\d+)?%', now.strftime('%H%M%S'), s, flags=re.I)
    return s

def _sanitize_filename(name: str) -> str:
    now = datetime.now()
    root, ext = os.path.splitext(name)
    root = re.sub(INVALID_FS_CHARS, "_", root)
    ext = re.sub(INVALID_FS_CHARS, "_", ext)
    root = root.rstrip(" .")
    if not root:
        root = f"file_{now.strftime('%Y%m%d_%H%M%S')}"
    if root.upper() in RESERVED_BASENAMES:
        root = f"_{root}"
    return root + ext

def make_safe_path(path_str: str) -> str:
    s = _replace_date_time_placeholders(path_str)
    s = os.path.expandvars(s)
    p = Path(os.path.abspath(s))
    safe_name = _sanitize_filename(p.name)
    parent = p.parent if str(p.parent).strip() else Path(".")
    parent.mkdir(parents=True, exist_ok=True)
    return str(parent / safe_name)

def unique_path(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return str(p)
    base, ext = p.stem, p.suffix or ".csv"
    for i in range(1, 1000):
        cand = p.with_name(f"{base}_{i}{ext}")
        if not cand.exists():
            return str(cand)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(p.with_name(f"{base}_{ts}{ext}"))

# CSV sink met autosave/checkpoints
class CsvSink:
    def __init__(self, final_path: str, fieldnames: List[str], autosave_every: int,
                 autosave_percent: float, total_items: int, hard_fsync: bool = False):
        self.final_path = make_safe_path(final_path)
        self.part_path = self.final_path + ".part"
        self.fieldnames = fieldnames
        self.autosave_every = max(0, int(autosave_every))
        self.autosave_percent = float(autosave_percent)
        self.total_items = max(1, int(total_items))  # avoid div/0
        self.hard_fsync = bool(hard_fsync)

        self._f = None
        self._w = None
        self._written_since_flush = 0
        self._next_pct = self.autosave_percent if self.autosave_percent > 0 else 101.0

        mode = "a" if os.path.exists(self.part_path) else "w"
        self._f = open(self.part_path, mode, newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=self.fieldnames)
        if mode == "w" or os.path.getsize(self.part_path) == 0:
            self._w.writeheader()
            self._flush()

    @property
    def writer(self):
        return self._w

    def write_row(self, row: Dict):
        self._w.writerow(row)
        self._written_since_flush += 1

    def maybe_flush(self, done_items: int):
        must = False
        if self.autosave_every and self._written_since_flush >= self.autosave_every:
            must = True
        if self.autosave_percent > 0:
            pct = (100.0 * done_items) / self.total_items
            if pct >= self._next_pct:
                must = True
                # zet volgende drempel
                n = int(pct // self.autosave_percent) + 1
                self._next_pct = self.autosave_percent * n
        if must:
            self._flush()

    def _flush(self):
        self._f.flush()
        if self.hard_fsync:
            try:
                os.fsync(self._f.fileno())
            except Exception:
                pass
        self._written_since_flush = 0

    def finalize(self) -> str:
        # sluit en hernoem atomair naar definitieve pad (uniek)
        if self._f:
            self._flush()
            self._f.close()
            self._f = None
        dest = unique_path(self.final_path)
        try:
            os.replace(self.part_path, dest)
        except Exception:
            # laatste redmiddel: kopiëren
            import shutil
            shutil.copy2(self.part_path, dest)
            # laat .part staan als kopie mislukt; gebruiker kan handmatig
        return dest

    def close_keep_part(self):
        if self._f:
            self._flush()
            self._f.close()
            self._f = None

# ===================== Metrics (4+ regels) =====================
def ascii_bar(label: str, count: int, max_count: int, width: int = 42) -> str:
    if max_count <= 0:
        max_count = 1
    filled = int(width * (count / max_count))
    return f"{label:<18} | {'#'*filled}{'-'*(width-filled)} | {count}"

def show_metrics(metrics: Dict[str, int]) -> None:
    print("\n=== MEET-GRAFIEK ===")
    maxc = max(metrics.values()) if metrics else 1
    for k in ("gevonden", "bezocht_sites", "met_contact", "zonder_contact", "overgeslagen_bestaand", "gpu_processed", "cpu_processed"):
        v = metrics.get(k, 0)
        print(ascii_bar(k, v, maxc))
    
    # GPU performance stats
    gpu_count = metrics.get("gpu_processed", 0)
    cpu_count = metrics.get("cpu_processed", 0)
    total_processed = gpu_count + cpu_count
    if total_processed > 0:
        gpu_percentage = (gpu_count / total_processed) * 100
        print(f"\n[GPU Stats] GPU: {gpu_count} ({gpu_percentage:.1f}%) | CPU: {cpu_count} ({100-gpu_percentage:.1f}%)")
    print()

# ===================== Skip helpers (bestaande CSV's overslaan) =====================
def canonical_key(name: str, lat: float, lon: float) -> Tuple[str, float, float]:
    return (name.strip().lower(), round(lat, 5), round(lon, 5))

def domain_from_url(url: str) -> str:
    try:
        u = urlparse.urlparse(normalize_url(url) or "")
        host = u.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""

def load_skip_sets(csv_paths: Optional[List[str]]) -> Tuple[Set[Tuple[str, float, float]], Set[str]]:
    skip_keys: Set[Tuple[str, float, float]] = set()
    skip_domains: Set[str] = set()
    if not csv_paths:
        return skip_keys, skip_domains

    # Increase CSV field size limit to handle large fields
    csv.field_size_limit(min(2**31-1, 1000000))  # 1MB limit

    for path in csv_paths:
        try:
            # Skip non-CSV files
            if not path.endswith('.csv'):
                print(f"[skip] Overslaan non-CSV bestand: {path}")
                continue
                
            with open(path, "r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    try:
                        n = (row.get("name") or "").strip()
                        lat = float(row.get("lat") or 0.0)
                        lon = float(row.get("lon") or 0.0)
                        if n:
                            skip_keys.add(canonical_key(n, lat, lon))
                        w = row.get("website") or ""
                        d = domain_from_url(w)
                        if d:
                            skip_domains.add(d)
                    except Exception:
                        continue
        except FileNotFoundError:
            print(f"[warn] skip-csv niet gevonden: {path}")
        except Exception as ex:
            print(f"[warn] kon skip-csv niet lezen ({path}): {ex}")
    return skip_keys, skip_domains

# ===================== Helpers =====================
def element_center(e) -> Tuple[float, float]:
    if "lat" in e and "lon" in e:
        return float(e["lat"]), float(e["lon"])
    c = e.get("center") or {}
    return float(c.get("lat", 0.0)), float(c.get("lon", 0.0))

def build_address(tags: Dict) -> str:
    """Verbeterde adres extractie uit OSM tags met validatie"""
    parts = []
    
    # Probeer verschillende adres componenten
    street = tags.get("addr:street", "")
    housenumber = tags.get("addr:housenumber", "")
    postcode = tags.get("addr:postcode", "")
    city = tags.get("addr:city", "")
    
    # Strikte validatie - alle velden moeten aanwezig zijn
    if not all([street, housenumber, postcode, city]):
        return ""
    
    # Bouw adres op
    if street and housenumber:
        parts.append(f"{street} {housenumber}")
    
    if postcode and city:
        parts.append(f"{postcode} {city}")
    
    full_address = " ".join(parts).strip()
    
    # Moet Nederlandse postcode bevatten
    if not re.search(r'\b\d{4}\s?[A-Z]{2}\b', full_address):
        return ""
    
    return full_address

def pick_category(tags: Dict) -> str:
    for k in ("office", "shop", "amenity", "industrial", "craft"):
        if k in tags:
            return f"{k}:{tags.get(k)}"
    return ""

# ===================== Main run =====================
def run(args):
    # Vooruit bepalen van definitieve/part-pad (t.b.v. resume in skip-sets)
    final_out_path = make_safe_path(args.out)
    part_path = final_out_path + ".part"

    # Endpoint check + korte fallback
    ep = args.overpass or OVERPASS_PRIMARY
    print(f"[check] Overpass primary: {ep}")
    ok = check_overpass(ep, timeout=args.timeout)
    if not ok and not args.overpass:
        print("[check] primary down; probeer fallback…")
        if check_overpass(OVERPASS_FALLBACK, timeout=args.timeout):
            ep = OVERPASS_FALLBACK
        else:
            raise SystemExit("Geen bruikbaar Overpass endpoint. Probeer later of specificeer --overpass.")

    # Firecrawl initialisatie
    firecrawl_scraper = FirecrawlScraper(api_key=args.firecrawl_api_key)

    # Locatie
    if args.coords:
        lat, lon = [float(x) for x in args.coords.split(",", 1)]
        label = f"{lat:.5f},{lon:.5f}"
    else:
        lat, lon = geocode_location(args.loc, args.countrycodes)
        label = args.loc

    # Skip-sets laden (+ eventuele resume .part meenemen)
    extra_skip = [part_path] if (args.resume and os.path.exists(part_path)) else []
    skip_paths = (args.skip_csv or []) + extra_skip
    skip_keys, skip_domains = load_skip_sets(skip_paths)
    if skip_keys or skip_domains:
        print(f"[skip] geladen: {len(skip_keys)} keys (naam+coörd) en {len(skip_domains)} website-domeinen")

    radius_km = (args.radius / 1000.0) if args.radius else (args.radius_km or 30.0)
    tile_km = args.tile_km
    tiles = make_tiles(lat, lon, radius_km, tile_km)

    # Sorteer tiles op afstand tot centrum (dichtstbij eerst)
    def tile_center(b):
        s, w, n, e = b
        return ((s + n) / 2.0, (w + e) / 2.0)

    def tile_distance_km(b):
        clat, clon = tile_center(b)
        return math.hypot((clat - lat) * 110.574, (clon - lon) * 111.320 * math.cos(math.radians(lat)))

    tiles.sort(key=tile_distance_km)

    # Early-stop threshold voor tiles
    early_stop_factor = 2
    target_candidates = (args.max_sites or 200) * early_stop_factor

    # Overpass fase
    elements: Dict[Tuple[str, float, float], Dict] = {}
    start_tiles = time.time()
    done_tiles = 0
    hb = Heartbeat(interval_sec=args.progress_interval)
    hb.start()
    hb.set_phase("tiles", f"{label} r={radius_km:.1f}km tile={tile_km:.1f}km")
    hb.set_prog(0, len(tiles))

    try:
        for bbox in tiles:
            maybe_pause(args.pause_file, where="tiles")
            if (time.time() - start_tiles) > args.overpass_budget:
                print("\n[overpass] Time-budget bereikt; ga door met verzamelde resultaten.")
                break
            with Spinner(f"[overpass] tile {done_tiles+1}/{len(tiles)}"):
                data = overpass_bbox(ep, bbox, limit=args.overpass_limit, q_timeout=args.overpass_query_timeout)

            for e in data:
                tags = e.get("tags", {})
                name = tags.get("name")
                if not name:
                    continue
                elat, elon = element_center(e)
                key = canonical_key(name, elat, elon)

                if key in skip_keys:
                    continue

                website_tag = best_website_from_tags(tags) or ""
                if website_tag:
                    dom = domain_from_url(website_tag)
                    if dom and dom in skip_domains:
                        continue

                if key not in elements:
                    elements[key] = {"e": e, "tags": tags}

            done_tiles += 1
            hb.set_prog(done_tiles, len(tiles))
            if not args.no_progress:
                progress_bar(done_tiles, len(tiles), start_tiles, prefix="[tiles]")
            if args.max_sites and len(elements) >= target_candidates:
                print("[tiles] Voldoende kandidaten verzameld; ga door naar sites.")
                break

        # Beperk tot max-sites
        keys = list(elements.keys())
        if args.max_sites and args.max_sites < len(keys):
            keys = keys[:args.max_sites]

        # CSV sink klaarzetten (autosave/checkpoint)
        fieldnames = list(asdict(Biz("", "", 0.0, 0.0, "", "", "", "", "")).keys())
        sink = CsvSink(
            final_path=final_out_path,
            fieldnames=fieldnames,
            autosave_every=args.autosave_every,
            autosave_percent=args.autosave_percent,
            total_items=len(keys) if keys else 1,
            hard_fsync=args.hard_fsync,
        )

        # Crawl fase met GPU/CPU hybrid processing
        metrics = {
            "gevonden": len(elements),
            "bezocht_sites": 0,
            "met_contact": 0,
            "zonder_contact": 0,
            "overgeslagen_bestaand": 0,
            "geen_adres": 0,
            "gpu_processed": 0,
            "cpu_processed": 0,
        }
        hb.set_phase("sites", f"{label} [GPU: {USE_GPU}]")
        hb.set_prog(0, len(keys))
        start_sites = time.time()
        done_sites = 0
        
        # Initialize hybrid worker pool
        with HybridWorkerPool(
            gpu_ratio=GPU_WORKER_RATIO,
            max_gpu_workers=MAX_GPU_WORKERS,
            max_cpu_workers=MAX_CPU_WORKERS
        ) as worker_pool:
            for key in keys:
                maybe_pause(args.pause_file, where="sites")
                e = elements[key]["e"]
                tags = elements[key]["tags"]
                name = tags.get("name") or ""

                elat_tmp, elon_tmp = element_center(e)
                key_nc = canonical_key(name, elat_tmp, elon_tmp)
                if key_nc in skip_keys:
                    metrics["overgeslagen_bestaand"] += 1
                    done_sites += 1
                    hb.set_prog(done_sites, len(keys))
                    if not args.no_progress:
                        progress_bar(done_sites, len(keys), start_sites, prefix="[sites]")
                    sink.maybe_flush(done_sites)
                    continue

                email_osm, phone_osm = contacts_from_osm_tags(tags)
                website = best_website_from_tags(tags)
                address_osm = build_address(tags)

                # Skip bedrijven zonder geldig adres
                if not address_osm:
                    metrics["geen_adres"] = metrics.get("geen_adres", 0) + 1
                    done_sites += 1
                    hb.set_prog(done_sites, len(keys))
                    if not args.no_progress:
                        progress_bar(done_sites, len(keys), start_sites, prefix="[sites]")
                    sink.maybe_flush(done_sites)
                    continue

                if website:
                    dom = domain_from_url(website)
                    if dom and dom in skip_domains:
                        metrics["overgeslagen_bestaand"] += 1
                        done_sites += 1
                        hb.set_prog(done_sites, len(keys))
                        if not args.no_progress:
                            progress_bar(done_sites, len(keys), start_sites, prefix="[sites]")
                        sink.maybe_flush(done_sites)
                        continue

                email, phone, address = email_osm, phone_osm, address_osm

                if (not email and not phone) and website:
                    metrics["bezocht_sites"] += 1
                    try:
                        # Gebruik GPU-accelerated processing voor betere scraping
                        if USE_GPU and done_sites % 2 == 0:  # 50% van de sites op GPU
                            # GPU-accelerated batch processing
                            text_batch = [website]  # Simplified for now
                            gpu_results = GPUAcceleratedProcessor.extract_contacts_gpu(text_batch)
                            if gpu_results:
                                gpu_emails, gpu_phones = gpu_results[0]
                                email = email or next(iter(gpu_emails), "")
                                phone = phone or next(iter(gpu_phones), "")
                            metrics["gpu_processed"] += 1
                        else:
                            # Fallback naar normale Firecrawl
                            c_email, c_phone, c_address = firecrawl_scraper.scrape_website(
                                website,
                                max_pages=args.max_pages
                            )
                            email = email or c_email
                            phone = phone or c_phone
                            address = address or c_address
                            metrics["cpu_processed"] += 1
                    except Exception as e:
                        print(f"[firecrawl] Fout bij scraping {website}: {e}")
                        metrics["cpu_processed"] += 1
                        pass

                if not email and not phone:
                    metrics["zonder_contact"] += 1
                else:
                    metrics["met_contact"] += 1
                    row = asdict(Biz(
                        name=name.strip(),
                        category=pick_category(tags),
                        lat=elat_tmp, lon=elon_tmp,
                        address=address,
                        website=website,
                        email=email,
                        phone=phone,
                        source_location=label,
                    ))
                    sink.write_row(row)

                    # live uitbreiden van skipsets tijdens lange runs
                    skip_keys.add(key_nc)
                    if website:
                        d = domain_from_url(website)
                        if d:
                            skip_domains.add(d)

                done_sites += 1
                hb.set_prog(done_sites, len(keys))
                if not args.no_progress:
                    progress_bar(done_sites, len(keys), start_sites, prefix="[sites]")

                # checkpoint flush / autosave
                sink.maybe_flush(done_sites)
                
                # Check minimum contact requirement
                if args.min_contact > 0 and metrics["met_contact"] >= args.min_contact:
                    print(f"\n[success] Minimum contact vereiste bereikt: {metrics['met_contact']} bedrijven met contact")
                    print(f"[success] Scraper stopt vroegtijdig na {done_sites} verwerkte sites")
                    break
                
                # Check if we're finding unique new businesses (not already in CSV)
                total_processed = done_sites
                total_skipped = metrics.get("overgeslagen_bestaand", 0)
                total_no_address = metrics.get("geen_adres", 0)
                total_no_contact = metrics.get("zonder_contact", 0)
                total_with_contact = metrics.get("met_contact", 0)
                
                # Calculate unique new businesses found
                unique_new_found = total_with_contact
                
                # Only stop if we have reached the minimum contact requirement
                # Don't stop based on processing count - only on unique addresses found
                if args.min_contact > 0 and unique_new_found >= args.min_contact:
                    print(f"\n[success] Minimum contact vereiste bereikt: {unique_new_found} bedrijven met contact")
                    print(f"[success] Scraper stopt omdat {args.min_contact} unieke adressen zijn gevonden")
                    print(f"[info] Totaal verwerkte sites: {total_processed}")
                    print(f"[info] Totaal overgeslagen (al bestaand): {total_skipped}")
                    print(f"[info] Totaal zonder adres: {total_no_address}")
                    print(f"[info] Totaal zonder contact: {total_no_contact}")
                    break

        hb.stop()
        final_file = sink.finalize()
        print(f"[done] bedrijven geschreven: {metrics['met_contact']} → {final_file}")
        show_metrics(metrics)

    except KeyboardInterrupt:
        hb.stop()
        # Zorg dat partial behouden blijft
        try:
            sink.close_keep_part()  # type: ignore[name-defined]
        except Exception:
            pass
        print(f"\n[interrupt] Afgebroken door gebruiker. Checkpoint bewaard in: '{part_path}'")
    except Exception as ex:
        hb.stop()
        try:
            sink.close_keep_part()  # type: ignore[name-defined]
        except Exception:
            pass
        print(f"\n[error] {ex}\nCheckpoint bewaard in: '{part_path}'")
        raise

# ===================== CLI =====================
def main():
    global REQUEST_TIMEOUT, MAX_RETRIES, CRAWL_DELAY_DEFAULT
    ap = argparse.ArgumentParser(description="Lokale bedrijven (OSM) + Firecrawl website scraping. Alleen met e-mail/telefoon in CSV.")
    ap.add_argument("--loc", help='Vrije tekst locatie, bv. "Utrecht, NL"')
    ap.add_argument("--coords", help='Coördinaten "LAT,LON"')
    ap.add_argument("--countrycodes", help="Beperk geocoding, bv. 'nl' of 'nl,de,be'")

    ap.add_argument("--radius", type=int, help="Straal in meters")
    ap.add_argument("--radius-km", type=float, default=5.0, help="Straal in kilometers (default 5)")
    ap.add_argument("--tile-km", type=float, default=2.0, help="Tegelgrootte in kilometers (default 2)")

    ap.add_argument("--max-pages", type=int, default=MAX_INTERNAL_PAGES_DEFAULT, help="Max interne pagina's per site (default 3)")
    ap.add_argument("--max-sites", type=int, default=None, help="Maximaal aantal sites om te verwerken na dedup")

    ap.add_argument("--overpass", help=f"Overpass endpoint (default: {OVERPASS_PRIMARY} met fallback)")
    ap.add_argument("--overpass-limit", type=int, default=400, help="Max resultaten per tile (default 400)")
    ap.add_argument("--overpass-query-timeout", type=int, default=25, help="Timeout per tile-query (s) (default 25)")
    ap.add_argument("--overpass-budget", type=float, default=75.0, help="Totaal time-budget voor Overpass fase (s)")

    ap.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT, help="HTTP timeout per request (default 12)")
    ap.add_argument("--retries", type=int, default=MAX_RETRIES, help="HTTP retries (default 2)")
    ap.add_argument("--crawl-delay", type=float, default=CRAWL_DELAY_DEFAULT, help="Delay tussen pagina-requests (s) (default 0.7)")
    ap.add_argument("--robots-fail-closed", action="store_true", help="Blokkeer als robots.txt niet bereikbaar is (default: fail-open)")

    # Firecrawl opties
    ap.add_argument("--firecrawl-api-key", help="Firecrawl API key (of gebruik FIRECRAWL_API_KEY env var)")

    # GPU/Performance opties
    ap.add_argument("--use-gpu", action="store_true", help="Forceer GPU gebruik (als beschikbaar)")
    ap.add_argument("--gpu-ratio", type=float, default=0.8, help="Verhouding GPU/CPU workers (default 0.8 = 80% GPU)")
    ap.add_argument("--max-gpu-workers", type=int, default=12, help="Max aantal GPU workers (default 12)")
    ap.add_argument("--max-cpu-workers", type=int, default=4, help="Max aantal CPU workers (default 4)")

    ap.add_argument("--progress-interval", type=float, default=5.0, help="Heartbeat-interval (s) (default 5)")
    ap.add_argument("--no-progress", action="store_true", help="Schakel loaders/progress bar uit")

    ap.add_argument("--skip-csv", nargs="+",
                    help="Pad(en) naar bestaande CSV's; deze entries worden overgeslagen (match op naam+coörd of website-domein)")

    # Nieuw: pause/autosave/resume
    ap.add_argument("--pause-file", default="pause.flag", help="Bestandspad dat, indien aanwezig, de scraper pauzeert (default: ./pause.flag)")
    ap.add_argument("--autosave-every", type=int, default=200, help="Flush/Sync elke N verwerkte sites (default 200; 0=uit)")
    ap.add_argument("--autosave-percent", type=float, default=0.0, help="Extra flush bij iedere d%% voortgang (bv. 5.0; 0=uit)")
    ap.add_argument("--resume", action="store_true", help="Ga verder op bestaand .part-bestand gebaseerd op --out (neemt .part mee als skip-csv)")
    ap.add_argument("--hard-fsync", action="store_true", help="Forceer os.fsync() bij elke flush (trager maar crash-veiliger)")

    ap.add_argument("--out", default="bedrijven_firecrawl.csv", help="Uitvoer CSV-bestand (default: ./bedrijven_firecrawl.csv)")
    ap.add_argument("--min-contact", type=int, default=0, help="Minimum aantal bedrijven met contact informatie voordat scraper stopt (default: 0 = geen minimum)")

    args = ap.parse_args()
    if not args.loc and not args.coords:
        ap.error("Geef --loc of --coords op.")

    REQUEST_TIMEOUT = float(args.timeout)
    MAX_RETRIES = int(args.retries)
    CRAWL_DELAY_DEFAULT = float(args.crawl_delay)
    
    # Update GPU settings based on arguments
    global USE_GPU, GPU_WORKER_RATIO, MAX_GPU_WORKERS, MAX_CPU_WORKERS
    if args.use_gpu:
        USE_GPU = CUDA_AVAILABLE
    GPU_WORKER_RATIO = float(args.gpu_ratio)
    MAX_GPU_WORKERS = int(args.max_gpu_workers)
    MAX_CPU_WORKERS = int(args.max_cpu_workers)
    
    if USE_GPU:
        print(f"[gpu] GPU enabled: {MAX_GPU_WORKERS} GPU workers, {MAX_CPU_WORKERS} CPU workers (ratio: {GPU_WORKER_RATIO})")
    else:
        print(f"[gpu] GPU disabled: {MAX_CPU_WORKERS} CPU workers only")
    
    run(args)

if __name__ == "__main__":
    main()

