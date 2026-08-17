"""Proje yapilandirmasini yukleyen yardimcilar."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def yukle(yol: str | Path) -> dict[str, Any]:
    """YAML yapilandirmasini sozluk olarak yukler."""
    with Path(yol).resolve().open("r", encoding="utf-8") as dosya:
        return yaml.safe_load(dosya) or {}


def aktif_ortam(config: dict[str, Any]) -> dict[str, Any]:
    """Aktif ortam ayarlarini dondurur."""
    ad = config.get("aktif_ortam", "lokal")
    ortamlar = config.get("ortamlar", {})
    if ad not in ortamlar:
        raise ValueError(f"Bilinmeyen ortam: {ad}")
    return ortamlar[ad]
