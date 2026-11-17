"""CHEM OCR encoder.

Memes, screenshots and posters carry much of their meaning in text that is
rendered *inside* the image.  CLIP sees the picture but cannot reliably read the
caption, and the text encoder never sees it at all.  This module extracts that
embedded text with EasyOCR so the API can run it through the same text
pipeline (toxic-bert + phrase patterns) as typed text.

Design notes
------------
* EasyOCR is imported and its models are loaded lazily and only once.
* Failure is never silent: ``extract`` always returns a dict with an ``error``
  field (``None`` on success) so the API can show the user *why* no text was
  found instead of quietly scoring the image as if it had none.
* Thresholds come from ``policy/policy_config.json``
  (``ocr_min_confidence``, ``ocr_min_chars``).

Public API
----------
``OCREncoder(languages=None, gpu=None, config_path=None)``
``OCREncoder.extract(image) -> dict``
    ``{"text": str, "lines": [{"text", "confidence"}], "available": bool,
       "error": str | None}``
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from utils.config import get_coefficient

# Images whose shorter side is below this are upscaled before OCR; small meme
# thumbnails are otherwise read poorly.  This is an image-size heuristic, not a
# harm coefficient.
_MIN_SIDE_PX = 640
# Upper bound on the longer side so huge photos do not make OCR crawl on CPU.
_MAX_SIDE_PX = 2400
# Cap on returned characters (the text encoder truncates at 256 tokens anyway).
_MAX_CHARS = 1500


class OCREncoder:
    """Extract embedded text from an image with EasyOCR."""

    modality = "ocr"

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        gpu: Optional[bool] = None,
        config_path: Optional[str] = None,
    ):
        langs = languages or [
            code.strip()
            for code in os.environ.get("CHEM_OCR_LANGS", "en").split(",")
            if code.strip()
        ]
        self._languages = langs or ["en"]
        self._gpu = gpu
        self._config_path = config_path
        self._min_conf = float(get_coefficient("ocr_min_confidence", config_path))
        self._min_chars = int(get_coefficient("ocr_min_chars", config_path))

        self._reader = None
        self.available = False
        self.load_error: Optional[str] = None
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        """Load the EasyOCR reader (downloads model weights on first use)."""
        try:
            import easyocr  # noqa: WPS433 - optional heavy dependency

            if self._gpu is None:
                try:
                    import torch

                    self._gpu = bool(torch.cuda.is_available())
                except Exception:
                    self._gpu = False

            print(
                f"[CHEM][ocr_encoder] Loading EasyOCR langs={self._languages} "
                f"gpu={self._gpu} ..."
            )
            self._reader = easyocr.Reader(self._languages, gpu=self._gpu, verbose=False)
            self.available = True
            print("[CHEM][ocr_encoder] Ready.")
        except Exception as exc:  # missing package, no network for weights, ...
            self.available = False
            self.load_error = f"{type(exc).__name__}: {exc}"
            print(f"[CHEM][ocr_encoder] OCR unavailable: {self.load_error}")

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _to_array(image: Any):
        """Return an RGB numpy array, upscaled/downscaled to a sane OCR size."""
        import numpy as np
        from PIL import Image

        if not isinstance(image, Image.Image):
            raise TypeError(f"expected PIL.Image, got {type(image)!r}")

        pil = image.convert("RGB")
        w, h = pil.size
        short, long_ = min(w, h), max(w, h)

        scale = 1.0
        if short < _MIN_SIDE_PX:
            scale = _MIN_SIDE_PX / float(short)
        if long_ * scale > _MAX_SIDE_PX:
            scale = _MAX_SIDE_PX / float(long_)

        if abs(scale - 1.0) > 0.01:
            pil = pil.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.LANCZOS,
            )
        return np.array(pil)

    @staticmethod
    def _reading_order(results: List[Any]) -> List[Any]:
        """Sort detections top-to-bottom, then left-to-right within a row."""

        def key(item: Any):
            box = item[0]
            ys = [pt[1] for pt in box]
            xs = [pt[0] for pt in box]
            return (round(min(ys) / 25.0), min(xs))

        return sorted(results, key=key)

    # --------------------------------------------------------------- extract
    def extract(self, image: Any) -> Dict[str, Any]:
        """Read all text in ``image``.

        Inputs: a PIL.Image.
        Outputs: dict with ``text`` (joined, reading order), per-line
        ``lines`` with confidences, ``available`` and ``error``.
        """
        if not self.available or self._reader is None:
            return {
                "text": "",
                "lines": [],
                "available": False,
                "error": self.load_error or "OCR engine is not loaded",
            }

        try:
            array = self._to_array(image)
            raw = self._reader.readtext(array, detail=1, paragraph=False)
        except Exception as exc:
            print(f"[CHEM][ocr_encoder] OCR failed: {exc}")
            return {
                "text": "",
                "lines": [],
                "available": True,
                "error": f"{type(exc).__name__}: {exc}",
            }

        kept = [r for r in raw if float(r[2]) >= self._min_conf and str(r[1]).strip()]
        lines = [
            {"text": str(r[1]).strip(), "confidence": round(float(r[2]), 4)}
            for r in self._reading_order(kept)
        ]
        text = " ".join(line["text"] for line in lines)[:_MAX_CHARS].strip()

        # Ignore noise: a couple of stray characters is not a caption.
        if sum(ch.isalnum() for ch in text) < self._min_chars:
            text, lines = "", []

        print(
            f"[CHEM][ocr_encoder] -> {len(lines)} line(s), {len(text)} chars "
            f"(raw detections={len(raw)})"
        )
        return {"text": text, "lines": lines, "available": True, "error": None}
