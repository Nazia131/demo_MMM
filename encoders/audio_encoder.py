"""CHEM audio encoder.

Analyses an audio clip along two complementary axes:

* **Transcript toxicity** -- Whisper (``base``) transcribes speech; the
  transcript is scored by a reused :class:`TextEncoder` so spoken slurs,
  threats, etc. surface through the same policy machinery as written text.
* **Acoustic agitation** -- librosa extracts low-level acoustic features
  (RMS energy, spectral centroid, pitch via ``pyin``, zero-crossing rate,
  tempo).  Agitation is flagged *statistically* from the intra-clip pitch
  variability (std / mean ratio) against the configured
  ``acoustic_agitation_sigma_ratio`` -- there are no hardcoded acoustic
  thresholds.

The final harm score is the config-weighted blend
``audio_transcript_weight * transcript + audio_acoustic_weight * acoustic``.

Public API
----------
``AudioEncoder(text_encoder=None, config_path=None, device=None, whisper_model="base")``
``AudioEncoder.encode(audio_path) -> dict``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from utils.config import get_coefficient
from utils.embeddings import to_fixed_dim
from encoders.text_encoder import TextEncoder


class AudioEncoder:
    """Encode audio into CHEM elements, blending transcript + acoustic signals.

    Whisper and (a reference to) a :class:`TextEncoder` are held for the
    lifetime of the instance so no model is reloaded per call.

    Parameters
    ----------
    text_encoder : TextEncoder, optional
        A pre-built text encoder to reuse for transcript toxicity.  A new one
        is constructed if not supplied.
    config_path : str, optional
        Override path to ``policy_config.json``.
    device : str, optional
        Torch device string; auto-selected when omitted.
    whisper_model : str, default "base"
        Whisper checkpoint size to load.

    Attributes
    ----------
    modality : str
        Always ``"audio"``.
    """

    #: This encoder's fixed modality tag.
    modality = "audio"

    def __init__(
        self,
        text_encoder: Optional[TextEncoder] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        whisper_model: str = "base",
    ):
        """Load Whisper once and attach/create a reusable TextEncoder.

        Inputs: optional text encoder, config path, device, whisper size.
        Outputs: a ready ``AudioEncoder`` instance.
        """
        import torch
        import whisper  # openai-whisper

        self._config_path = config_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(
            f"[CHEM][audio_encoder] Loading Whisper '{whisper_model}' "
            f"on device={self.device} ..."
        )
        self._whisper = whisper.load_model(whisper_model, device=self.device)

        # Reuse an existing TextEncoder if provided, else build one so that
        # transcript toxicity flows through identical policy logic as text.
        if text_encoder is not None:
            print("[CHEM][audio_encoder] Reusing provided TextEncoder.")
            self._text_encoder = text_encoder
        else:
            print("[CHEM][audio_encoder] No TextEncoder provided; building one.")
            self._text_encoder = TextEncoder(config_path=config_path, device=device)

        # Config-driven blend weights and the statistical agitation threshold.
        self._w_transcript = float(get_coefficient("audio_transcript_weight", config_path))
        self._w_acoustic = float(get_coefficient("audio_acoustic_weight", config_path))
        self._agitation_sigma_ratio = float(
            get_coefficient("acoustic_agitation_sigma_ratio", config_path)
        )

        print(
            f"[CHEM][audio_encoder] Ready. weights(transcript={self._w_transcript}, "
            f"acoustic={self._w_acoustic}) sigma_ratio={self._agitation_sigma_ratio}"
        )

    # ------------------------------------------------------------------ utils
    def _extract_acoustic_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract librosa acoustic features and a statistical agitation flag.

        Features: mean RMS energy, mean spectral centroid, pitch mean/std
        (via ``librosa.pyin``), zero-crossing rate, tempo, and clip duration.
        Agitation is flagged when the pitch coefficient of variation
        (std / mean) exceeds ``acoustic_agitation_sigma_ratio`` -- a purely
        relative, per-clip statistic with no hardcoded frequency thresholds.

        Inputs: path to an audio file.
        Outputs: a dict of features plus ``agitation`` (float in 0..1) and
        ``agitated`` (bool) and ``duration`` (seconds).
        """
        import librosa
        import numpy as np

        # Load at native sample rate (mono).
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = float(librosa.get_duration(y=y, sr=sr)) if len(y) else 0.0

        if len(y) == 0:
            return {
                "rms_energy": 0.0,
                "spectral_centroid": 0.0,
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "zero_crossing_rate": 0.0,
                "tempo": 0.0,
                "duration": 0.0,
                "agitation": 0.0,
                "agitated": False,
            }

        # RMS energy (loudness proxy).
        rms = float(np.mean(librosa.feature.rms(y=y)))
        # Spectral centroid (brightness / sharpness proxy).
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        # Zero-crossing rate (noisiness proxy).
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        # Tempo (rhythmic intensity proxy).
        try:
            tempo = float(np.atleast_1d(librosa.beat.tempo(y=y, sr=sr))[0])
        except Exception:
            tempo = 0.0

        # Pitch track via probabilistic YIN; drop unvoiced (NaN) frames.
        try:
            f0, _voiced, _prob = librosa.pyin(
                y,
                fmin=float(librosa.note_to_hz("C2")),
                fmax=float(librosa.note_to_hz("C7")),
                sr=sr,
            )
            voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        except Exception:
            voiced = np.array([])

        if voiced.size > 1:
            pitch_mean = float(np.mean(voiced))
            pitch_std = float(np.std(voiced))
        else:
            pitch_mean = float(voiced[0]) if voiced.size == 1 else 0.0
            pitch_std = 0.0

        # STATISTICAL agitation: coefficient of variation of the pitch track.
        # High relative pitch variability => agitated / shouting-like prosody.
        if pitch_mean > 1e-6:
            cov = pitch_std / pitch_mean
        else:
            cov = 0.0
        agitated = cov > self._agitation_sigma_ratio
        # Normalise the CoV against the threshold into a bounded 0..1 signal.
        agitation = float(min(1.0, cov / self._agitation_sigma_ratio)) if self._agitation_sigma_ratio > 0 else 0.0

        return {
            "rms_energy": rms,
            "spectral_centroid": centroid,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pitch_cov": float(cov),
            "zero_crossing_rate": zcr,
            "tempo": tempo,
            "duration": duration,
            "agitation": agitation,
            "agitated": bool(agitated),
        }

    def _feature_vector(self, features: Dict[str, Any]) -> List[float]:
        """Turn the acoustic feature dict into a fixed 768-d node embedding.

        Inputs: the acoustic feature dict from :meth:`_extract_acoustic_features`.
        Outputs: list[float] length 768 (small feature vector, zero-padded).
        """
        raw = [
            features.get("rms_energy", 0.0),
            features.get("spectral_centroid", 0.0),
            features.get("pitch_mean", 0.0),
            features.get("pitch_std", 0.0),
            features.get("pitch_cov", 0.0),
            features.get("zero_crossing_rate", 0.0),
            features.get("tempo", 0.0),
            features.get("agitation", 0.0),
        ]
        return to_fixed_dim(raw)

    # ------------------------------------------------------------------ encode
    def encode(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe, score toxicity, and analyse acoustics for one clip.

        Inputs
        ------
        audio_path : str
            Path to an audio file readable by Whisper / librosa.

        Outputs
        -------
        dict
            ``{"elements": [...], "harm_score": float, "transcript": str,
               "language": str, "acoustic_features": dict,
               "modality": "audio"}``.  Every element carries a
            ``temporal_position`` (seconds).
        """
        print(f"[CHEM][audio_encoder] encode() path={audio_path}")

        # --- Transcription ---------------------------------------------------
        # Failures here used to be swallowed, which looked like "audio is
        # silent / harmless".  Every problem is now collected in ``warnings``
        # and returned so the API/UI can show it.
        transcript = ""
        language = "unknown"
        warnings: List[str] = []

        # Decode ourselves (soundfile/librosa) and hand Whisper a 16 kHz mono
        # array.  Whisper's own file loader shells out to the ``ffmpeg``
        # binary and raises if it is not installed; this avoids that for
        # wav/flac/ogg and most mp3 files.
        audio_input: Any = audio_path
        try:
            import librosa

            samples, _sr = librosa.load(audio_path, sr=16000, mono=True)
            if samples is not None and len(samples) > 0:
                audio_input = samples
            else:
                warnings.append("the audio file decoded to zero samples (empty file?).")
        except Exception as exc:
            warnings.append(
                f"could not decode the file with librosa ({type(exc).__name__}: {exc}); "
                "falling back to Whisper's ffmpeg loader."
            )

        try:
            result = self._whisper.transcribe(
                audio_input,
                fp16=(self.device == "cuda"),
            )
            transcript = (result.get("text") or "").strip()
            language = result.get("language", "unknown")
            if not transcript:
                warnings.append("no speech was detected in the audio.")
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            if "ffmpeg" in msg.lower() or isinstance(exc, FileNotFoundError):
                msg += " -- install ffmpeg (https://ffmpeg.org) or upload a .wav file."
            warnings.append(f"transcription failed: {msg}")
            print(f"[CHEM][audio_encoder] Whisper transcription failed: {msg}")

        # --- Transcript toxicity via the reused TextEncoder ------------------
        text_result = self._text_encoder.encode(transcript) if transcript else {
            "elements": [],
            "harm_score": 0.0,
        }
        transcript_harm = float(text_result.get("harm_score", 0.0))

        # --- Acoustic feature extraction ------------------------------------
        try:
            features = self._extract_acoustic_features(audio_path)
        except Exception as exc:
            print(f"[CHEM][audio_encoder] Acoustic extraction failed: {exc}")
            warnings.append(
                f"acoustic analysis failed ({type(exc).__name__}: {exc}); "
                "only the transcript was scored."
            )
            features = {
                "rms_energy": 0.0,
                "spectral_centroid": 0.0,
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "pitch_cov": 0.0,
                "zero_crossing_rate": 0.0,
                "tempo": 0.0,
                "duration": 0.0,
                "agitation": 0.0,
                "agitated": False,
            }
        acoustic_harm = float(features.get("agitation", 0.0))
        duration = float(features.get("duration", 0.0))

        # --- Blend into a single harm score ---------------------------------
        # What is *said* is the evidence; a loud or agitated voice only adds to
        # it.  Taking the max with the transcript score stops a clearly toxic
        # transcript spoken calmly from being diluted to 0.7x (which could
        # fall under the scorer's audio cut-off).
        blended = self._w_transcript * transcript_harm + self._w_acoustic * acoustic_harm
        harm_score = max(transcript_harm, blended)

        # --- Build elements (all with temporal_position) --------------------
        elements: List[Dict[str, Any]] = []
        idx = 0

        # (a) Base audio node carrying the acoustic feature embedding.
        elements.append(
            {
                "id": f"audio_{idx}",
                "label": "audio",
                "type": "audio",
                "embedding": self._feature_vector(features),
                "confidence": 1.0,
                "harmful_signal": bool(features.get("agitated", False)),
                "full_label": (
                    "acoustically agitated speech"
                    if features.get("agitated")
                    else "audio segment"
                ),
                "modality": self.modality,
                "temporal_position": 0.0,
                "text": transcript,
                "language": language,
                "acoustic_features": features,
            }
        )
        idx += 1

        # (b) Re-emit transcript-derived text elements as audio-modality nodes
        #     so downstream graph edges see them under this clip's timeline.
        for tel in text_result.get("elements", []):
            # Skip the plain base text node (idx 0 in text output) — the audio
            # base node above already represents the clip; keep harmful ones.
            if not tel.get("harmful_signal", False):
                continue
            el = dict(tel)
            el["id"] = f"audio_{idx}"
            el["modality"] = self.modality
            el["temporal_position"] = 0.0
            el["text"] = transcript
            el["language"] = language
            elements.append(el)
            idx += 1

        print(
            f"[CHEM][audio_encoder] -> lang={language} transcript_harm={transcript_harm:.4f} "
            f"acoustic_harm={acoustic_harm:.4f} harm_score={harm_score:.4f} "
            f"duration={duration:.2f}s elements={len(elements)}"
        )
        return {
            "elements": elements,
            "harm_score": float(harm_score),
            "transcript": transcript,
            "language": language,
            "acoustic_features": features,
            "warnings": warnings,
            "modality": self.modality,
        }
