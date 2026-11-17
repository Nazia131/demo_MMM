"""CHEM FastAPI inference service.

Exposes the full Compositional Harm Emergence Modeling pipeline over HTTP:

    POST /analyze   multipart form (text?, image?, audio?, video?) -> verdict
    GET  /health    liveness + model-load status

Prototype inference flow:

    Text/Image/Audio/Video
            ↓
        Encoders
            ↓
    Content Composition Graph
            ↓
    Deterministic PrototypeHarmScorer
            ↓
    Policy-aware verdict + explanation

The learned GAT is intentionally not used for the prototype decision layer.
This keeps the multimodal architecture intact while replacing the currently
unreliable/untrained GAT inference with deterministic inference.
"""

from __future__ import annotations

import io
import os
import tempfile
import traceback
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from utils.config import get_coefficient, load_policy_config


def _is_exculpatory(text: Optional[str]) -> bool:
    """Return True if text contains an exculpatory keyword."""
    if not text:
        return False

    cfg = load_policy_config()
    keywords = cfg.get("exculpatory_keywords", {}).get("keywords", [])

    lower = text.lower()
    return any(kw.lower() in lower for kw in keywords)


# ---------------------------------------------------------------------------
# Pipeline singleton
# ---------------------------------------------------------------------------

class CHEMPipeline:
    """End-to-end CHEM inference pipeline."""

    def __init__(self) -> None:
        self.loaded = False
        self.load_error: Optional[str] = None

        self.text_encoder = None
        self.image_encoder = None
        self.audio_encoder = None
        self.video_encoder = None

        self.graph_builder = None
        self.harm_scorer = None

        self.shap_explainer = None
        self.gnn_explainer = None
        self.report_generator = None

        self._fuse = None

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load all required components once."""

        if self.loaded:
            return

        try:
            print("[CHEM][api] Loading models (one-time startup)...")

            # ---------------------------------------------------------- #
            # Encoders
            # ---------------------------------------------------------- #

            from encoders.text_encoder import TextEncoder
            from encoders.image_encoder import ImageEncoder
            from graph.graph_builder import ContentCompositionGraphBuilder

            from model.prototype_scorer import PrototypeHarmScorer

            # Optional explainability/reporting components.
            try:
                from explainability.shap_explainer import SHAPExplainer
            except Exception as exc:
                SHAPExplainer = None
                print(
                    f"[CHEM][api] SHAP explainer unavailable: {exc}"
                )

            try:
                from explainability.gnn_explainer import GNNExplainerWrapper
            except Exception as exc:
                GNNExplainerWrapper = None
                print(
                    f"[CHEM][api] GNN explainer unavailable: {exc}"
                )

            try:
                from explainability.report_generator import ReportGenerator
            except Exception as exc:
                ReportGenerator = None
                print(
                    f"[CHEM][api] Report generator unavailable: {exc}"
                )

            # ---------------------------------------------------------- #
            # Text
            # ---------------------------------------------------------- #

            print("[CHEM][api] Loading text encoder...")
            self.text_encoder = TextEncoder()

            # ---------------------------------------------------------- #
            # Image
            # ---------------------------------------------------------- #

            print("[CHEM][api] Loading image encoder...")
            self.image_encoder = ImageEncoder()

            # ---------------------------------------------------------- #
            # Audio
            # ---------------------------------------------------------- #

            try:
                from encoders.audio_encoder import AudioEncoder

                print("[CHEM][api] Loading audio encoder...")
                self.audio_encoder = AudioEncoder(
                    text_encoder=self.text_encoder
                )

                print("[CHEM][api] Audio encoder ready.")

            except Exception as exc:
                self.audio_encoder = None

                print(
                    f"[CHEM][api] Audio encoder unavailable: {exc}"
                )

            # ---------------------------------------------------------- #
            # Video
            # ---------------------------------------------------------- #

            try:
                from encoders.video_encoder import VideoEncoder

                print("[CHEM][api] Loading video encoder...")
                self.video_encoder = VideoEncoder(
                    image_encoder=self.image_encoder
                )

                print("[CHEM][api] Video encoder ready.")

            except Exception as exc:
                self.video_encoder = None

                print(
                    f"[CHEM][api] Video encoder unavailable: {exc}"
                )

            # ---------------------------------------------------------- #
            # Graph builder
            # ---------------------------------------------------------- #

            print("[CHEM][api] Initializing graph builder...")
            self.graph_builder = ContentCompositionGraphBuilder()

            # ---------------------------------------------------------- #
            # Deterministic prototype scorer
            # ---------------------------------------------------------- #

            print("[CHEM][api] Initializing prototype harm scorer...")
            self.harm_scorer = PrototypeHarmScorer()

            # ---------------------------------------------------------- #
            # Optional explainability
            # ---------------------------------------------------------- #

            # The prototype scorer does not expose a learned GAT model.
            # Therefore learned-GAT explainers are disabled unless a model
            # is available for compatibility with the existing pipeline.

            gat = getattr(self.harm_scorer, "model", None)

            if SHAPExplainer is not None and gat is not None:
                try:
                    self.shap_explainer = SHAPExplainer(gat)
                except Exception as exc:
                    self.shap_explainer = None
                    print(
                        f"[CHEM][api] SHAP initialization skipped: {exc}"
                    )

            if GNNExplainerWrapper is not None and gat is not None:
                try:
                    self.gnn_explainer = GNNExplainerWrapper(gat)
                except Exception as exc:
                    self.gnn_explainer = None
                    print(
                        f"[CHEM][api] GNN explainer initialization skipped: {exc}"
                    )

            if ReportGenerator is not None:
                try:
                    self.report_generator = ReportGenerator()
                except Exception as exc:
                    self.report_generator = None
                    print(
                        f"[CHEM][api] Report generator unavailable: {exc}"
                    )

            self.loaded = True
            self.load_error = None

            print("[CHEM][api] All available models loaded.")
            print("[CHEM][api] Deterministic prototype inference is ACTIVE.")
            print("[CHEM][api] Service ready.")

        except Exception as exc:
            self.loaded = False
            self.load_error = f"{type(exc).__name__}: {exc}"

            print(
                f"[CHEM][api] Model load FAILED: {self.load_error}"
            )

            traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        text: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run CHEM inference on any subset of modalities."""

        if not self.loaded:
            raise RuntimeError(
                "CHEM models are not loaded: "
                f"{self.load_error or 'unknown error'}"
            )

        elements: List[Dict[str, Any]] = []
        modality_scores: Dict[str, float] = {}

        # ==============================================================
        # TEXT
        # ==============================================================

        if text:
            print("[CHEM][api] Encoding text...")

            res = self.text_encoder.encode(text)

            text_elements = res.get("elements", [])

            elements.extend(
                _reindex(
                    text_elements,
                    "text",
                    len(elements),
                )
            )

            modality_scores["text"] = float(
                res.get("harm_score", 0.0)
            )

        # ==============================================================
        # IMAGE
        # ==============================================================

        if image_bytes is not None:
            print("[CHEM][api] Encoding image...")

            from PIL import Image

            pil = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            res = self.image_encoder.encode(pil)

            image_elements = res.get("elements", [])

            elements.extend(
                _reindex(
                    image_elements,
                    "image",
                    len(elements),
                )
            )

            modality_scores["image"] = float(
                res.get("harm_score", 0.0)
            )

        # ==============================================================
        # AUDIO
        # ==============================================================

        if audio_path is not None:

            if self.audio_encoder is None:
                raise ValueError(
                    "Audio analysis is unavailable because "
                    "the audio encoder failed to load."
                )

            print("[CHEM][api] Encoding audio...")

            res = self.audio_encoder.encode(audio_path)

            audio_elements = res.get("elements", [])

            elements.extend(
                _reindex(
                    audio_elements,
                    "audio",
                    len(elements),
                )
            )

            modality_scores["audio"] = float(
                res.get("harm_score", 0.0)
            )

        # ==============================================================
        # VIDEO
        # ==============================================================

        if video_path is not None:

            if self.video_encoder is None:
                raise ValueError(
                    "Video analysis is unavailable because "
                    "the video encoder failed to load."
                )

            print("[CHEM][api] Encoding video...")

            res = self.video_encoder.encode(video_path)

            video_elements = res.get("elements", [])

            elements.extend(
                _reindex(
                    video_elements,
                    "video",
                    len(elements),
                )
            )

            modality_scores["video"] = float(
                res.get("harm_score", 0.0)
            )

        # ==============================================================
        # EMPTY INPUT
        # ==============================================================

        if not elements:
            return {
                "harm_score": 0.0,
                "verdict": "SAFE",
                "policy_clause": None,
                "triggered_category": None,
                "triggered_elements": [],
                "triggered_patterns": [],
                "nodes": [],
                "edges": [],
                "explanation": (
                    "No content was provided to analyse."
                ),
                "cross_modal_flag": False,
                "harmful_modalities": [],
                "shap_scores": [],
                "modality_scores": {},
            }

        # ==============================================================
        # EXCULPATORY CONTEXT
        # ==============================================================

        if _is_exculpatory(text):

            print(
                "[CHEM][api] Exculpatory context detected."
            )

            factor = get_coefficient(
                "exculpatory_dampening_factor"
            )

            for mod in ("image", "video"):

                if mod in modality_scores:

                    modality_scores[mod] = (
                        modality_scores[mod] * factor
                    )

        # ==============================================================
        # CONTENT COMPOSITION GRAPH
        # ==============================================================

        print(
            "[CHEM][api] Building Content Composition Graph..."
        )

        graph = self.graph_builder.build(elements)

        # ==============================================================
        # DETERMINISTIC PROTOTYPE SCORING
        # ==============================================================

        print(
            "[CHEM][api] Running deterministic prototype scorer..."
        )

        score = self.harm_scorer.score(
            graph,
            modality_scores=modality_scores,
        )

        # ==============================================================
        # FINAL DECISION
        # ==============================================================

        final_score = float(
            score.get("harm_score", 0.0)
        )

        threshold = float(
    get_coefficient("final_harm_threshold")
)

        verdict = (
            "HARMFUL"
            if final_score >= threshold
            else "SAFE"
        )

        matched_pattern = score.get(
            "matched_pattern"
        )

        policy_clause = score.get(
            "policy_clause"
        )

        triggered_category = None

        if isinstance(policy_clause, dict):

            triggered_category = policy_clause.get(
                "clause_id"
            )

        # ==============================================================
        # EXPLANATION
        # ==============================================================

        shap_scores: List[Any] = []

        explanation = score.get(
            "explanation",
            "",
        )

        # The prototype scorer already produces a deterministic
        # explanation. Learned explainability is optional and should
        # never prevent the demo from working.

        if (
            self.shap_explainer is not None
            or self.gnn_explainer is not None
        ):

            try:

                shap_result = (
                    self.shap_explainer.explain(graph)
                    if self.shap_explainer
                    else {}
                )

                gnn_result = (
                    self.gnn_explainer.explain(graph)
                    if self.gnn_explainer
                    else {}
                )

                shap_scores = shap_result.get(
                    "node_scores",
                    shap_result.get(
                        "attributions",
                        []
                    ),
                )

                if self.report_generator is not None:

                    report = self.report_generator.generate(
                        graph_dict=graph,
                        shap_result=shap_result,
                        gnn_result=gnn_result,
                        matched_pattern=matched_pattern,
                        policy_clause=policy_clause,
                    )

                    report_explanation = report.get(
                        "explanation",
                        "",
                    )

                    if report_explanation:
                        explanation = report_explanation

            except Exception as exc:

                print(
                    "[CHEM][api] Explainability degraded: "
                    f"{exc}"
                )

        # ==============================================================
        # RESPONSE
        # ==============================================================

        return {
            "harm_score": round(
                final_score,
                4,
            ),

            "verdict": verdict,

            "policy_clause": policy_clause,

            "triggered_category": triggered_category,

            "matched_pattern": matched_pattern,

            "triggered_elements": score.get(
                "triggered_elements",
                [],
            ),

            "triggered_patterns": score.get(
                "triggered_patterns",
                [],
            ),

            "harmful_modalities": score.get(
                "harmful_modalities",
                [],
            ),

            "nodes": graph.get(
                "nodes",
                [],
            ),

            "edges": graph.get(
                "edges",
                [],
            ),

            "explanation": explanation,

            "cross_modal_flag": bool(
                score.get(
                    "cross_modal_flag",
                    False,
                )
            ),

            "shap_scores": shap_scores,

            "modality_scores": {
                key: round(
                    float(value),
                    4,
                )
                for key, value in modality_scores.items()
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reindex(
    elements: List[Dict[str, Any]],
    modality: str,
    offset: int,
) -> List[Dict[str, Any]]:
    """Make element IDs globally unique across modalities."""

    out: List[Dict[str, Any]] = []

    for i, element in enumerate(elements):

        el = dict(element)

        el.setdefault(
            "modality",
            modality,
        )

        el["id"] = (
            f"{modality}_{offset + i}"
        )

        out.append(el)

    return out


def _fallback_explanation(
    policy_clause: Any,
    triggered: List[Any],
) -> str:
    """Generate a minimal fallback explanation."""

    if isinstance(policy_clause, dict):

        clause = policy_clause.get(
            "title",
            "an unmapped policy",
        )

    else:

        clause = "an unmapped policy"

    return (
        f"This content was flagged under {clause}. "
        f"{len(triggered)} element(s) contributed "
        "to the decision."
    )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=(
        "CHEM — "
        "Compositional Harm Emergence Modeling"
    ),

    description=(
        "Policy-aware multimodal content "
        "moderation with explainable decisions."
    ),

    version=load_policy_config()
    .get("meta", {})
    .get("version", "1.0.0"),
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",") if o.strip()],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


_static_dir = os.path.join(
    os.path.dirname(
        os.path.dirname(__file__)
    ),
    "static",
)


if os.path.isdir(_static_dir):

    app.mount(
        "/static",
        StaticFiles(
            directory=_static_dir
        ),
        name="static",
    )


pipeline = CHEMPipeline()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/",
    include_in_schema=False,
)
def root() -> FileResponse:

    return FileResponse(
        os.path.join(
            _static_dir,
            "index.html",
        )
    )


@app.on_event("startup")
def _startup() -> None:
    """Load the pipeline once at server startup."""

    pipeline.load()


@app.get("/health")
def health() -> Dict[str, Any]:
    """Return service/model health."""

    return {
        "status": (
            "ok"
            if pipeline.loaded
            else "degraded"
        ),

        "models_loaded": pipeline.loaded,

        "inference_mode": (
            "deterministic_prototype"
            if pipeline.loaded
            else None
        ),

        "audio_available": (
            pipeline.audio_encoder is not None
            if pipeline.loaded
            else False
        ),

        "video_available": (
            pipeline.video_encoder is not None
            if pipeline.loaded
            else False
        ),

        "error": pipeline.load_error,
    }


@app.post("/analyze")
async def analyze_endpoint(
    text: Optional[str] = Form(
        default=None
    ),

    image: Optional[UploadFile] = File(
        default=None
    ),

    audio: Optional[UploadFile] = File(
        default=None
    ),

    video: Optional[UploadFile] = File(
        default=None
    ),
) -> JSONResponse:
    """Analyse multimodal content."""

    if not pipeline.loaded:

        return JSONResponse(
            status_code=503,

            content={
                "error": (
                    "Models not loaded: "
                    f"{pipeline.load_error}"
                )
            },
        )

    tmp_paths: List[str] = []

    try:

        # --------------------------------------------------------------
        # Image
        # --------------------------------------------------------------

        image_bytes = (
            await image.read()
            if image is not None
            else None
        )

        # --------------------------------------------------------------
        # Audio
        # --------------------------------------------------------------

        audio_path = None

        if audio is not None:

            audio_path = _spool(
                await audio.read(),

                suffix=os.path.splitext(
                    audio.filename or "a.wav"
                )[1],
            )

            tmp_paths.append(
                audio_path
            )

        # --------------------------------------------------------------
        # Video
        # --------------------------------------------------------------

        video_path = None

        if video is not None:

            video_path = _spool(
                await video.read(),

                suffix=os.path.splitext(
                    video.filename or "v.mp4"
                )[1],
            )

            tmp_paths.append(
                video_path
            )

        # --------------------------------------------------------------
        # Run pipeline
        # --------------------------------------------------------------

        result = pipeline.analyze(
            text=text,
            image_bytes=image_bytes,
            audio_path=audio_path,
            video_path=video_path,
        )

        return JSONResponse(
            content=_sanitize(result)
        )

    except Exception as exc:

        traceback.print_exc()

        return JSONResponse(
            status_code=500,

            content={
                "error": (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            },
        )

    finally:

        for path in tmp_paths:

            try:
                os.remove(path)

            except OSError:
                pass


# ---------------------------------------------------------------------------
# JSON sanitization
# ---------------------------------------------------------------------------

def _sanitize(obj: Any) -> Any:
    """Replace NaN/Infinity values with None."""

    import math

    if isinstance(obj, dict):

        return {
            key: _sanitize(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):

        return [
            _sanitize(value)
            for value in obj
        ]

    if isinstance(obj, float):

        if (
            math.isnan(obj)
            or math.isinf(obj)
        ):
            return None

    return obj


# ---------------------------------------------------------------------------
# Temporary-file helper
# ---------------------------------------------------------------------------

def _spool(
    data: bytes,
    suffix: str = "",
) -> str:
    """Write uploaded bytes to a temporary file."""

    fd, path = tempfile.mkstemp(
        suffix=suffix
    )

    with os.fdopen(
        fd,
        "wb",
    ) as handle:

        handle.write(data)

    return path


# ---------------------------------------------------------------------------
# Local execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )