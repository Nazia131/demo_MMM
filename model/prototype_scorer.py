from typing import Any, Dict, List, Optional
import re

from utils.config import get_coefficient, get_policy_clause


class PrototypeHarmScorer:
    """
    CHEM deterministic prototype inference layer.

    Version 4:
    - Conservative image-only evidence
    - Context-aware text evidence
    - Stronger multimodal agreement
    - Prevents weak CLIP predictions from dominating
    - Keeps graph/policy components intact
    """

    def __init__(self):
        self.threshold = float(
            get_coefficient("final_harm_threshold")
        )

        self.cross_modal_bonus = float(
            get_coefficient("cross_modal_bonus")
        )

        self.min_image_score = float(
            get_coefficient("cross_modal_min_image_score")
        )

        self.min_audio_score = float(
            get_coefficient("cross_modal_min_audio_score")
        )

    @staticmethod
    def _safe_score(value: Any) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        if value != value:
            return 0.0

        return max(0.0, min(1.0, value))

    def _get_nodes(
        self,
        graph: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        return graph.get("nodes", []) or []

    def _get_edges(
        self,
        graph: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        return graph.get("edges", []) or []

    def _harmful_nodes(
        self,
        graph: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        return [
            node
            for node in self._get_nodes(graph)
            if bool(node.get("harmful_signal", False))
        ]

    def _categories(
        self,
        nodes: List[Dict[str, Any]],
    ) -> List[str]:

        categories = []

        for node in nodes:

            category = node.get(
                "policy_category"
            )

            if (
                category
                and category not in categories
            ):
                categories.append(category)

        return categories

    # ============================================================
    # CONTEXTUAL TEXT ANALYSIS
    # ============================================================

    def _text_pattern_evidence(
        self,
        graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Neutral by design.

        This used to hold ~300 lines of hand-written regexes that were written
        from individual Hateful Memes dev examples (e.g. the "white privilege
        discount" and "knowing white people ... baby father" captions), which
        contaminates any dev evaluation. It also never saw the real caption (the
        node ``text`` field was never populated). Caption-level hate detection is
        now done by the learned fusion model (model/hm_fusion.py), not rules.
        """
        return {"score": 0.0, "categories": [], "patterns": []}

    # ============================================================
    # PRIMARY CATEGORY
    # ============================================================

    def _primary_category(
        self,
        harmful_nodes: List[Dict[str, Any]],
        modality_scores: Dict[str, float],
        text_evidence: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Optional[str]:

        if text_evidence:

            categories = text_evidence.get(
                "categories",
                [],
            )

            if categories:
                return categories[0]

        categories = self._categories(
            harmful_nodes
        )

        if categories:
            return categories[0]

        return None

    # ============================================================
    # POLICY
    # ============================================================

    def _policy_clause(
        self,
        category: Optional[str],
    ) -> Optional[Dict[str, Any]]:

        if not category:
            return None

        try:
            return get_policy_clause(
                category
            )
        except Exception:
            return None

    # ============================================================
    # PATTERNS
    # ============================================================

    def _detect_patterns(
        self,
        harmful_nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        harmful_modalities: List[str],
        text_evidence: Optional[
            Dict[str, Any]
        ] = None,
    ) -> List[str]:

        patterns = []

        if text_evidence:

            patterns.extend(
                text_evidence.get(
                    "patterns",
                    [],
                )
            )

        for node in harmful_nodes:

            category = node.get(
                "policy_category"
            )

            label = str(
                node.get("label") or ""
            ).lower()

            if category:
                patterns.append(
                    category.upper()
                )

            if (
                "weapon" in label
                or "gun" in label
                or "knife" in label
            ):

                patterns.append(
                    "WEAPON_RELATED_CONTENT"
                )

            if "threat" in label:

                patterns.append(
                    "THREATENING_LANGUAGE"
                )

            if (
                "hate" in label
                or "extrem" in label
            ):

                patterns.append(
                    "HATE_OR_EXTREMIST_CONTENT"
                )

        harmful_pattern_edges = [
            edge
            for edge in edges
            if edge.get("edge_type_name")
            == "harmful_pattern"
        ]

        if harmful_pattern_edges:

            patterns.append(
                "HARMFUL_GRAPH_PATTERN"
            )

        if len(harmful_modalities) >= 2:

            patterns.append(
                "CROSS_MODAL_ESCALATION"
            )

        return list(
            dict.fromkeys(patterns)
        )

    # ============================================================
    # MAIN SCORER
    # ============================================================

    def score(
        self,
        graph: Dict[str, Any],
        modality_scores: Optional[
            Dict[str, float]
        ] = None,
    ) -> Dict[str, Any]:

        modality_scores = (
            modality_scores or {}
        )

        normalized = {
            key: self._safe_score(value)
            for key, value
            in modality_scores.items()
            if value is not None
        }

        harmful_nodes = (
            self._harmful_nodes(graph)
        )

        edges = self._get_edges(graph)

        text_evidence = (
            self._text_pattern_evidence(
                graph
            )
        )

        text_score = normalized.get(
            "text",
            0.0,
        )

        image_score = normalized.get(
            "image",
            0.0,
        )

        audio_score = normalized.get(
            "audio",
            0.0,
        )

        video_score = normalized.get(
            "video",
            0.0,
        )

        explicit_text = (
            text_evidence["score"]
        )

        # ========================================================
        # EVIDENCE MODEL
        # ========================================================

        # Generic Toxic-BERT text scores are useful, but they are
        # not trusted at full strength because Hateful Memes
        # contains sarcastic/insulting language that is not
        # necessarily hateful.
        #
        # Explicit contextual patterns are trusted strongly.

        if explicit_text >= 0.55:

            text_evidence_score = explicit_text

        else:

            text_evidence_score = (
                text_score * 0.35
            )

        # --------------------------------------------------------
        # IMAGE
        # --------------------------------------------------------

        # CLIP visual concepts are intentionally conservative.
        #
        # A moderate visual score such as 0.35-0.60 is NOT enough
        # to independently classify a meme as harmful.
        #
        # Only exceptionally strong visual evidence gets to act
        # independently.

        if image_score >= 0.80:

            image_evidence_score = (
                image_score * 0.65
            )

        elif image_score >= 0.60:

            image_evidence_score = (
                image_score * 0.40
            )

        else:

            image_evidence_score = (
                image_score * 0.15
            )

        # --------------------------------------------------------
        # AUDIO / VIDEO
        # --------------------------------------------------------

        audio_evidence_score = (
            audio_score
            if audio_score >= 0.55
            else audio_score * 0.35
        )

        video_evidence_score = (
            video_score
            if video_score >= 0.55
            else video_score * 0.35
        )

        # ========================================================
        # BASE SCORE
        # ========================================================

        base_score = max(
            text_evidence_score,
            image_evidence_score,
            audio_evidence_score,
            video_evidence_score,
        )

        # ========================================================
        # STRONG TEXT + IMAGE AGREEMENT
        # ========================================================

        multimodal_bonus = 0.0

        strong_text = (
            text_evidence_score >= 0.55
        )

        meaningful_image = (
            image_score >= 0.30
        )

        if (
            strong_text
            and meaningful_image
        ):

            multimodal_bonus = 0.10

        # ========================================================
        # EXPLICIT TEXT + STRONG IMAGE
        # ========================================================

        if (
            explicit_text >= 0.75
            and image_score >= 0.45
        ):

            multimodal_bonus = 0.15

        # ========================================================
        # CROSS-MODAL AUDIO / VIDEO
        # ========================================================

        meaningful_modalities = 0

        if strong_text:
            meaningful_modalities += 1

        if image_score >= 0.65:
            meaningful_modalities += 1

        if audio_score >= 0.55:
            meaningful_modalities += 1

        if video_score >= 0.55:
            meaningful_modalities += 1

        cross_modal_flag = (
            meaningful_modalities >= 2
        )

        if cross_modal_flag:

            base_score += min(
                self.cross_modal_bonus,
                0.10,
            )

        base_score += (
            multimodal_bonus
        )

        # ========================================================
        # GRAPH EVIDENCE
        # ========================================================

        harmful_pattern_edges = [
            edge
            for edge in edges
            if edge.get("edge_type_name")
            == "harmful_pattern"
        ]

        # Graph evidence is supportive, not dominant.

        if harmful_pattern_edges:

            base_score += 0.03

        # ========================================================
        # FINAL SCORE
        # ========================================================

        final_score = min(
            1.0,
            max(
                0.0,
                base_score,
            ),
        )

        verdict = (
            "HARMFUL"
            if final_score >= self.threshold
            else "SAFE"
        )

        harmful_modalities = []

        if (
            explicit_text >= 0.55
            or text_score >= 0.75
        ):

            harmful_modalities.append(
                "text"
            )

        # IMPORTANT:
        # Ordinary image scores below 0.65 do not independently
        # count as a harmful modality.

        if image_score >= 0.65:

            harmful_modalities.append(
                "image"
            )

        if audio_score >= 0.55:

            harmful_modalities.append(
                "audio"
            )

        if video_score >= 0.55:

            harmful_modalities.append(
                "video"
            )

        category = (
            self._primary_category(
                harmful_nodes,
                normalized,
                text_evidence,
            )
        )

        policy_clause = (
            self._policy_clause(
                category
            )
        )

        triggered_patterns = (
            self._detect_patterns(
                harmful_nodes,
                edges,
                harmful_modalities,
                text_evidence,
            )
        )

        explanation_parts = []

        if text_evidence["patterns"]:

            explanation_parts.append(
                "Contextual text-pattern "
                "evidence detected."
            )

        if harmful_modalities:

            explanation_parts.append(
                "Meaningful harmful evidence "
                "detected in: "
                + ", ".join(
                    harmful_modalities
                )
            )

        if (
            image_score > 0
            and image_score < 0.65
        ):

            explanation_parts.append(
                "Visual evidence was treated "
                "as supporting rather than "
                "decisive evidence."
            )

        if harmful_pattern_edges:

            explanation_parts.append(
                "The content composition graph "
                "contains a harmful-pattern "
                "relationship."
            )

        if cross_modal_flag:

            explanation_parts.append(
                "Multiple meaningful modalities "
                "reinforce the signal."
            )

        if not explanation_parts:

            explanation_parts.append(
                "No strong deterministic "
                "harmful pattern was detected."
            )

        explanation = " ".join(
            explanation_parts
        )

        return {

            "harm_score": round(
                final_score,
                4,
            ),

            "verdict": verdict,

            "matched_pattern": (
                triggered_patterns[0]
                if triggered_patterns
                else None
            ),

            "policy_clause": (
                policy_clause
            ),

            "policy_clause_id": (
                policy_clause.get(
                    "clause_id"
                )
                if isinstance(
                    policy_clause,
                    dict,
                )
                else None
            ),

            "triggered_elements": (
                harmful_nodes
            ),

            "triggered_patterns": (
                triggered_patterns
            ),

            "attention_weights": {},

            "modality_scores": normalized,

            "cross_modal_flag": (
                cross_modal_flag
            ),

            "harmful_modalities": (
                harmful_modalities
            ),

            "text_pattern_score": round(
                explicit_text,
                4,
            ),

            "text_pattern_categories": (
                text_evidence[
                    "categories"
                ]
            ),

            "explanation": explanation,
        }