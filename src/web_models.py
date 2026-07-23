"""Validated HTTP contracts for the RegulaAI web adapter."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Jurisdiction = Literal["all", "EU", "US", "UK"]


class AiSystemDescription(BaseModel):
    """The ten project fields, kept explicit for guided analysis."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    system_objective: str = Field(alias="objectif_du_systeme", min_length=1, max_length=800)
    affected_people: str = Field(alias="personnes_concernees", min_length=1, max_length=800)
    data_used: str = Field(alias="donnees_utilisees", min_length=1, max_length=800)
    decisions_produced: str = Field(alias="decisions_produites", min_length=1, max_length=800)
    autonomy_level: str = Field(alias="degre_autonomie", min_length=1, max_length=800)
    business_sector: str = Field(alias="secteur_activite", min_length=1, max_length=800)
    deployment_countries: str = Field(alias="pays_deploiement", min_length=1, max_length=800)
    biometrics: str = Field(alias="presence_biometrie", min_length=1, max_length=800)
    model_provider: str = Field(alias="fournisseur_modele", min_length=1, max_length=800)
    human_intervention: str = Field(alias="intervention_humaine", min_length=1, max_length=800)

    def as_agent_question(self) -> str:
        labels = {
            "system_objective": "objectif du système",
            "affected_people": "personnes concernées",
            "data_used": "données utilisées",
            "decisions_produced": "décisions produites",
            "autonomy_level": "degré d’autonomie",
            "business_sector": "secteur d’activité",
            "deployment_countries": "pays de déploiement",
            "biometrics": "présence de biométrie",
            "model_provider": "fournisseur du modèle",
            "human_intervention": "possibilité d’intervention humaine",
        }
        return "Évaluer ce système IA. " + "; ".join(
            f"{labels[name]}: {getattr(self, name)}" for name in labels
        )


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["guided", "free"] = "free"
    question: str | None = Field(default=None, max_length=4000)
    system: AiSystemDescription | None = None
    jurisdiction: Jurisdiction = "all"
    top_k: int = Field(default=5, ge=1, le=10)
    self_consistency_k: int = Field(default=3, ge=3, le=5)

    @model_validator(mode="after")
    def validate_mode_payload(self) -> "AnalysisRequest":
        if self.mode == "free" and not (self.question or "").strip():
            raise ValueError("Une question est requise en mode libre.")
        if self.mode == "guided" and self.system is None:
            raise ValueError("Les dix champs sont requis en mode guidé.")
        return self

    def agent_question(self) -> str:
        return self.system.as_agent_question() if self.mode == "guided" and self.system else (self.question or "").strip()


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=3, max_length=1000)
    jurisdiction: Jurisdiction = "all"
    top_k: int = Field(default=5, ge=1, le=10)


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


class SecurityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=4000)


class ToolInvokeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arguments: dict[str, Any] = Field(default_factory=dict)
