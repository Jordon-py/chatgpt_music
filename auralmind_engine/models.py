"""
models.py

Pydantic models for AuralMind MCP server.

ALL MCP tools accept and return these models as JSON contracts.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


# ---------------------------------------------------------
# Base
# ---------------------------------------------------------

class BaseResponse(BaseModel):
    """Base response for all tools."""

    success: bool = Field(..., description="True if operation succeeded")
    message: str = Field(..., description="Human readable message")


# ---------------------------------------------------------
# Presets
# ---------------------------------------------------------

class PresetsResponse(BaseResponse):
    """Response listing available mastering presets."""

    presets: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of preset configurations"
    )


# ---------------------------------------------------------
# Analyze
# ---------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """
    Analyze audio file from URL.

    ChatGPT cannot access local files, so input must be URL.
    """

    file_url: str = Field(
        ...,
        description="Public URL to audio file"
    )


class AnalyzeResponse(BaseResponse):
    """Analysis result."""

    analysis: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------
# Master
# ---------------------------------------------------------

class MasterRequest(BaseModel):
    """
    Request to master track.
    """

    file_url: str = Field(
        ...,
        description="Public URL of audio file"
    )

    reference_url: Optional[str] = Field(
        None,
        description="Optional reference track URL"
    )

    preset: str = Field(
        "trap_modern",
        description="Preset name"
    )

    output_name: str = Field(
        "mastered.wav",
        description="Output filename"
    )


class MasterResponse(BaseResponse):
    """
    Mastering result.
    """

    output_url: Optional[str] = Field(
        None,
        description="Download URL"
    )

    report: Dict[str, Any] = Field(default_factory=dict)