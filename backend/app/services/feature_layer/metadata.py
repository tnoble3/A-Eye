from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import ExifTags, Image


CAMERA_TAG_NAMES = {
    "Make",
    "Model",
    "LensMake",
    "LensModel",
    "BodySerialNumber",
}
SOFTWARE_TAG_NAMES = {
    "Software",
    "ProcessingSoftware",
    "HostComputer",
}
KNOWN_SOFTWARE_MARKERS = (
    "photoshop",
    "adobe",
    "lightroom",
    "gimp",
    "canva",
    "stable diffusion",
    "midjourney",
    "dall-e",
    "firefly",
)


@dataclass(slots=True)
class MetadataFeatures:
    #These fields stay numeric and lightweight so they can feed a shallow classifier later while still supporting user-facing explanations now.
    exif_tag_count: int
    camera_tag_count: int
    missing_exif_score: float
    missing_camera_data_score: float
    software_marker_score: float
    has_exif: bool
    has_camera_data: bool
    software_tag_value: str | None


def _extract_tagged_exif(image: Image.Image) -> dict[str, object]:
    exif = image.getexif()
    if not exif:
        return {}
    return {
        str(ExifTags.TAGS.get(tag_id, tag_id)): value
        for tag_id, value in exif.items()
        if value not in (None, "")
    }


def _software_marker_score(software_value: str | None) -> float:
    if not software_value:
        return 0.0

    lowered = software_value.lower()
    if any(marker in lowered for marker in KNOWN_SOFTWARE_MARKERS):
        return 1.0
    return 0.6


def extract_metadata_features(image: Image.Image) -> MetadataFeatures:
    tagged_exif = _extract_tagged_exif(image)
    info_values = {str(key).lower(): str(value) for key, value in image.info.items()}

    #camera provenance is useful because authentic camera captures often keep
    #at least some EXIF trail, while screenshots, edited assets, and many generated images do not.
    has_exif = bool(tagged_exif)
    camera_tag_count = sum(1 for tag_name in CAMERA_TAG_NAMES if tagged_exif.get(tag_name))
    has_camera_data = camera_tag_count > 0
    software_tag_value = None
    for tag_name in SOFTWARE_TAG_NAMES:
        if tagged_exif.get(tag_name):
            software_tag_value = str(tagged_exif[tag_name])
            break
    if software_tag_value is None and info_values.get("software"):
        software_tag_value = info_values["software"]

    missing_exif_score = 0.0 if has_exif else 1.0
    missing_camera_data_score = 0.0 if has_camera_data else 1.0
    software_marker_score = _software_marker_score(software_tag_value)

    return MetadataFeatures(
        exif_tag_count=len(tagged_exif),
        camera_tag_count=camera_tag_count,
        missing_exif_score=missing_exif_score,
        missing_camera_data_score=missing_camera_data_score,
        software_marker_score=software_marker_score,
        has_exif=has_exif,
        has_camera_data=has_camera_data,
        software_tag_value=software_tag_value,
    )
