"""Read selected photography metadata from an image's EXIF tags."""

import json

from PIL import Image
from PIL.ExifTags import TAGS

from examples.common import image_path, output_path


PHOTOGRAPHER_TAGS = {
    "Make",
    "Model",
    "LensModel",
    "LensSpecification",
    "ExposureTime",
    "FNumber",
    "ISOSpeedRatings",
    "RecommendedExposureIndex",
    "FocalLength",
    "FocalLengthIn35mmFilm",
    "DigitalZoomRatio",
    "ExposureMode",
    "ExposureProgram",
    "ExposureBiasValue",
    "MeteringMode",
    "BrightnessValue",
    "LightSource",
    "Flash",
    "WhiteBalance",
    "ColorSpace",
    "Software",
    "CustomRendered",
    "Sharpness",
    "Contrast",
    "Saturation",
    "MaxApertureValue",
    "SceneCaptureType",
    "SensitivityType",
}


def normalize(value):
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        try:
            return float(value)
        except (TypeError, ValueError, ZeroDivisionError):
            return str(value)
    if isinstance(value, tuple):
        return tuple(normalize(item) for item in value)
    return value


def get_photographer_metadata(filename: str) -> dict[str, object]:
    with Image.open(image_path(filename)) as image:
        raw_exif = image.getexif()

    return {
        TAGS.get(tag_id, tag_id): normalize(value)
        for tag_id, value in raw_exif.items()
        if TAGS.get(tag_id, tag_id) in PHOTOGRAPHER_TAGS
    }


def main() -> None:
    metadata = get_photographer_metadata("rocks.jpg")
    report = json.dumps(metadata, indent=2, default=str)
    destination = output_path("metadata/rocks_exif.json")
    destination.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"Saved metadata: {destination}")


if __name__ == "__main__":
    main()
