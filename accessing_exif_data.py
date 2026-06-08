from PIL import Image
from PIL.ExifTags import TAGS


PHOTOGRAPHER_TAGS = [
    # Camera & Lens
    "Make",
    "Model",
    "LensModel",
    "LensSpecification",

    # Exposure
    "ExposureTime",
    "FNumber",
    "ISOSpeedRatings",
    "RecommendedExposureIndex",

    # Focal Length
    "FocalLength",
    "FocalLengthIn35mmFilm",
    "DigitalZoomRatio",

    # Exposure Settings
    "ExposureMode",
    "ExposureProgram",
    "ExposureBiasValue",
    "MeteringMode",

    # Lighting
    "BrightnessValue",
    "LightSource",
    "Flash",

    # Color
    "WhiteBalance",
    "ColorSpace",

    # Processing
    "Software",
    "CustomRendered",
    "Sharpness",
    "Contrast",
    "Saturation",

    # Additional useful settings
    "MaxApertureValue",
    "SceneCaptureType",
    "SensitivityType",
]


def normalize(value):
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        try:
            return float(value)
        except Exception:
            return str(value)

    if isinstance(value, tuple):
        return tuple(normalize(v) for v in value)

    return value


def get_photographer_metadata(image_path):
    img = Image.open(image_path)

    raw_exif = img._getexif()

    if not raw_exif:
        return {}

    exif_data = {}

    for tag_id, value in raw_exif.items():
        tag_name = TAGS.get(tag_id, tag_id)

        if tag_name in PHOTOGRAPHER_TAGS:
            exif_data[tag_name] = normalize(value)

    return exif_data


if __name__ == "__main__":

    image_path = r"C:\\Users\\amitk\\workspace\\Learning_Computer_Vision\\images\\rocks.jpg"

    metadata = get_photographer_metadata(image_path)

    for key in sorted(metadata):
        print(f"{key}: {metadata[key]}")


