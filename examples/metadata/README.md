# Metadata

The EXIF example reads photography-related tags such as camera make/model, exposure time, aperture, and focal length.

Run from the repository root:

```sh
python -m examples.metadata.accessing_exif_data
```

The selected tags are printed and saved as JSON to `outputs/metadata/rocks_exif.json`. Some sample images do not contain EXIF metadata; in that case the output is an empty JSON object.
