# SatQuery AI v1 scope

## Goal

Demonstrate genuine model inference over one prepared optical before/after
image pair. The prototype is a local research workflow, not a general-purpose
change-detection service.

## Inputs

The prototype accepts exactly two files:

- `before`: the earlier optical image.
- `after`: the later optical image.

For the first experiment, both must be 256 × 256, three-channel, 8-bit RGB
PNG images. They must cover the same extent, have the same ground sampling
distance and projection, and already be accurately co-registered. The caller,
not v1, is responsible for atmospheric correction, orthorectification,
co-registration, resampling, cloud removal, and selecting comparable seasons.

The 256 × 256 constraint preserves the exact training/inference tensor path in
Change-Agent while avoiding its incorrect non-256 resize branch. Supporting
larger scenes by explicit tiling is outside this first experiment.

## Outputs

For every successful pair, v1 produces:

1. An indexed, single-channel 256 × 256 PNG whose pixel values are `0`
   (background/no detected target change), `1` (road change), or `2`
   (building change).
2. A human-readable color mask with a fixed legend.
3. A 256 × 256 overlay of that mask on the `after` image at a documented alpha.
4. A UTF-8 text or JSON caption when the tested MCI caption decoder is enabled;
   otherwise the result explicitly records that captioning is unavailable.
5. A small run record containing input names, model/checkpoint identity, device,
   elapsed time, and output paths.

“Background” does not prove no real-world change occurred; it means the model
did not label the pixel as a road or building change.

## Limitations

- Only road and building changes are represented by the candidate model.
- Inputs are prepared RGB visual imagery, not arbitrary multispectral rasters.
- There is no geospatial reprojection, registration, cloud masking, tiling,
  vectorization, area measurement, or uncertainty calibration in v1.
- Resolution changes, shadows, seasonal differences, sensor differences, and
  imperfect alignment can all create false changes.
- Output remains in 256 × 256 image coordinates; no GeoTIFF metadata is emitted.
- The model was trained on LEVIR-MCI imagery, so performance outside that domain
  is unverified.
- The generated caption is constrained to the supplied 468-token vocabulary
  and is not evidence independent of the mask.
- The upstream licensing inconsistency must be resolved before commercial use
  or redistribution of code/checkpoints.

## Acceptance criteria

The first prototype is accepted when, from a clean project environment:

- One known, aligned 256 × 256 RGB pair passes input validation.
- A real checkpoint forward pass completes without silently substituting image
  differencing, synthetic data, or a mock result.
- The indexed mask contains only `{0, 1, 2}` and has the input dimensions.
- The color mask and overlay are created from the indexed mask, have a visible
  class legend, and do not modify the source files.
- If captioning is enabled, the caption comes from the checkpoint's decoder and
  its recorded vocabulary, with no external language model required.
- The run record identifies CPU or MPS, checkpoint SHA-256, timing, and any MPS
  CPU fallback setting.
- Repeating the same pair with the same software and checkpoint produces the
  same indexed mask.
- A reviewer can visually compare before, after, mask, and overlay and can find
  no hidden dependency on the full LEVIR-MCI dataset.
