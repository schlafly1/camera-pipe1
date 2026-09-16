# Motorcycle/"Bicycle" class ? eyeballing real images

**Date:** 2026-09-15. Follow-up to notes/detect_min_conf_rejects.md (motorcycle
class ~0.29% accept rate, looks structurally broken independent of confidence).

## Why this isn't a straight "look at rejected snapshots" exercise

pipeline_multi.py only writes a permanent JPEG to snapshots/ after the VLM accepts
a detection (pipeline_multi.py:659-661, after the _vlm_says_absent check at line
638). Rejected events never get a permanent file - their frame lives only in the
transient /tmp/frame_camN_*.jpg ring buffer (900 files/camera, SNAPSHOT_RING_FILES),
long since rotated away for anything in the historical logs. And the reject log
line itself only keeps the first 70 chars of the VLM's answer
(pipeline_multi.py:642).

So there's no way to look at live-rejected motorcycle frames after the fact.
Instead, this used the next best real evidence: eval_vllm_12b_n3.json re-ran
gemma4:12b (Ollama) and vLLM google/gemma-4-12B-it against already-saved
snapshots/*.jpg files whose original detector label was "motorcycle" (i.e.,
historically accepted by whatever VLM was live at the time). 13 distinct
motorcycle-labeled images are in that eval; I opened the actual JPEGs and
compared them to what each model said.

## What the images actually show (8 examples, paths under snapshots/)

| file | what's really in frame | gemma4:12b (Ollama) said | vLLM said | verdict |
|---|---|---|---|---|
| cam6_src5_motorcycle_evt9961.jpg | Two people + a dog walking a path, far away. No bike. | "dark-colored bicycle being ridden" - accepted | correctly said no bicycle, "two people walking a dog" - rejected | Ollama hallucinated |
| cam1_src0_motorcycle_evt1921.jpg | Heavily glitched/corrupted frame over a person at a desk (cam1, office). No bike, and barely a coherent image. | "white and black sportbike" - accepted | correctly described an indoor desk scene - rejected | Ollama hallucinated on a corrupted frame |
| cam5_src4_motorcycle_evt2594.jpg | Person walking a driveway carrying a basket/caddy. No bike. | "dark-colored bicycle is parked... person standing next to stationary bike" - accepted | correctly rejected | Ollama hallucinated |
| cam4_src3_motorcycle_evt208.jpg | Person walking a dog on a wooded path. No bike. | "white motorcycle being ridden to the right" - accepted | correctly said "person walking a dog" - rejected | Ollama hallucinated |
| cam6_src5_motorcycle_evt10.jpg | Person walking with a backpack, dog trailing. No bike, no helmet. | "dark-colored dirt bike... dark helmet... light-colored pants" - accepted | "white and blue bicycle... black helmet... blue shorts" - accepted | Both models hallucinated a bike, and disagree with each other on color/helmet |
| cam4_src3_motorcycle_evt991.jpg | Person walking with an animal on a leash. No bike, no helmet. | "dark-colored bicycle... white top" - accepted | "white helmet... dark-colored bicycle" - accepted | Both models hallucinated a bike |
| cam1_src0_motorcycle_evt213.jpg | Person bent over a desk in the office (cam1). No bike. | correctly rejected | correctly rejected | Both correct |
| cam2_src0_motorcycle_evt4711.jpg + ..._evt4876.jpg | Completely empty, static office room (cam2), no person, no object resembling a bike, ~16 hours apart (2026-06-13 17:42 -> 2026-06-14 09:54) | correctly rejected | correctly rejected | Both correct, but see below |

## The bigger detector-side finding

ls snapshots/ | grep motorcycle turns up ~250+ saved "motorcycle" snapshots, far
more than the 51 counted as "saved" in the 7.25-day log window from the prior
analysis - this directory holds a much longer history. Camera 2 (office) has
runs of consecutive, unbroken evt numbers in the "motorcycle" label - e.g.
evt4672 through evt4876 (205 in a row), evt1972-evt2068, evt5250-evt5272. I
opened the first and last frame of the 4672-4876 run: both show the same empty,
unoccupied office, 16 hours apart, with nothing bike-shaped anywhere. Whatever
is triggering RT-DETR's "Bicycle" class (app label "motorcycle") in this room
is a static-scene artifact - furniture, cabling, a monitor glare pattern,
something - not a bicycle, and it doesn't even require motion or a person
present. This got "accepted" and saved 205+ times in a row historically,
meaning whatever VLM was live back then (likely gemma4:26b, pre-dates these
evals) was rubber-stamping it too.

## Verdict

This is two separate, real problems - not one confidence-threshold problem:

1. Detector-side (RT-DETR "Bicycle"/class 3) is unreliable on these cameras.
   It fires on: pedestrians, dogs, a corrupted/glitched frame, and a static
   empty office room for hours at a stretch. None of this is a
   confidence-threshold issue - notes/detect_min_conf_rejects.md already
   showed 100% reject at every confidence level including 0.84-0.88. The
   class itself needs attention: check whether RT-DETR's bicycle class is
   miscalibrated for this camera set, whether it's worth dropping class 3
   entirely for the office cams (1/2), or investigating what specific
   feature in cam2's empty room keeps re-triggering it.

2. VLM hallucination on ambiguous "is there a bike" prompts - happens on
   both candidate models, not just Ollama. In 2 of 13 sampled images, both
   gemma4:12b (Ollama) and vLLM 12B confidently invented a detailed,
   confident bike/rider description (different colors/helmets from each
   other) for a plain pedestrian photo, and both got "accepted" - meaning
   fabricated vehicle sightings are being saved into ChromaDB right now,
   not just rejected-and-dropped. This is arguably worse than the reject
   problem, since it pollutes search results silently. It also means the
   vLLM cutover won't fix this class - vLLM matched Ollama's mistake in
   both hallucination cases, so this isn't a "vLLM is smarter" story for
   this particular class.

Recommendation: don't spend effort raising DETECT_MIN_CONF for motorcycle
(confirmed unhelpful again here). Instead: (a) separately investigate the
RT-DETR bicycle-class false-trigger source on cam2 (the empty-room repro is
trivial and reproducible - the frames exist right now), and (b) if
bicycle/motorcycle detections matter enough to keep, consider tightening
the VLM prompt for class 3 specifically (e.g. explicitly ask it to describe
visible wheels/frame/handlebars before confirming, or add a stricter
refusal instruction) since two different models both hallucinated the same
way on the same kind of image.

## Caveats

- Sample size is small (13 distinct images, 8 discussed above) - drawn from
  whatever happened to be in the eval's stratified snapshots/ sample, not a
  random draw across the full motorcycle history.
- This used historically accepted snapshots re-scored by new candidate
  models, not a true sample of currently-rejected live events (those frames
  don't exist on disk - see above). The reject-side conclusion ("still
  rejects at all confidences") comes from notes/detect_min_conf_rejects.md's
  log-based analysis, not from images.
- Did not check cam2's other consecutive runs (1972-2068, 5250-5272,
  4672-4876's full middle) frame-by-frame - only first/last of the biggest
  run.
