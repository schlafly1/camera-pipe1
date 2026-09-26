#!/usr/bin/env python3
"""Offline test for pipeline_multi._vlm_says_absent (VLM "object not there"
detection). Touches nothing live: imports pipeline_multi with LOG_DIR pointed
away from logs/ and only calls the pure string matcher.

    LOG_DIR=/tmp/vlm_selftest .venv/bin/python3 tools/test_vlm_absent.py
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("LOG_DIR", "/tmp/vlm_selftest")

from pipeline_multi import _vlm_says_absent  # noqa: E402

PERSON, CAR, MOTO = 2, 0, 3

# (class_id, text) the VLM uses when the detected object is NOT in the frame.
ABSENT = [
    (PERSON, "The image does not contain any people to describe."),
    (PERSON, "The image provided is a night-vision security camera still that "
             "does not contain any people. The scene shows a gate."),
    (PERSON, "The image does not contain a person to describe."),
    (PERSON, "The provided image does not contain any visible people to describe."),
    (PERSON, "This image does not contain any humans."),
    (PERSON, "The image does not appear to contain any people."),
    (PERSON, "The frame doesn\u2019t show a person, only a fence."),
    (PERSON, "There are no people in this image."),
    (PERSON, "There are no clearly visible people in this image."),
    (PERSON, "There is no clearly identifiable person visible in this image."),
    (PERSON, "There are no persons visible."),
    (PERSON, "No one is visible in the frame."),
    (PERSON, "There is no one in the image, just an empty path."),
    (PERSON, "Nobody is present in the scene."),
    (PERSON, "There isn't anyone in the picture."),
    (PERSON, "The person is not visible in this image."),
    (PERSON, "People are not present in this frame."),
    (PERSON, "I cannot see any person in this image."),
    (PERSON, "I can't see anyone in the frame."),
    (PERSON, "I don't see a person here; it is a dark street."),
    (PERSON, "It is not possible to describe a person; I could not find any people."),
    (PERSON, "The scene is empty and no pedestrians are present."),
    (PERSON, "No human figures can be seen in the infrared image."),
    (PERSON, "Unable to identify a person in this image."),
    (PERSON, "There is no sign of any people on the sidewalk."),
    (PERSON, "No person is visible in the frame."),
    (PERSON, "The image contains no\npeople at all."),
    (PERSON, "The image shows no visible person, only a gate."),
    (PERSON, "It is not possible to describe a person in this image as no "
             "clearly identifiable individual is visible."),
    (PERSON, "A person is not visible in the image; it shows an empty chair."),
    (PERSON, "The image shows an empty room with no people present."),
    (PERSON, "There are no people, but a bench is visible."),
    (PERSON, "The image does not contain any people, but shows an empty gate."),
    (PERSON, "The sidewalk is empty, with no pedestrians."),
    (CAR,    "There are no vehicles, but the driveway is lit."),
    (PERSON, "It is impossible to describe a person because the scene is blurry."),
    (CAR,    "There are no vehicles in this image."),
    (CAR,    "The image does not show any car, only a driveway."),
    (CAR,    "This is not a vehicle; it is a shadow on the pavement."),
    (MOTO,   "There is no motorcycle or bicycle in the image."),
    (MOTO,   "The image doesn't contain a bike."),
]

# Real descriptions (object present) that must NOT be flagged.
PRESENT = [
    (PERSON, "A person is visible walking along the path toward the right."),
    (PERSON, "An adult male in a dark jacket. There are no other people "
             "besides the man in the frame."),
    (PERSON, "No one else is visible; the woman is walking a dog."),
    (PERSON, "A young woman with long hair; nobody else is around."),
    (PERSON, "An adult walking left; there are no pedestrians other than him."),
    (PERSON, "The man has no backpack and no hat."),
    (PERSON, "He is carrying no personal items and wears dark sneakers."),
    (PERSON, "The person's face is not visible due to the hood."),
    (PERSON, "Due to the low resolution, it is impossible to determine the "
             "person's age or gender. They wear a light top."),
    (PERSON, "The individual is wearing a dark jacket and is not carrying anything."),
    (PERSON, "It is not possible to determine the person's age, gender, or hair "
             "color as they are mostly obscured by a metal fence."),
    (PERSON, "Someone in a grey hoodie is standing by the gate."),
    (PERSON, "Based on the image provided, there is no clearly visible person to "
             "describe. A person is partially visible in the foreground, sitting "
             "in a chair and facing away from the camera."),
    # From the Claude review of the first draft:
    (PERSON, "The image does not show the person clearly, but they wear a red jacket."),
    (PERSON, "The face of the person is not visible, but they wear a dark hoodie."),
    (PERSON, "The lower half of the person is not visible behind the car."),
    (PERSON, "A lone man walks down the empty street with no people around him."),
    (PERSON, "There are no people, other than him, on the sidewalk."),
    (PERSON, "No one, except the cyclist, is on the path."),
    (PERSON, "It is impossible to identify the person's age or gender."),
    (PERSON, "No one but the guard is on the path."),
    (CAR,    "A white sedan with no visible damage heading left."),
    (CAR,    "A dark SUV; the car has no license plate visible."),
    (CAR,    "A red pickup truck with no cargo in the bed."),
    (MOTO,   "A red scooter parked with no rider."),
    (MOTO,   "A cyclist with no helmet riding a black bicycle."),
    (MOTO,   "A sport motorcycle with no passenger, heading north."),
]


def main():
    fails = []
    for cls, text in ABSENT:
        if not _vlm_says_absent(text, cls):
            fails.append(("should be ABSENT ", cls, text))
    for cls, text in PRESENT:
        if _vlm_says_absent(text, cls):
            fails.append(("should be PRESENT", cls, text))
    for kind, cls, text in fails:
        print(f"FAIL {kind} class={cls}: {text}")
    total = len(ABSENT) + len(PRESENT)
    print(f"{total - len(fails)}/{total} passed "
          f"({len(ABSENT)} absent, {len(PRESENT)} present cases)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())