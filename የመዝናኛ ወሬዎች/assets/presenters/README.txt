Put a different presenter photo here for each day you want to rotate through
(a different suit, or a different but similarly-posed person) and it will be
used automatically - one per day, cycling through whatever is here.

DO NOT drop photos in directly. Each photo needs its own landmarks file (where
the mouth/eyes/head are), which the animation reads to know where to move
things. Generate that with:

    python tools/add_presenter.py path/to/photo.jpg tuesday-navy-suit

This detects the eyes in the new photo, works out how it differs from the
reference photo (assets/journalist.png) in position/rotation/zoom, and applies
that same difference to every landmark. It then writes:

    assets/presenters/tuesday-navy-suit.png
    assets/presenters/tuesday-navy-suit.json
    assets/presenters/tuesday-navy-suit_check.png   <- LOOK AT THIS ONE FIRST

The _check.png has the detected face/mouth/eyes/torso drawn on top of your
photo. If they line up with the actual face, it's good to use, and it will
join the rotation automatically (delete the _check.png file whenever you like,
it's just for you to inspect - it's not read by the app). If they DON'T line
up, delete the three files and try a photo framed more like the reference one:
front-facing, shoulders up, similar distance from the camera, similar lighting.
This only works well for similarly-framed headshots - it is not a full
face-tracking model.

EASIER OPTION: if you just want a different colour suit on the SAME person,
skip this tool entirely. Edit assets/journalist.png directly (change only the
suit/tie colour, keep the face and crop pixel-identical - a photo editor or an
AI image-edit tool can do this), save it into assets/presenters/ with its own
name, and copy assets/face.json next to it under the same name. Since nothing
about the face or framing changed, the existing landmarks still line up
exactly - no detection needed, and it can never be misaligned.

If this folder has no valid pairs, the app falls back to the single
config.json "presenter_image" + assets/face.json, exactly as before.
