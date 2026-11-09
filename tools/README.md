# engine_clips.py

Turns a recorded engine-sound section into clean clips and a seamless driving loop
for Speedo Engine. Usage is in the script's docstring (`python3 tools/engine_clips.py -h`).

- **split**: cuts a compilation into one clip per car at the silent gaps and writes
  `index.csv` with times, a rough firing frequency, and blank `car`/`game` columns to fill in by ear.
- **cut**: trims silence, de-clicks, normalises to -16 LUFS, then exports WAV + MP3.
- **loop**: grows a 30–60 s loop from a short steady-rpm stretch by joining random grains
  at phase-aligned points, with a phase-aligned crossfade at the wrap. It prints the loop's base rpm.

Use the **WAV** loop in the app. MP3 adds encoder padding, so some browsers leave a tiny gap at the loop point.

## Pitching the loop with speed

The loop was recorded at a base rpm `rpm0` (printed by `loop`). Each frame:

    rate = clamp(rpm / rpm0, 0.5, 2.0)
    src.playbackRate.setTargetAtTime(rate, ctx.currentTime, 0.05)

- `rpm` comes from the app's existing gear model (`rpmAt(speed, gear)`), so the pitch
  follows gear changes. Shifts drop it and each gear pulls it back up, the way the real car sounds.
- Past about ×1.6 the sound gets thin. For a wider range, record two or three loops (for example
  low, mid and high rpm) and crossfade between neighbours with equal-power gains,
  each playing at `rpm / rpmN`.
- Throttle sets loudness and tone, not pitch: volume rises with load, and a low-pass opens with
  throttle (for example 1.5 kHz coasting to 8 kHz flat out). The `accel` clip can be layered in briefly on hard launches.
