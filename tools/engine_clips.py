#!/usr/bin/env python3
"""Cut, clean and loop engine recordings for Speedo Engine.

Needs ffmpeg (with libmp3lame) and numpy.

  # 1. split a compilation section into one clip per car (gaps of silence)
  python3 tools/engine_clips.py split section.wav --out clips/

  # 2. cut clean clips from the best V10 (times in seconds or mm:ss)
  python3 tools/engine_clips.py cut section.wav --start 1:02 --end 1:09 --name lambo-v10-start
  python3 tools/engine_clips.py cut section.wav --start 1:09 --end 1:14 --name lambo-v10-rev
  python3 tools/engine_clips.py cut section.wav --start 1:14 --end 1:22 --name lambo-v10-accel

  # 3. build a seamless 30-60 s loop from a short steady-rpm stretch
  python3 tools/engine_clips.py loop section.wav --start 1:15.2 --end 1:17.0 --length 45 --name lambo-v10-loop

Every output is written as NAME.wav (48 kHz 16-bit stereo) and NAME.mp3 (320 kbps),
loudness-normalised to the same target so the files sit at a consistent level.
Audio files are git-ignored: game audio is for personal use only, never commit it.
"""
import argparse, csv, os, re, subprocess, sys
import numpy as np

SR = 48000
LUFS = -16.0      # integrated loudness target for every output
TP = -1.0         # true-peak ceiling (dBTP)


def ts(v):
    """'75', '1:15', '1:15.5', '0:01:15' -> seconds."""
    s = 0.0
    for part in str(v).split(":"):
        s = s * 60 + float(part)
    return s


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def read(path, start=None, end=None):
    """Decode to float32 stereo at SR with ffmpeg."""
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"] if start is None else ["-t", f"{end - start:.3f}"]
    cmd += ["-i", path, "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).copy()


def ff(x, af):
    """Run float stereo audio through an ffmpeg filter chain."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", "-",
                        "-af", af, "-f", "f32le", "-"], input=x.astype(np.float32).tobytes(),
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32).reshape(-1, 2).copy()


def lufs(x):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", "-",
                        "-af", "ebur128", "-f", "null", "-"], input=x.astype(np.float32).tobytes(),
                       capture_output=True, text=False, check=True).stderr.decode()
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", p)[-1])


def write(x, name, outdir, declick=True, wrap=False):
    """De-click, normalise to LUFS with one static gain, soft-limit peaks, export WAV + MP3.

    Every step is stateless or wrap-safe, so a seamless loop stays seamless (ffmpeg's
    loudnorm/adeclick ramp in at the start of a file, which would put a dip at the seam).
    """
    os.makedirs(outdir, exist_ok=True)
    n = len(x)
    if declick:
        x = ff(np.concatenate([x, x, x]), "adeclick")[n:2 * n] if wrap else ff(x, "adeclick")[:n]
    x = x * 10 ** ((LUFS - lufs(x)) / 20)
    # Static soft knee above -6 dBFS towards the ceiling: no attack/release state, so loop-safe.
    ceil, knee = 10 ** (TP / 20), 0.5
    a = np.abs(x)
    over = a > knee
    x[over] = np.sign(x[over]) * (knee + (ceil - knee) * np.tanh((a[over] - knee) / (ceil - knee)))
    wav, mp3 = (os.path.join(outdir, f"{name}.{e}") for e in ("wav", "mp3"))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", "-",
                    "-c:a", "pcm_s16le", wav], input=x.astype(np.float32).tobytes(), check=True)
    run(["ffmpeg", "-v", "error", "-y", "-i", wav, "-c:a", "libmp3lame", "-b:a", "320k", mp3])
    print(f"  {wav}\n  {mp3}  ({n / SR:.2f} s, {lufs(x):.1f} LUFS)")
    return wav


def rms_db(x, win):
    mono = x.mean(1)
    n = len(mono) // win
    r = np.sqrt((mono[:n * win].reshape(n, win) ** 2).mean(1) + 1e-12)
    return 20 * np.log10(r)


def trim_silence(x, thresh_db=-50, pad=0.02):
    """Trim quiet lead-in/tail; keep a few ms so the attack isn't clipped."""
    win = int(SR * 0.01)
    db = rms_db(x, win)
    loud = np.where(db > max(thresh_db, db.max() - 45))[0]
    if not len(loud):
        return x
    a = max(0, loud[0] * win - int(pad * SR))
    b = min(len(x), (loud[-1] + 1) * win + int(pad * SR))
    return x[a:b]


def fades(x, fin=0.005, fout=0.03):
    n1, n2 = int(fin * SR), int(fout * SR)
    x[:n1] *= np.linspace(0, 1, n1)[:, None]
    x[-n2:] *= np.linspace(1, 0, n2)[:, None]
    return x


def firing_hz(x):
    """Dominant low-frequency periodicity (engine firing frequency) via autocorrelation."""
    mono = x.mean(1)[: SR * 4]
    mono = mono - mono.mean()
    n = 1 << (len(mono) * 2 - 1).bit_length()
    f = np.fft.rfft(mono, n)
    ac = np.fft.irfft(f * np.conj(f))[: len(mono)]
    lo, hi = int(SR / 900), int(SR / 25)          # 25-900 Hz: idle..redline of 4-12 cyl (rough; octave errors possible)
    if hi >= len(ac):
        return None
    lag = lo + int(np.argmax(ac[lo:hi]))
    return SR / lag


def ncc(x, t):
    """Normalised cross-correlation of template t at every offset in x."""
    n = len(x) + len(t)
    nf = 1 << (n - 1).bit_length()
    num = np.fft.irfft(np.fft.rfft(x, nf) * np.conj(np.fft.rfft(t, nf)), nf)[:len(x) - len(t) + 1]
    e = np.concatenate([[0], np.cumsum(x.astype(np.float64) ** 2)])
    den = np.sqrt((e[len(t):] - e[:-len(t)]) * (t @ t)) + 1e-9
    return num / den


def best_matches(x, t, keep=0.95):
    c = ncc(x, t)
    return np.where(c >= keep * c.max())[0]


# ------------------------------------------------------------------ commands

def cmd_split(a):
    x = read(a.input)
    win = int(SR * 0.05)
    db = rms_db(x, win)
    quiet = db < a.silence_db
    min_gap = int(a.gap / 0.05)
    clips, start, run_q = [], None, 0
    for i, q in enumerate(np.append(quiet, True)):
        if not q:
            if start is None:
                start = i
            run_q = 0
        else:
            run_q += 1
            if start is not None and (run_q >= min_gap or i == len(quiet)):
                end = i - run_q + 1
                if (end - start) * 0.05 >= a.min_len:
                    clips.append((start * 0.05, end * 0.05))
                start = None
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for k, (s, e) in enumerate(clips, 1):
        seg = x[int(s * SR):int(e * SR)]
        hz = firing_hz(seg)
        name = f"clip-{k:02d}"
        path = os.path.join(a.out, name + ".wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", "-",
                        "-c:a", "pcm_s16le", path], input=seg.tobytes(), check=True)
        off = ts(a.offset) if a.offset else 0
        rows.append({"clip": name, "start": f"{s + off:.2f}", "end": f"{e + off:.2f}",
                     "seconds": f"{e - s:.1f}", "firing_hz": f"{hz:.0f}" if hz else "",
                     "rpm_if_v10": f"{hz * 60 / 5:.0f}" if hz else "", "car": "", "game": ""})
        print(f"{name}  {s + off:7.2f}-{e + off:7.2f}s  {e - s:5.1f}s  ~{hz or 0:.0f} Hz")
    with open(os.path.join(a.out, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys() if rows else ["clip"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} clips -> {a.out}/  (fill in car/game in index.csv while listening)")


def cmd_cut(a):
    s, e = ts(a.start), ts(a.end)
    x = trim_silence(read(a.input, s, e))
    write(fades(x), a.name, a.out, declick=not a.no_declick)


def cmd_loop(a):
    s, e = ts(a.start), ts(a.end)
    src = read(a.input, s, e)
    L, xf = int(a.length * SR), int(a.xfade * SR)
    grain = min(len(src), int(a.grain * SR))
    if grain <= 2 * xf:
        sys.exit("source stretch too short for the crossfade; pick a longer steady section or smaller --xfade")
    rng = np.random.default_rng(7)
    mono = src.mean(1)
    # Engine sound is quasi-periodic, so two pieces crossfaded at random phase partly cancel
    # (an audible "whoosh"). Like WSOLA, every join is placed where the waveforms line up.
    up = np.linspace(0, 1, xf)[:, None]
    dn = 1 - up
    search = int(0.5 * SR)
    out = np.zeros((L + search + 4 * xf + grain, 2), np.float32)
    o = int(rng.integers(0, len(src) - grain + 1))
    out[:grain] = src[o:o + grain]
    pos = grain - xf
    while pos < len(out) - grain:
        tmpl = mono[o + grain - xf:o + grain]          # what the previous grain would have continued with
        cand = best_matches(mono[:len(src) - grain + xf], tmpl)
        o = int(rng.choice(cand))                       # random among good matches -> no fixed repeat period
        g = src[o:o + grain]
        out[pos:pos + xf] = out[pos:pos + xf] * dn + g[:xf] * up
        out[pos + xf:pos + grain] = g[xf:]
        pos += grain - xf
    # Wrap: find where, around L, the stream lines up with the loop's head, and fade into it there.
    head = out[xf:2 * xf]
    lo = xf + L - search
    p = lo + int(np.argmax(ncc(out[lo:lo + 2 * search + xf].mean(1), head.mean(1))))
    loop = out[xf:p].copy()
    loop[:xf] = head * up + out[p:p + xf] * dn      # loop[0] == out[p], the sample after loop[-1]
    hz = firing_hz(loop)
    write(loop, a.name, a.out, declick=not a.no_declick, wrap=True)
    if hz:
        print(f"  steady firing frequency ~{hz:.0f} Hz -> base rpm ~{hz * 60 / 5:.0f} if V10 "
              f"(use this as the loop's reference rpm in the app)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split", help="split a compilation into per-car clips")
    s.add_argument("input"); s.add_argument("--out", default="clips")
    s.add_argument("--silence-db", type=float, default=-45, help="below this RMS counts as a gap")
    s.add_argument("--gap", type=float, default=0.35, help="min gap length (s) between cars")
    s.add_argument("--min-len", type=float, default=2.0, help="drop clips shorter than this (s)")
    s.add_argument("--offset", default="0", help="add to printed times, e.g. 0:26 if the file starts there")
    c = sub.add_parser("cut", help="cut + clean one clip")
    c.add_argument("input"); c.add_argument("--start", required=True); c.add_argument("--end", required=True)
    c.add_argument("--name", required=True); c.add_argument("--out", default="out")
    c.add_argument("--no-declick", action="store_true")
    l = sub.add_parser("loop", help="build a seamless loop from a steady stretch")
    l.add_argument("input"); l.add_argument("--start", required=True); l.add_argument("--end", required=True)
    l.add_argument("--name", required=True); l.add_argument("--out", default="out")
    l.add_argument("--length", type=float, default=45, help="loop length in seconds (30-60)")
    l.add_argument("--grain", type=float, default=1.2, help="grain length in seconds")
    l.add_argument("--xfade", type=float, default=0.25, help="crossfade length in seconds")
    l.add_argument("--no-declick", action="store_true")
    a = p.parse_args()
    {"split": cmd_split, "cut": cmd_cut, "loop": cmd_loop}[a.cmd](a)


if __name__ == "__main__":
    main()
