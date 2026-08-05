"""Split a video into parts at keyframe boundaries, without re-encoding.

Used to manufacture multi-video test material out of a single clip: two halves
of one recording share people, which is what cross-video work needs and what
the available media otherwise does not offer.

Cuts snap to keyframes and the streams are copied, not re-encoded. A cut placed
mid-GOP would leave the next part starting on a frame that references pixels it
no longer has, which decodes as garbage until the following keyframe; snapping
avoids that and costs nothing in quality or time. The clip has B-frames, so
packets are selected and shifted by *decode* timestamp rather than display
order.

The exact boundaries land in a sidecar JSON, because a part's timestamps start
at zero and mapping them back to the source needs the offset.

    python scripts/split_video.py media/test.mp4 --parts 2
"""

import argparse
import bisect
import json
import string
from pathlib import Path

import av


def keyframe_times(path: str) -> list[float]:
    """Decode-order timestamps of every video keyframe."""
    with av.open(path) as container:
        stream = container.streams.video[0]
        return [
            float(packet.dts * stream.time_base)
            for packet in container.demux(stream)
            if packet.dts is not None and packet.is_keyframe
        ]


def plan_cuts(path: str, parts: int) -> tuple[list[float], float]:
    """Cut points for `parts` roughly equal pieces, snapped to keyframes."""
    with av.open(path) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base)

    keyframes = keyframe_times(path)
    if len(keyframes) < parts:
        raise SystemExit(
            f"Only {len(keyframes)} keyframes; cannot make {parts} independently "
            f"decodable parts. Re-encode with a shorter GOP first."
        )

    cuts = []
    for i in range(1, parts):
        target = duration * i / parts
        index = bisect.bisect_left(keyframes, target)
        # Nearest keyframe either side of the ideal cut, never reusing one.
        candidates = [t for t in keyframes[max(0, index - 1): index + 1] if t not in cuts]
        if not candidates:
            raise SystemExit(f"No unused keyframe near {target:.2f}s")
        cuts.append(min(candidates, key=lambda t: abs(t - target)))
    return cuts, duration


def write_part(src: str, dst: str, start: float, end: float | None) -> int:
    """Copy every packet in [start, end) into a new container, rebased to zero."""
    written = 0
    with av.open(src) as inp, av.open(dst, "w") as out:
        streams = [s for s in inp.streams if s.type in ("video", "audio")]
        mapping = {s.index: out.add_stream_from_template(s) for s in streams}

        for packet in inp.demux(streams):
            if packet.dts is None:
                continue  # flush packet, emitted once per stream at the end
            seconds = float(packet.dts * packet.time_base)
            if seconds < start:
                continue
            if end is not None and seconds >= end:
                continue
            offset = int(start / packet.time_base)
            packet.dts -= offset
            if packet.pts is not None:
                packet.pts -= offset
            packet.stream = mapping[packet.stream.index]
            out.mux(packet)
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--parts", type=int, default=2)
    parser.add_argument("--out-dir", default=None, help="defaults to the source directory")
    args = parser.parse_args()

    src = Path(args.video)
    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cuts, duration = plan_cuts(str(src), args.parts)
    bounds = [0.0, *cuts, duration]

    parts = []
    for i, (start, end) in enumerate(zip(bounds, bounds[1:])):
        suffix = string.ascii_lowercase[i]
        dst = out_dir / f"{src.stem}_{suffix}{src.suffix}"
        last = i == args.parts - 1
        count = write_part(str(src), str(dst), start, None if last else end)
        parts.append({
            "path": str(dst).replace("\\", "/"),
            "source_start": round(start, 3),
            "source_end": round(end, 3),
            # Add this to a timestamp inside the part to get the source timestamp.
            "offset": round(start, 3),
            "packets": count,
            "size_mb": round(dst.stat().st_size / 1e6, 2),
        })
        print(f"{dst.name}: {start:.3f}-{end:.3f}s  {count} packets  {parts[-1]['size_mb']} MB")

    sidecar = out_dir / f"{src.stem}_split.json"
    sidecar.write_text(
        json.dumps({"source": str(src).replace("\\", "/"),
                    "duration": round(duration, 3),
                    "parts": parts}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {sidecar}")


if __name__ == "__main__":
    main()
