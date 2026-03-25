import moviepy.editor as mp

# import moviepy.video.fx.all as vfx
import os

# FFmpeg params for Drive/rclone upload compatibility:
# - libx264, preset fast, crf 20: good quality, fast encode
# - pix_fmt yuv420p, profile baseline, level 3.1: max compatibility
# - movflags +faststart: metadata at start for streaming
def _drive_safe_ffmpeg_params(crf: int = 20):
    return [
        "-preset", "fast",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-profile:v", "baseline",
        "-level", "3.1",
        "-movflags", "+faststart",
    ]


def get_drive_safe_videofile_kwargs(fps=None, bitrate=None, crf: int = 20, **overrides):
    """Kwargs for write_videofile that produce Drive/rclone-uploadable MP4s."""
    kwargs = {
        "codec": "libx264",
        "audio": False,
        "verbose": False,
        "logger": None,
        "ffmpeg_params": _drive_safe_ffmpeg_params(crf=crf),
    }
    if fps is not None:
        kwargs["fps"] = fps
    if bitrate is not None:
        kwargs["bitrate"] = bitrate
    kwargs.update(overrides)
    return kwargs


class Video:
    def __init__(self, frame_path: str, fps: float = 20.0, res="high"):
        frame_path = str(frame_path)
        self.fps = fps

        crf = 28 if res == "low" else 20
        self._conf = get_drive_safe_videofile_kwargs(fps=self.fps, crf=crf)

        # Load video
        # video = mp.VideoFileClip(video1_path, audio=False)
        # Load with frames (only frame_*.png to avoid stray files)
        frames = [
            os.path.join(frame_path, x)
            for x in sorted(os.listdir(frame_path))
            if x.startswith("frame_")
        ]
        video = mp.ImageSequenceClip(frames, fps=fps)
        self.video = video
        self.duration = video.duration

    def save(self, out_path):
        out_path = str(out_path)
        self.video.subclip(0, self.duration).write_videofile(out_path, **self._conf)
