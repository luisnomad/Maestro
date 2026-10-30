"""Resolve explicit retake boundary images without losing legacy continuity."""
import os


def resolve_retake_anchors(*, source_start, source_end, user_start=None,
                           user_end=None, clip_frames, strength=1.0,
                           spatial_inpaint=False):
    if clip_frames < 1:
        raise ValueError('Retake must contain at least one frame')
    anchors = []
    for user_path, source_path, index in (
        (user_start, source_start, 0),
        (user_end, source_end, clip_frames - 1),
    ):
        if user_path:
            if not os.path.isfile(user_path):
                raise ValueError(f'Retake anchor image not found: {user_path}')
            path = user_path
        else:
            path = None if spatial_inpaint else source_path
        if path:
            anchors.append((path, index, strength, 'lanczos'))
    return anchors
