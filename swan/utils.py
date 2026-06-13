import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import tempfile

def create_colors_for_tracking_video(trajectories, visibilities, type, **kwargs):
    drone_colors = np.zeros((trajectories.shape[0], trajectories.shape[1], 3)) # (n_frames, n_drones, 3)

    if type == "first_x" or type == "first_y":
        dim_idx = 0 if type == "first_x" else 1
        cmap = plt.get_cmap('rainbow')
        val_min = np.nanmin(trajectories[:, :, dim_idx])
        val_max = np.nanmax(trajectories[:, :, dim_idx])

        for drone_idx in range(trajectories.shape[1]):
            active_frames = np.where(visibilities[:, drone_idx])[0]
            if len(active_frames) > 0:
                first_t = active_frames[0]
                first_val = trajectories[first_t, drone_idx, dim_idx]
                # Normalize the value to [0, 1] for the colormap
                norm_val = (first_val - val_min) / (val_max - val_min + 1e-6)
                drone_colors[:,drone_idx] = np.array(cmap(norm_val)[:3]) * 255
            else:
                drone_colors[:,drone_idx] = [255, 255, 255] # Fallback for completely inactive
    elif type == "constant":
        color = kwargs.get("color", (0, 255, 255)) # Default to cyan if no color provided
        drone_colors[:] = color
    elif type == "video_default":
        # sample color from the video
        video = kwargs.get("video")
        for frame_idx in range(trajectories.shape[0]):
            for drone_idx in range(trajectories.shape[1]):
                if not visibilities[frame_idx, drone_idx]:
                    continue
                x, y = int(trajectories[frame_idx, drone_idx, 0]), int(trajectories[frame_idx, drone_idx, 1])
                x = np.clip(x, 0, video.shape[2]-1)
                y = np.clip(y, 0, video.shape[1]-1)
                drone_colors[frame_idx, drone_idx] = video[frame_idx, y, x]
    else:    
        raise ValueError(f"Unknown color type: {type}")

    return drone_colors.astype(np.uint8)


def create_custom_tracking_video(video, trajectories, visibilities, colors, video_resolution=None, fps=18, path=None, trace_length=0, radius=3, background_color=(0, 0, 0)):
    """
    Renders tracking points onto a video or black background using fast OpenCV drawing.
    """
    T, N, _ = trajectories.shape
    if video_resolution is not None and video is None:
        H, W = video_resolution
    elif video is not None and video_resolution is None:
        _, H, W, _ = video.shape
    else:
        raise ValueError("Must provide either video or video_resolution (if using a single color background), but not both.")

    # Setup the output file
    if path is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = temp_file.name
    else:
        output_path = path

    # render video
    writer = imageio.get_writer(output_path, fps=fps)
    for t in range(T):
        if video is None:
            frame = np.full((H, W, 3), background_color, dtype=np.uint8)
        else:
            frame = video[t].copy()

        # 1. Draw Traces (if requested)
        if trace_length > 0:
            start_trace = max(0, t - trace_length)
            for i in range(N):
                # Skip tracing if currently invisible
                if not visibilities[t, i]: 
                    continue
                for s in range(start_trace, t):
                    if visibilities[s, i] and visibilities[s+1, i]:
                        pt1 = (int(trajectories[s, i, 0]), int(trajectories[s, i, 1]))
                        pt2 = (int(trajectories[s+1, i, 0]), int(trajectories[s+1, i, 1]))
                        cv2.line(frame, pt1, pt2, tuple(colors[s, i].tolist()), 1, lineType=cv2.LINE_AA)
        # 2. Draw Current Points on top
        for i in range(N):
            if visibilities[t, i]:
                x, y = int(trajectories[t, i, 0]), int(trajectories[t, i, 1])
                cv2.circle(frame, (x, y), radius, tuple(colors[t, i].tolist()), -1, lineType=cv2.LINE_AA)
        writer.append_data(frame)
    writer.close()
    return output_path