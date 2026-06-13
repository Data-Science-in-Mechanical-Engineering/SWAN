import swan.utils

from dataclasses import dataclass, field
import numpy as np
import os
import cv2
import tqdm
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import map_coordinates
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import torch
import cotracker.predictor
from lang_sam import LangSAM
import sys
from contextlib import contextmanager
from PIL import Image



@dataclass
class SegmentationConfig:
    sam2_path: str = "/weights/sam2.1-hiera-large/sam2.1_hiera_large.pt"
    grouding_dino_path: str = "/weights/grounding-dino-base/"

# optained from our bayesian optimization experiments.
OPTIMIZED_DRONE_PARAMS = {
    50: {
        'max_horizon_length': 2, 'overcrowding_distance_factor': 0.4147513694061138, 
        'low_coverage_distance_factor': 1.1124143627567165, 'ff_repulsion_radius_factor': 1.3631738225436325, 
        'ff_boundary_margin_size': 20.0, 'ff_tracking_weight': 0.8, 
        'ff_repulsion_weight': 0.22589539353214488, 'ff_boundary_weight': 0.5670620684355481
    },
    250: {
        'max_horizon_length': 2, 'overcrowding_distance_factor': 0.2, 
        'low_coverage_distance_factor': 1.106129892087347, 'ff_repulsion_radius_factor': 1.5, 
        'ff_boundary_margin_size': 10.622973548133466, 'ff_tracking_weight': 0.8, 
        'ff_repulsion_weight': 0.30563290428868073, 'ff_boundary_weight': 0.4256148613892782
    },
    500: {
        'max_horizon_length': 2, 'overcrowding_distance_factor': 0.2, 
        'low_coverage_distance_factor': 1.1713568399678815, 'ff_repulsion_radius_factor': 1.5, 
        'ff_boundary_margin_size': 9.576279193057687, 'ff_tracking_weight': 0.8, 
        'ff_repulsion_weight': 0.3740588390950938, 'ff_boundary_weight': 0.42249632770494844
    },
    1000: {
        'max_horizon_length': 2, 'overcrowding_distance_factor': 0.2, 
        'low_coverage_distance_factor': 1.164900156883248, 'ff_repulsion_radius_factor': 1.5, 
        'ff_boundary_margin_size': 9.01002343363785, 'ff_tracking_weight': 1.0214873358639214, 
        'ff_repulsion_weight': 0.328727037496562, 'ff_boundary_weight': 0.2717458329705635
    },
    2000: {
        'max_horizon_length': 2, 'overcrowding_distance_factor': 0.29224123261231677, 
        'low_coverage_distance_factor': 1.3823813978137773, 'ff_repulsion_radius_factor': 1.5, 
        'ff_boundary_margin_size': 3.3720347836882407, 'ff_tracking_weight': 0.8243337562876815, 
        'ff_repulsion_weight': 0.40347803028917395, 'ff_boundary_weight': 0.6828900100690122
    }
}

@dataclass
class TrackingConfig:
    segmentation_config: SegmentationConfig = field(default_factory=SegmentationConfig)
    n_simultaneous_tracking_points: int = 50 # n_drones or slightly less
    # These are set from OPTIMIZED_DRONE_PARAMS in __post_init__
    max_horizon_length: int = 2
    ff_tracking_weight: float = 1.0
    ff_repulsion_weight: float = 0.1
    ff_boundary_weight: float = 0.1
    ff_repulsion_radius_factor: float = 1.0
    ff_boundary_margin_size: float = 0.1
    overcrowding_distance_factor: float = 0.5
    low_coverage_distance_factor: float = 1.2
    # always this
    outlier_margin: float = 5.0
    check_collisions_inter_frames: bool = True

    def __post_init__(self):
        # 1. Find the closest n_simultaneous_tracking_points count in our optimization table
        optimized_counts = list(OPTIMIZED_DRONE_PARAMS.keys())
        closest_count = min(optimized_counts, key=lambda x: abs(x - self.n_simultaneous_tracking_points))
        best_params = OPTIMIZED_DRONE_PARAMS[closest_count]

        # 2. Apply the optimized parameters safely
        for param_name, param_value in best_params.items():
            
            # Check what the hardcoded default value is for this specific field
            default_val = self.__dataclass_fields__[param_name].default
            current_val = getattr(self, param_name)
            
            # Only overwrite if the user didn't manually pass a custom value during instantiation
            if current_val == default_val:
                setattr(self, param_name, param_value)


class FilteredStream:
    def __init__(self, original_stream, filter_strings):
        self.original_stream = original_stream
        self.filter_strings = filter_strings

    def write(self, text):
        # Drop the text if it contains any of the spam strings
        if not any(spam_string in text for spam_string in self.filter_strings):
            self.original_stream.write(text)
    def flush(self):
        self.original_stream.flush()

@contextmanager
def mute_spam(spam_texts):
    original_stdout = sys.stdout
    sys.stdout = FilteredStream(original_stdout, spam_texts)
    try:
        yield
    finally:
        # Always restore original stdout!
        sys.stdout = original_stdout

def grounded_sam2_segment_video(video, segmentation_prompt, segmentation_config: SegmentationConfig, cache_file=None):
	assert video is not None, "Video input is 'None'"

	if cache_file is not None and os.path.exists(cache_file):
		print(f"Loading cached segmentation results from {cache_file}")
		segmentation_results = np.load(cache_file, allow_pickle=True)["segmentation_results"]
		return segmentation_results

	model = LangSAM(
		sam_type="sam2.1_hiera_large",
		sam_ckpt_path=segmentation_config.sam2_path,
		gdino_model_ckpt_path=segmentation_config.grouding_dino_path,
		gdino_processor_ckpt_path=segmentation_config.grouding_dino_path
	)

	segmentation_results = []

	with mute_spam(["Predicting 1 masks", "Predicted 1 masks"]):
		for _, frame in enumerate(video):
			image = Image.fromarray(frame)
			segmentation_result = model.predict(images_pil=[image], texts_prompt=[segmentation_prompt])
			segmentation_results.append(segmentation_result)

	if cache_file is not None:
		np.savez_compressed(cache_file, segmentation_results=segmentation_results)
		print(f"Saved segmentation results to {cache_file}, size: {os.path.getsize(cache_file) / (1024*1024):.2f} MB")
	return segmentation_results


def get_combined_mask(segmentation_results, frame_idx):
	masks = segmentation_results[frame_idx][0]["masks"]
	combined_mask = np.any(masks, axis=0)
	return combined_mask

def erode_masks(segmentation_results, frame_idx, erosion_kernel_size=5):
	combined_mask = get_combined_mask(segmentation_results, frame_idx)
	if (
		erosion_kernel_size < 0 or (erosion_kernel_size % 2 == 0 and erosion_kernel_size != 0)
	):  # kernel size should be odd
		raise ValueError(
			f"erosion_kernel_size={erosion_kernel_size}, but should be a positive odd integer"
		)
	if erosion_kernel_size == 0:
		return combined_mask
	round_erosion_kernel = cv2.getStructuringElement(
		cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size)
	)
	return cv2.erode(
		combined_mask.astype(np.uint8), round_erosion_kernel, iterations=1
	).astype(bool)

def sample_farthest_points(points, N):
    """FPS for initialization."""
    # Start with a random point
    indices = [np.random.randint(len(points))]
    # Distance to the set of selected points
    min_dists = np.sum((points - points[indices[0]])**2, axis=1) # squared dist is faster
    
    for _ in range(N - 1):
        # Update distances based on the last added point
        dist_to_last = np.sum((points - points[indices[-1]])**2, axis=1)
        min_dists = np.minimum(min_dists, dist_to_last)
        
        # Pick furthest
        indices.append(np.argmax(min_dists))
        
    return points[indices]

def create_sampling_points_cvt(mask, target_frame, num_points, use_fps_init=True):
    """
    Samples N points using Centroidal Voronoi Tessellation (CVT) via K-Means.
    
    Args:
        mask (np.array): 2D boolean mask.
        target_frame (int): The frame index for which points are being sampled.
        num_points (int): Number of points.
        use_fps_init (bool): If True, uses FPS to find initial seeds (faster convergence).
                             If False, uses k-means++ (standard).
    
    Returns:
        np.array: (N, 3) coordinates of the optimized points.
    """

    # 1. Extract all valid pixel coordinates
    y_idxs, x_idxs = np.where(mask)
    pixel_coords = np.column_stack((y_idxs, x_idxs)).astype(np.float32)
    
    # Safety check
    if len(pixel_coords) <= num_points:
        return pixel_coords

    # 2. Initialize seeds
    init_seeds = 'k-means++'
    if use_fps_init:
        # Run a quick FPS (Strategy 1) to get good starting positions
        # This prevents 'orphan' clusters and speeds up the K-Means
        init_seeds = sample_farthest_points(pixel_coords, num_points)

    # 3. Run K-Means (Discrete Lloyd's Algorithm)
    kmeans = KMeans(n_clusters=num_points, init=init_seeds, n_init=1, max_iter=30, tol=1e-4)
    kmeans.fit(pixel_coords)
    
    # The cluster centers are the CVT points cast to integer pixel coordinates
    centroids = kmeans.cluster_centers_
    sampling_points = np.column_stack((np.full(num_points, target_frame), centroids[:, 1], centroids[:, 0]))
    sampling_points = sampling_points.astype(np.int32)
    return sampling_points

def get_sampling_points(segmentation_results, n_drones, erosion_kernel_size, max_mask_area, mask_areas, frame_idx=0):
    eroded_mask = erode_masks(segmentation_results, frame_idx=frame_idx, erosion_kernel_size=erosion_kernel_size)    
    expected_active_points = int(n_drones * (mask_areas[frame_idx][1] / max_mask_area))
    initial_sampling_points = create_sampling_points_cvt(mask=eroded_mask, target_frame=frame_idx, num_points=expected_active_points)
    return initial_sampling_points

def pick_initial_tracking_frame(segmentation_results, selection="largest_mask_area"):
    # 'largest_mask_area' vs. 'largest_box_area'
    values = []
    for frame_idx, result in enumerate(segmentation_results):
        masks = result[0]["masks"]
        boxes = result[0]["boxes"]
        if len(masks) == 0:
            values.append((frame_idx, 0))
            continue

        mask = get_combined_mask(segmentation_results=segmentation_results, frame_idx=frame_idx)

        area = 0
        if selection == "largest_box_area": # largest box area (total box area if multiple boxes)
            for i in range(len(masks)):
                box = boxes[i]
                x1, y1, x2, y2 = box
                area += (x2 - x1) * (y2 - y1)
        elif selection == "largest_mask_area":
            area = np.sum(mask)
        values.append((frame_idx, area))

    largest_value_frame_idx = max(values, key=lambda x: x[1])[0]
    return largest_value_frame_idx, values

def calc_avg_min_distances(coords):
    """Calculate average minimum distance between all pairs of coordinates."""
    if coords.shape[0] < 2:
        return 0
    distances = squareform(pdist(coords))
    np.fill_diagonal(distances, np.inf)
    min_distances = np.min(distances, axis=1)
    avg_min_distance = np.mean(min_distances)
    return avg_min_distance

def track_points(video, query_points, backward_tracking=True, model=None):
    cotracker_model_path = "/weights/CoTracker3/baseline_offline.pth"

    # Theoretically we could adjust the window length, that is of the time window. See paper for details.
    # We keept at the default which is 60 == 3 seconds at 20 fps.
    
    # prepare our input data for CoTracker
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    sampling_points_tensor = torch.from_numpy(query_points).unsqueeze(0).float()
    
    if model is None:
        model = cotracker.predictor.CoTrackerPredictor(checkpoint=cotracker_model_path, offline=True, window_len=60)
    trajectories, visibilities = model(video=video_tensor, queries=sampling_points_tensor, backward_tracking=backward_tracking)
        
    # squeeze batch dimension
    trajectories = trajectories.squeeze(0).numpy()
    visibilities = visibilities.squeeze(0).numpy()
    return trajectories, visibilities, model

def calc_boundary_forces(coords, mask, boundary_margin_size, signed_distance_field):
    """
    Calculates repulsive forces from the boundaries of the mask using a Signed Distance Field
    and bilinear interpolation to get subpixel forces.
    """
    grad_y, grad_x = np.gradient(signed_distance_field)
    
    # clip coords
    y_coords = np.clip(coords[:, 1], 0, mask.shape[0] - 1)
    x_coords = np.clip(coords[:, 0], 0, mask.shape[1] - 1)
    eval_coords = np.vstack((y_coords, x_coords))

    distances = map_coordinates(signed_distance_field, eval_coords, order=1, mode='nearest') # bilinear interpolation
    gx = map_coordinates(grad_x, eval_coords, order=1, mode='nearest')
    gy = map_coordinates(grad_y, eval_coords, order=1, mode='nearest')

    norms = np.hypot(gx, gy) + 1e-6 # avoid division by zero
    directions = np.stack([gx / norms, gy / norms], axis=1)
    
    F_boundary = np.zeros_like(coords)
    active_mask = distances < boundary_margin_size
    
    if np.any(active_mask):
        # use linear force, 0 if not inside boundary margin size
        magnitudes = np.maximum(0, boundary_margin_size - distances[active_mask])
        F_boundary[active_mask] = directions[active_mask] * magnitudes[:, np.newaxis]
    
    return F_boundary

def calc_repulsion_forces(coords, repulsion_radius):
    """Calculates forces between drones that are closer than the repulsion radius"""

    n_points = len(coords)
    if n_points < 2: 
        return np.zeros_like(coords)
    # compute matrix of pairwise distances
    dist_matrix = cdist(coords, coords)
    diff_matrix = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    directions = diff_matrix / (dist_matrix[:, :, np.newaxis] + 1e-6) # avoid division by zero
    # If we had really many points, we would need to consider a cutoff radius and some special data structure
    # Siehe Vorlesung von Ketan Abhishek (molecular dynamics)
    # tensor with coordinates differences: diff_matrix[i,j] = coords[i] - coords[j]
    F_repel_pairwise = np.maximum(0, repulsion_radius - dist_matrix)
    np.fill_diagonal(F_repel_pairwise, 0) # drones don't repel themselves
    F_repel = np.sum(F_repel_pairwise[:, :, np.newaxis] * directions, axis=1)
    return F_repel

def apply_force_field(prev_coords, current_raw_coords, future_mask, weights, repulsion_radius, boundary_margin_size, signed_distance_field):
    """" Applies a force field to the current raw coordinates from tracking"""

    def evaluate_derivatives(coords):
        v_tracking = current_raw_coords - coords
        if weights['boundary'] > 0:
            F_bound = calc_boundary_forces(coords, future_mask, boundary_margin_size=boundary_margin_size, signed_distance_field=signed_distance_field)
        else:
            F_bound = np.zeros_like(coords)
        if weights['repulsion'] > 0:
            F_repel = calc_repulsion_forces(coords, repulsion_radius)
        else:
            F_repel = np.zeros_like(coords)
        v_total = (v_tracking * weights['tracking'] + F_bound * weights['boundary'] + F_repel * weights['repulsion'])
        return v_total, v_tracking, F_bound, F_repel # latter 3 are for plotting

    # RK4 integration
    k1, v_tracking_plot, F_bound_plot, F_repel_plot = evaluate_derivatives(prev_coords)
    k2, _, _, _ = evaluate_derivatives(prev_coords + 0.5 * k1)
    k3, _, _, _ = evaluate_derivatives(prev_coords + 0.5 * k2)
    k4, _, _, _ = evaluate_derivatives(prev_coords + k3)
    displacement = (k1 + 2*k2 + 2*k3 + k4) / 6
    
    new_coords = prev_coords + displacement

    return new_coords

def get_outliers_sdf(points, signed_distance_field, outlier_margin):
    """
    Returns indices of points that are further outside the mask than the outlier_margin.
    Utilizes the pre-computed Signed Distance Field (SDF).
    """

    if len(points) == 0:
        return []
        
    y_coords = points[:, 1]
    x_coords = points[:, 0]
    eval_coords = np.vstack((y_coords, x_coords))
    
    # Evaluate SDF at point locations using bilinear interpolation.
    # mode='nearest' elegantly handles off-screen coordinates by clamping to the image border.
    point_distances = map_coordinates(signed_distance_field, eval_coords, order=1, mode='nearest')
    
    # A point is an outlier if its distance is more negative than the allowed margin
    outlier_indices = np.where(point_distances < -outlier_margin)[0]
    
    return outlier_indices.tolist()

def remove_overcrowded_points(coords, coords_prev, visibilities_prev, min_distance, check_between_frames, check_frequency=5):
	n_points = len(coords)
	valid = np.ones(n_points, dtype=bool) # points that are not removed yet
	distances = squareform(pdist(coords)) # pdist -> pairwise distances as a list, squareform -> convert to symmetric square matrix.
	np.fill_diagonal(distances, np.inf)  # ignore self-distance

	violations = distances < min_distance

	# do greedy removal
	while True:
		current_violations = violations[valid][:, valid]
		degrees = current_violations.sum(axis=1)
		if degrees.max() == 0:
			break  # no more violations, we are done

		worst_idx_in_valid = np.argmax(
			degrees
		)  # TODO use a better tie-breaking strategy
		valid_indices = np.where(valid)[0]
		worst_idx = valid_indices[worst_idx_in_valid]
		valid[worst_idx] = False

	valid_sum = valid.sum()

	if check_between_frames:
		assert coords_prev is not None and visibilities_prev is not None, "coords_prev and visibilities_prev must be provided when check_between_frames is True"
		# we linearly interpolate the positions of the remaing valid points between the previous and current frame.

		for step in range(1, check_frequency):
			alpha = step / check_frequency # interpolation coeff
			intermediate_coords = (1 - alpha) * coords_prev + alpha * coords

			while True:
				active_and_valid = valid & visibilities_prev # don't interpolate visible or already removed points
				if active_and_valid.sum() < 2:
					break # should never happen

				active_indices = np.where(active_and_valid)[0]
				active_coords = intermediate_coords[active_indices]

				intermediate_distances = squareform(pdist(active_coords))
				np.fill_diagonal(intermediate_distances, np.inf)
				interp_violations = intermediate_distances < min_distance
				degrees = interp_violations.sum(axis=1)
				if degrees.max() == 0:
					break  # no more violations, we are done

				worst_idx_in_active = np.argmax(degrees)
				worst_idx = active_indices[worst_idx_in_active]
				valid[worst_idx] = False
		
		valid_sum_after_interp = valid.sum()
		# print(f"Removed {valid_sum - valid_sum_after_interp} additional points after interpolation check between frames")
						
	removed_indices = np.where(~valid)[0]
	remaining_indices = np.where(valid)[0]
	
	return removed_indices, remaining_indices

def get_neighbors_knn(coords, remaining_indices, removed_indices, k=5):
	if len(removed_indices) == 0:
		return np.array([], dtype=int)
	if len(remaining_indices) == 0:
		raise ValueError("remaining_indices must be non-empty")
	remaining_coords = coords[remaining_indices]
	removed_coords = coords[removed_indices]
	tree = KDTree(remaining_coords)
	_, neighbor_indices = tree.query(removed_coords, k=k)
	neighbor_indices = neighbor_indices.flatten()
	neighbor_indices = np.unique(neighbor_indices)  # remove duplicates
	return remaining_indices[neighbor_indices] 

def find_low_coverage_spots_fps(mask, active_points, num_points, min_distance=10.0):
	"""
	Uses Farthest Point Sampling (FPS) on the Euclidean Distance Transform 
	to fill low coverage voids, avoiding clustering artifacts.
	"""
	if num_points <= 0:
		return np.empty((0, 2), dtype=np.float32), None

	grid = np.ones_like(mask, dtype=bool)
	
	# Safely mask the active points
	valid_y = np.clip(active_points[:, 1].astype(int), 0, mask.shape[0] - 1)
	valid_x = np.clip(active_points[:, 0].astype(int), 0, mask.shape[1] - 1)
	grid[valid_y, valid_x] = False
	
	dist_map = distance_transform_edt(grid)
	dist_map[~mask] = 0
	
	# 1. Isolate the "voids" (areas with low coverage)
	void_mask = dist_map > min_distance
	void_y, void_x = np.where(void_mask)
	
	if len(void_y) == 0:
		return np.empty((0, 2), dtype=np.float32), dist_map

	void_coords = np.column_stack((void_x, void_y)).astype(np.float32)

	# 2. Farthest Point Sampling (FPS)
	actual_num_points = min(num_points, len(void_coords))
	
	# SMART INIT: Start with the point that has the absolute maximum distance
	# This guarantees we target the center of the largest hole first
	first_idx = np.argmax(dist_map[void_y, void_x])
	selected_indices = [first_idx]
	
	if actual_num_points > 1:
		min_dists_sq = np.sum((void_coords - void_coords[first_idx])**2, axis=1)        
		for _ in range(actual_num_points - 1):
			# Pick the pixel that is farthest from all currently selected points
			next_idx = np.argmax(min_dists_sq)
			selected_indices.append(next_idx)
			
			# Update the minimum distances for the remaining pixels
			dist_to_last = np.sum((void_coords - void_coords[next_idx])**2, axis=1)
			min_dists_sq = np.minimum(min_dists_sq, dist_to_last)

			# Early stopping if all remaining points are too close to selected points
			if min_dists_sq.max() < min_distance**2:
				break

	sampled_points = void_coords[selected_indices]

	return sampled_points, dist_map

def constrained_kmeans(mask, fixed_clusters, mobile_clusters, max_iter=30, tol=0.5):
	"""Return the mobile clusters after constrained k-means convergence"""
	if len(mobile_clusters) == 0 or len(fixed_clusters) == 0:
		raise ValueError("fixed_clusters and mobile_clusters must be non-empty")

	y_idxs, x_idxs = np.where(mask)
	pixel_coords = np.column_stack((x_idxs, y_idxs)).astype(
		np.float32
	)  # float because k-means uses subpixel accuracy

	n_fixed = len(fixed_clusters)
	current_mobile_clusters = mobile_clusters.copy()
	for iteration in range(max_iter):
		all_clusters = np.vstack((fixed_clusters, current_mobile_clusters))
		labels = pairwise_distances_argmin(pixel_coords, all_clusters)
		max_shift = 0

		new_mobile_clusters = np.zeros_like(current_mobile_clusters)
		for i in range(len(current_mobile_clusters)):
			cluster_points = pixel_coords[
				labels == n_fixed + i
			]  # get pixels assigned to current mobile cluster
			if len(cluster_points) > 0:
				new_center = cluster_points.mean(axis=0)
			else:
				print(
					f"Warning: Mobile cluster {i} has no points assigned, keeping it fixed."
				)
				new_center = current_mobile_clusters[i]

			shift = np.linalg.norm(new_center - current_mobile_clusters[i])
			max_shift = max(max_shift, shift)
			new_mobile_clusters[i] = new_center

		current_mobile_clusters = new_mobile_clusters

		if max_shift < tol:
			break
	return current_mobile_clusters

def multipass_tracking(
    video, # tensor
    segmentation_results: list,
    n_drones: int,
    max_horizon_length: int,
    ff_tracking_weight: float,
    ff_repulsion_weight: float,
    ff_boundary_weight: float,
    ff_repulsion_radius_factor: float,
    ff_boundary_margin_size: float,
    # erosion_kernel_size: int, Set to 0. Found no significant improvement from erosion in in preliminary experiments.
    overcrowding_distance_factor: float,
    low_coverage_distance_factor: float,
    outlier_margin: float,
    tracking_model = None, # use for batching to not reload CoTracker every time
    check_collisions_inter_frames: bool = True,
):
    """Returns drone trajectories and activities + Statistics"""

    n_frames = video.shape[0]
    n_drones = n_drones
    max_horizon_length = max_horizon_length

    drone_trajectories = np.zeros((n_frames, n_drones, 2))
    drone_activities = np.zeros((n_frames, n_drones), dtype=bool)
    drone_segment_starts = np.zeros((n_frames, n_drones), dtype=bool)

    inactive_drones = list(range(n_drones)) # current list of indices of inactive drones
    active_drones = [] # current list of indices of active drones

    # Step determine derived parameters
    max_mask_frame, mask_areas = pick_initial_tracking_frame(segmentation_results, 'largest_mask_area')
    max_mask_area = mask_areas[max_mask_frame][1]
    erosion_kernel_size = 0

    initial_sampling_points = get_sampling_points(segmentation_results, n_drones, erosion_kernel_size, max_mask_area, mask_areas, frame_idx=0)

    # distances
    sampling_points_max_frame = get_sampling_points(segmentation_results, n_drones, erosion_kernel_size, max_mask_area, mask_areas, frame_idx=max_mask_frame)
    avg_min_distance = calc_avg_min_distances(sampling_points_max_frame) # In case there is a scene where the tracking object comes into frame late
    overcrowding_min_distance = avg_min_distance * overcrowding_distance_factor
    low_coverage_min_distance = avg_min_distance * low_coverage_distance_factor
    
    # force field weights
    force_field_weights = {
        'tracking': ff_tracking_weight,
        'repulsion': ff_repulsion_weight,
        'boundary': ff_boundary_weight
    }

    # prepare loop
    active_drones = inactive_drones[:len(initial_sampling_points)]
    drone_segment_starts[0, active_drones] = True
    inactive_drones = inactive_drones[len(initial_sampling_points):]
    current_active_drone_coords = initial_sampling_points[:, 1:].astype(np.float32) # (t, x, y) -> (x, y)

    frame_idx = 0
    model = tracking_model

    # use tqdm for progress bar
    pbar = tqdm.tqdm(total=n_frames, desc="Tracking Frames")
    while frame_idx < n_frames:
        horizon_length = min(max_horizon_length, n_frames - frame_idx)
        video_chunk = video[frame_idx:frame_idx+horizon_length]

        if len(active_drones) > 0:
#--------------------------------------------------------------------------------------------------------------------------------
# Forward Tracking:
#--------------------------------------------------------------------------------------------------------------------------------
            cotracker_queries = np.hstack((np.zeros((len(active_drones), 1)), current_active_drone_coords))
            cotracker_trajectories, cotracker_visibilites, model = track_points(video_chunk, cotracker_queries, backward_tracking=False, model=model)
#--------------------------------------------------------------------------------------------------------------------------------
# Force Field + Intermediate checks:
#--------------------------------------------------------------------------------------------------------------------------------
            ff_trajectories = np.zeros_like(cotracker_trajectories)
            ff_trajectories[0] = current_active_drone_coords
            ff_visibilities = np.ones_like(cotracker_visibilites)
            
            for t_step in range(1, horizon_length):
                # print(f"Processing frame {frame_idx + t_step} of {n_frames}")
                t_future = frame_idx + t_step
                future_mask = get_combined_mask(segmentation_results, t_future)

                dist_inside_mask = distance_transform_edt(future_mask)
                dist_outside_mask = distance_transform_edt(~future_mask)
                signed_distance_field = dist_inside_mask - dist_outside_mask

                ff_trajectories[t_step] = apply_force_field(
                    prev_coords=ff_trajectories[t_step-1],
                    current_raw_coords=cotracker_trajectories[t_step],
                    future_mask=future_mask,
                    weights=force_field_weights,
                    repulsion_radius=ff_repulsion_radius_factor * avg_min_distance,
                    boundary_margin_size=ff_boundary_margin_size,
                    signed_distance_field=signed_distance_field,
                )

                # ff_trajectories[t_step] = cotracker_trajectories[t_step] # test without force field for now.

                # 1. Outlier check
                frame_outlier_indices = get_outliers_sdf(
                    points=ff_trajectories[t_step],
                    signed_distance_field=signed_distance_field,
                    outlier_margin=outlier_margin
                )

                # 2. Overcrowding check
                frame_overcrowded_indices, _ = remove_overcrowded_points(
                    coords=ff_trajectories[t_step],
                    coords_prev=ff_trajectories[t_step-1],
                    visibilities_prev=ff_visibilities[t_step-1],
                    min_distance=overcrowding_min_distance,
                    check_between_frames=check_collisions_inter_frames
                )

                frame_bad_indices = np.unique(np.concatenate((frame_outlier_indices, frame_overcrowded_indices)).astype(int))
                if len(frame_bad_indices) > 0:
                    ff_visibilities[t_step:, frame_bad_indices] = False

#--------------------------------------------------------------------------------------------------------------------------------
# Bookkeeping + Time advancement to end of horizon:
#--------------------------------------------------------------------------------------------------------------------------------
            chunk_slice = slice(frame_idx, frame_idx+horizon_length)
            drone_trajectories[chunk_slice, active_drones] = ff_trajectories
            drone_activities[chunk_slice, active_drones] = ff_visibilities
            drone_activities[chunk_slice, inactive_drones] = False

        frame_idx += horizon_length - 1
        pbar.update(horizon_length - 1)

        # print(f"Advanced to frame {frame_idx}.") 
        if frame_idx >= n_frames - 1:
            break

#---------------------------------------------------------------------------------------------------------------------------------
# Healing / Relaxation Pass
# --------------------------------------------------------------------------------------------------------------------------------
        eroded_mask = erode_masks(segmentation_results, frame_idx=frame_idx, erosion_kernel_size=erosion_kernel_size)
        expected_active_points = int(n_drones * (mask_areas[frame_idx][1] / max_mask_area))
        
        current_active_drone_coords = drone_trajectories[frame_idx, active_drones]
        is_visible_in_chunk = drone_activities[frame_idx, active_drones]

        visible_indices = np.where(is_visible_in_chunk)[0]
        invisible_indices = np.where(~is_visible_in_chunk)[0]

        # happens often
        if len(invisible_indices) > 0:
            inactive_drones.extend([active_drones[i] for i in invisible_indices])
            active_drones = [active_drones[i] for i in visible_indices]
            current_active_drone_coords = current_active_drone_coords[visible_indices]

        # should happen every time, otherwise we have a problem
        if len(active_drones) > 0:
            # local = indices refer to current_active_drone_coords
            outlier_local_indices = []

            if len(outlier_local_indices) > 0:
                valid_local_indices = np.setdiff1d(np.arange(len(active_drones)), outlier_local_indices)                
                inactive_drones.extend([active_drones[i] for i in outlier_local_indices])
                active_drones = [active_drones[i] for i in valid_local_indices]
                current_active_drone_coords = current_active_drone_coords[valid_local_indices]

            # Overcrowding
            # TODO: remove this, removed local indices is always empty as we already removed overcrowded points in for this frame
            removed_local_indices, remaining_local_indices = remove_overcrowded_points(coords=current_active_drone_coords, coords_prev=None, visibilities_prev=None, min_distance=overcrowding_min_distance, check_between_frames=False)
            
            inactive_drones.extend([active_drones[i] for i in removed_local_indices])
            # for now don't update active_drones

            # TODO: remove this, relaxed local indices is always empty, because
            relaxed_local_indices_old = get_neighbors_knn(current_active_drone_coords, remaining_local_indices, removed_local_indices, k=6)
            relaxed_local_indices = np.intersect1d(relaxed_local_indices_old, remaining_local_indices)
            if (relaxed_local_indices_old != relaxed_local_indices).any():
                print("ERROR: Relaxation local indices mismatch! This should not happen. Check the get_neighbors_knn function for bugs.")
            relaxed_coords = current_active_drone_coords[relaxed_local_indices]
            relaxed_global_indices = [active_drones[i] for i in relaxed_local_indices]
            
            fixed_local_indices = np.setdiff1d(remaining_local_indices, relaxed_local_indices)
            fixed_coords = current_active_drone_coords[fixed_local_indices]
            fixed_global_indices = [active_drones[i] for i in fixed_local_indices]

            valid_coords = current_active_drone_coords[remaining_local_indices]
            
            inserted_point_coords = np.empty((0, 2), dtype=np.float32)
            inserted_point_indices = []
            n_missing = max(0, expected_active_points - len(active_drones))
            if n_missing > 0 and len(inactive_drones) > 0:
                max_to_add = min(n_missing, len(inactive_drones))
                if len(valid_coords) > 0:
                    peaks_xy, _ = find_low_coverage_spots_fps(eroded_mask, valid_coords, num_points=max_to_add, min_distance=low_coverage_min_distance)
                    inserted_point_coords = peaks_xy[:max_to_add]
                else:
                    inserted_point_coords = create_sampling_points_cvt(mask=eroded_mask, target_frame=frame_idx, num_points=max_to_add).astype(np.float32)
                    inserted_point_coords = inserted_point_coords[:, 1:] # (t, x, y) -> (x, y)

                actual_to_add = len(inserted_point_coords)
                inserted_point_indices = inactive_drones[:actual_to_add] 
                inactive_drones = inactive_drones[actual_to_add:] # new drones need to be removed from pool of inactive drones
                drone_segment_starts[frame_idx, inserted_point_indices] = True
            
            if len(relaxed_coords) > 0 and len(inserted_point_coords) > 0:
                mobile_coords = np.vstack((relaxed_coords, inserted_point_coords))
            elif len(relaxed_coords) > 0:
                mobile_coords = relaxed_coords
            elif len(inserted_point_coords) > 0:
                mobile_coords = inserted_point_coords
            else:
                mobile_coords = np.empty((0, 2), dtype=np.float32)
            
            if len(mobile_coords) > 0:
                mobile_coords = constrained_kmeans(
                    eroded_mask, fixed_coords, mobile_coords, max_iter=100, tol=1e-3)
            else:
                mobile_coords = np.empty((0, 2), dtype=np.float32)

            current_active_drone_coords = np.vstack((fixed_coords, mobile_coords))
            active_drones = fixed_global_indices + inserted_point_indices + relaxed_global_indices

    return drone_trajectories, drone_activities, drone_segment_starts


def track_video(video, segmentation_prompt, output_dir, tracking_config: TrackingConfig):
    """Main tracking function. Takes a video and segmentation results. Returns drone trajectories and activities + Statistics"""
    
    segmentation_results = grounded_sam2_segment_video(video, segmentation_prompt, segmentation_config=tracking_config.segmentation_config,
        cache_file=os.path.join(output_dir, "segmentation_results.npz"))

    trajectories, visibilities, segment_starts = multipass_tracking(
        video=video,
        segmentation_results=segmentation_results,
        n_drones=tracking_config.n_simultaneous_tracking_points,
        max_horizon_length=tracking_config.max_horizon_length,
        ff_tracking_weight=tracking_config.ff_tracking_weight,
        ff_repulsion_weight=tracking_config.ff_repulsion_weight,
        ff_boundary_weight=tracking_config.ff_boundary_weight,
        ff_repulsion_radius_factor=tracking_config.ff_repulsion_radius_factor,
        ff_boundary_margin_size=tracking_config.ff_boundary_margin_size,
        overcrowding_distance_factor=tracking_config.overcrowding_distance_factor,
        low_coverage_distance_factor=tracking_config.low_coverage_distance_factor,
        outlier_margin=tracking_config.outlier_margin,
        check_collisions_inter_frames=tracking_config.check_collisions_inter_frames
    )

    np.savez_compressed(os.path.join(output_dir, "trajectories.npz"), trajectories=trajectories)
    np.savez_compressed(os.path.join(output_dir, "visibilities.npz"), visibilities=visibilities)
    np.savez_compressed(os.path.join(output_dir, "segment_starts.npz"), segment_starts=segment_starts)
    print(f"Saving tajectory data to {output_dir}")

    colors = swan.utils.create_colors_for_tracking_video(trajectories, visibilities=visibilities, type='constant')
    swan.utils.create_custom_tracking_video(
        video=video,
        trajectories=trajectories,
        visibilities=visibilities,
        colors=colors,
        path=os.path.join(output_dir, "tracking_visualization.mp4")
    )

    return os.path.join(output_dir, "tracking_visualization.mp4")