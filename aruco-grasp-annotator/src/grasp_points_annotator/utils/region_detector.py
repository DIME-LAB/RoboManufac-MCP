"""Adaptive Region Splitting - Detects grasp points from rendered images"""

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def generate_mask_from_image(image_path):
    """
    Generate binary mask from input image.
    Tries alpha channel first, then uses background subtraction.
    """
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Method 1: Try alpha channel
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        alpha = img_array[:, :, 3]
        mask = (alpha > 0).astype(np.uint8) * 255
    
    # Method 2: Background subtraction
    else:
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Sample corners to get background color
        h, w = gray.shape
        corner_samples = [
            gray[0:10, 0:10],
            gray[0:10, w-10:w],
            gray[h-10:h, 0:10],
            gray[h-10:h, w-10:w]
        ]
        bg_value = np.median([np.median(sample) for sample in corner_samples])
        
        # Threshold based on distance from background
        diff = np.abs(gray.astype(float) - bg_value)
        threshold = np.std(diff) * 0.5
        mask = (diff > threshold).astype(np.uint8) * 255
        
        # Clean up mask with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def clean_mask(mask):
    """Clean the mask by keeping only the largest connected component"""
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary)
    if num_labels > 1:
        areas = [(labels == i).sum() for i in range(1, num_labels)]
        largest_idx = 1 + np.argmax(areas)
        mask = ((labels == largest_idx).astype(np.uint8)) * 255
    return mask


def find_contours(mask):
    """Find contours in the mask"""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour


def is_point_on_boundary(x, y, mask):
    """Check if a point is exactly on the boundary of the mask"""
    h, w = mask.shape
    
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    
    if mask[y, x] == 0:
        return False
    
    # Check if any of the 8-connected neighbors is outside the mask
    neighbors = [
        (x-1, y-1), (x, y-1), (x+1, y-1),
        (x-1, y),             (x+1, y),
        (x-1, y+1), (x, y+1), (x+1, y+1)
    ]
    
    has_outside_neighbor = False
    for nx, ny in neighbors:
        if nx < 0 or ny < 0 or nx >= w or ny >= h or mask[ny, nx] == 0:
            has_outside_neighbor = True
            break
    
    return has_outside_neighbor


def extract_boundary_corners(contour, mask):
    """Extract corner points that are exactly on the boundary"""
    epsilon_values = [0.005, 0.01, 0.02, 0.03, 0.05]
    all_corners = []
    
    for epsilon_factor in epsilon_values:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        corners = [(point[0][0], point[0][1]) for point in approx]
        all_corners.extend(corners)
    
    # Remove duplicates and filter points that are not on the boundary
    unique_corners = []
    for corner in all_corners:
        x, y = corner
        
        if is_point_on_boundary(x, y, mask):
            is_duplicate = False
            for existing in unique_corners:
                if abs(x - existing[0]) < 5 and abs(y - existing[1]) < 5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_corners.append(corner)
    
    return unique_corners


def find_mask_boundary(mask):
    """Find the boundary of the mask as a set of line segments"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return []
    
    largest_contour = max(contours, key=cv2.contourArea)
    boundary_lines = []
    
    for i in range(len(largest_contour)):
        p1 = largest_contour[i][0]
        p2 = largest_contour[(i + 1) % len(largest_contour)][0]
        boundary_lines.append((p1[0], p1[1], p2[0], p2[1]))
    
    return boundary_lines


def draw_line_until_boundary(point, direction, mask):
    """Draw a line from a point in a specific direction until it hits the mask boundary"""
    x, y = point
    h, w = mask.shape
    
    if direction == 'up':
        for dy in range(1, y + 1):
            new_y = y - dy
            if new_y < 0 or mask[new_y, x] == 0:
                return (x, new_y + 1)
        return (x, 0)
    
    elif direction == 'down':
        for dy in range(1, h - y):
            new_y = y + dy
            if new_y >= h or mask[new_y, x] == 0:
                return (x, new_y - 1)
        return (x, h - 1)
    
    elif direction == 'left':
        for dx in range(1, x + 1):
            new_x = x - dx
            if new_x < 0 or mask[y, new_x] == 0:
                return (new_x + 1, y)
        return (0, y)
    
    elif direction == 'right':
        for dx in range(1, w - x):
            new_x = x + dx
            if new_x >= w or mask[y, new_x] == 0:
                return (new_x - 1, y)
        return (w - 1, y)
    
    return point


def line_overlaps_boundary(line, boundary_lines, threshold=5):
    """Check if a line overlaps with any boundary line"""
    x1, y1, x2, y2 = line
    
    for boundary_line in boundary_lines:
        bx1, by1, bx2, by2 = boundary_line
        
        if (abs(x1 - bx1) < threshold and abs(y1 - by1) < threshold) or \
           (abs(x1 - bx2) < threshold and abs(y1 - by2) < threshold) or \
           (abs(x2 - bx1) < threshold and abs(y2 - by1) < threshold) or \
           (abs(x2 - bx2) < threshold and abs(y2 - by2) < threshold):
            return True
    
    return False


def create_boundary_aware_lines(corner_points, mask, boundary_lines):
    """Create lines from corner points that respect mask boundaries"""
    all_lines = []
    
    for point in corner_points:
        x, y = point
        
        # Draw lines in all four directions
        directions = ['up', 'down', 'left', 'right']
        
        for direction in directions:
            end_point = draw_line_until_boundary(point, direction, mask)
            line = (x, y, end_point[0], end_point[1])
            
            # Only add if line has some length and doesn't overlap boundary
            if (abs(end_point[0] - x) > 1 or abs(end_point[1] - y) > 1) and not line_overlaps_boundary(line, boundary_lines):
                all_lines.append(line)
    
    # Add boundary lines as valid lines
    all_lines.extend(boundary_lines)
    
    return all_lines


def find_line_intersection(line1, line2):
    """Find intersection point between two lines"""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    
    return None


def find_line_intersections(lines):
    """Find intersections between all lines"""
    intersections = []
    
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines[i+1:], i+1):
            intersection = find_line_intersection(line1, line2)
            if intersection:
                intersections.append(intersection)
    
    return intersections


def is_region_inside_mask(left, top, right, bottom, mask):
    """Check if a region is inside the mask"""
    h, w = mask.shape
    
    if left < 0 or top < 0 or right >= w or bottom >= h:
        return False
    
    center_x = int((left + right) / 2)
    center_y = int((top + bottom) / 2)
    
    if mask[center_y, center_x] == 0:
        return False
    
    # Sample a few points within the region
    sample_points = [
        (int(left + (right - left) * 0.25), int(top + (bottom - top) * 0.25)),
        (int(left + (right - left) * 0.75), int(top + (bottom - top) * 0.25)),
        (int(left + (right - left) * 0.25), int(top + (bottom - top) * 0.75)),
        (int(left + (right - left) * 0.75), int(top + (bottom - top) * 0.75))
    ]
    
    inside_count = 0
    for px, py in sample_points:
        if 0 <= px < w and 0 <= py < h and mask[py, px] > 0:
            inside_count += 1
    
    return inside_count >= 2


def create_regions_from_boundary_aware_lines(lines, intersections, mask):
    """Create regions based on boundary-aware lines and intersections"""
    h, w = mask.shape
    
    # Get all unique x and y coordinates from lines and intersections
    all_x = set()
    all_y = set()
    
    for line in lines:
        all_x.add(line[0])
        all_x.add(line[2])
        all_y.add(line[1])
        all_y.add(line[3])
    
    for intersection in intersections:
        all_x.add(intersection[0])
        all_y.add(intersection[1])
    
    sorted_x = sorted(all_x)
    sorted_y = sorted(all_y)
    
    # Create regions based on grid formed by these coordinates
    regions = []
    region_id = 1
    
    for i in range(len(sorted_x) - 1):
        for j in range(len(sorted_y) - 1):
            left = sorted_x[i]
            right = sorted_x[i + 1]
            top = sorted_y[j]
            bottom = sorted_y[j + 1]
            
            # Check if this region is inside the mask
            if is_region_inside_mask(left, top, right, bottom, mask):
                region = {
                    'id': region_id,
                    'left': left,
                    'right': right,
                    'top': top,
                    'bottom': bottom,
                    'width': right - left,
                    'height': bottom - top,
                    'center_x': (left + right) / 2,
                    'center_y': (top + bottom) / 2
                }
                regions.append(region)
                region_id += 1
    
    return regions


def filter_regions_by_area(regions, min_area_threshold=1000):
    """Filter regions by minimum area threshold and renumber sequentially"""
    filtered_regions = []
    
    # Calculate adaptive threshold based on the largest region
    if regions:
        areas = [region['width'] * region['height'] for region in regions]
        max_area = max(areas)
        
        # For simple shapes, use a much higher threshold to avoid artificial splitting
        if max_area > 100000:  # Large object (like jenga block)
            adaptive_threshold = max_area * 0.1  # Keep only regions > 10% of largest
        else:
            adaptive_threshold = min_area_threshold
        
        for region in regions:
            area = region['width'] * region['height']
            if area >= adaptive_threshold:
                filtered_regions.append(region)
    else:
        filtered_regions = regions
    
    # Renumber regions sequentially starting from 1
    for i, region in enumerate(filtered_regions, 1):
        region['id'] = i
    
    return filtered_regions


def visualize_center_points_only(original_image_path, mask, regions, output_path):
    """Create visualization showing only the center points of each region"""
    # Load original image
    if Path(original_image_path).exists():
        img = cv2.imread(original_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Create visualization from mask
        img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        img[mask > 0] = [0, 255, 0]  # Green for mask
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(img)
    
    # Extract center points
    center_points = []
    for region in regions:
        center_points.append((region['center_x'], region['center_y']))
    
    # Draw center points
    if center_points:
        center_x = [p[0] for p in center_points]
        center_y = [p[1] for p in center_points]
        
        # Draw center points as large red circles
        ax.scatter(center_x, center_y, c='red', s=200, marker='o', 
                  edgecolors='black', linewidth=3, label='Region Centers', zorder=5)
        
        # Add region numbers next to each center point
        for region in regions:
            ax.text(region['center_x'], region['center_y'] - 30, f"R{region['id']}", 
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    ax.set_title(f'Region Center Points ({len(regions)} regions)', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.legend()
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def detect_grasp_points_2d(image_path, min_area_threshold=1000):
    """
    Main function to detect 2D grasp points from a rendered image.
    
    Args:
        image_path: Path to the rendered image
        min_area_threshold: Minimum area threshold for filtering regions
        
    Returns:
        Tuple of (center_points_2d, filtered_regions, mask)
        - center_points_2d: List of (x, y) tuples
        - filtered_regions: List of region dictionaries
        - mask: The generated mask
    """
    # Generate mask from image
    mask = generate_mask_from_image(image_path)
    
    # Clean mask
    mask = clean_mask(mask)
    
    # Find contours
    contour = find_contours(mask)
    if contour is None:
        raise ValueError("No contours found in mask")
    
    # Extract boundary corners
    corner_points = extract_boundary_corners(contour, mask)
    
    # Find mask boundary
    boundary_lines = find_mask_boundary(mask)
    
    # Create boundary-aware lines
    lines = create_boundary_aware_lines(corner_points, mask, boundary_lines)
    
    # Find line intersections
    intersections = find_line_intersections(lines)
    
    # Create regions
    regions = create_regions_from_boundary_aware_lines(lines, intersections, mask)
    
    # Filter regions
    filtered_regions = filter_regions_by_area(regions, min_area_threshold)
    
    # Extract center points as list of (x, y) tuples
    center_points_2d = [(region['center_x'], region['center_y']) 
                       for region in filtered_regions]
    
    return center_points_2d, filtered_regions, mask

