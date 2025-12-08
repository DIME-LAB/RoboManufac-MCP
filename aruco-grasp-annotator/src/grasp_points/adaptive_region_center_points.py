#!/usr/bin/env python3
"""
Adaptive Region Splitting - Center Points Only
Extracts corner points, creates boundary-aware regions, and shows only center points.

This script automatically generates masks from images and:
1. Generates mask from input image (tries alpha channel, then background subtraction)
2. Extracts boundary corner points from the mask
3. Creates boundary-aware region splitting
4. Filters out small regions to keep only meaningful ones
5. Shows only the center points of each region

Usage:
    python adaptive_region_center_points.py [image_path]
    
Arguments:
    image_path: Path to input image (default: images/c_green.png)
    
Examples:
    python adaptive_region_center_points.py
    python adaptive_region_center_points.py images/c_green.png
    python adaptive_region_center_points.py "images/fork_orange v5.png"
"""

import os
import sys
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
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
        print("  Using alpha channel for mask...")
        alpha = img_array[:, :, 3]
        mask = (alpha > 0).astype(np.uint8) * 255
    
    # Method 2: Background subtraction
    else:
        print("  Using background subtraction for mask...")
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
    print(f"  ✓ Mask cleaned")
    return mask

def load_mask(mask_path):
    """Load the mask image"""
    if not os.path.exists(mask_path):
        print(f"Error: Mask file not found - {mask_path}")
        sys.exit(1)
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not load mask - {mask_path}")
        sys.exit(1)
    
    print(f"  ✓ Mask loaded: {mask.shape}")
    return mask

def find_contours(mask):
    """Find contours in the mask"""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("Error: No contours found in mask")
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    print(f"  ✓ Found {len(contours)} contours, using largest with {len(largest_contour)} points")
    
    return largest_contour

def extract_boundary_corners(contour, mask):
    """Extract corner points that are exactly on the boundary"""
    epsilon_values = [0.005, 0.01, 0.02, 0.03, 0.05]
    all_corners = []
    
    for epsilon_factor in epsilon_values:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        corners = [(point[0][0], point[0][1]) for point in approx]
        all_corners.extend(corners)
        print(f"    Epsilon {epsilon_factor}: {len(corners)} points")
    
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
    
    print(f"  ✓ Boundary corner detection found {len(unique_corners)} valid corner points")
    return unique_corners

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
    
    print(f"  ✓ Found {len(boundary_lines)} boundary line segments")
    return boundary_lines

def create_boundary_aware_lines(corner_points, mask, boundary_lines):
    """Create lines from corner points that respect mask boundaries"""
    all_lines = []
    h, w = mask.shape
    
    print(f"  ✓ Creating boundary-aware lines from {len(corner_points)} corner points...")
    
    for i, point in enumerate(corner_points):
        x, y = point
        print(f"    Point {i+1}: ({x}, {y})")
        
        # Draw lines in all four directions
        directions = ['up', 'down', 'left', 'right']
        
        for direction in directions:
            end_point = draw_line_until_boundary(point, direction, mask)
            line = (x, y, end_point[0], end_point[1])
            
            # Only add if line has some length and doesn't overlap boundary
            if (abs(end_point[0] - x) > 1 or abs(end_point[1] - y) > 1) and not line_overlaps_boundary(line, boundary_lines):
                all_lines.append(line)
                print(f"      {direction}: ({x}, {y}) -> ({end_point[0]}, {end_point[1]})")
            else:
                print(f"      {direction}: Skipped (overlaps boundary or too short)")
    
    # Add boundary lines as valid lines
    all_lines.extend(boundary_lines)
    print(f"  ✓ Added {len(boundary_lines)} boundary lines")
    print(f"  ✓ Total lines: {len(all_lines)}")
    
    return all_lines

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

def find_line_intersections(lines):
    """Find intersections between all lines"""
    intersections = []
    
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines[i+1:], i+1):
            intersection = find_line_intersection(line1, line2)
            if intersection:
                intersections.append(intersection)
    
    print(f"  ✓ Found {len(intersections)} line intersections")
    return intersections

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
    
    print(f"  ✓ Using {len(sorted_x)} x-coordinates and {len(sorted_y)} y-coordinates")
    
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
    
    print(f"  ✓ Created {len(regions)} regions inside mask")
    return regions

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
        
        print(f"  ✓ Adaptive threshold: {adaptive_threshold:.0f} (max area: {max_area:.0f})")
        
        for region in regions:
            area = region['width'] * region['height']
            if area >= adaptive_threshold:
                filtered_regions.append(region)
    else:
        filtered_regions = regions
    
    # Renumber regions sequentially starting from 1
    for i, region in enumerate(filtered_regions, 1):
        region['id'] = i
    
    print(f"  ✓ Filtered from {len(regions)} to {len(filtered_regions)} regions")
    print(f"  ✓ Removed {len(regions) - len(filtered_regions)} small regions")
    print(f"  ✓ Renumbered regions sequentially (1-{len(filtered_regions)})")
    
    return filtered_regions

def visualize_center_points_only(original_image_path, mask, regions, output_path):
    """Create visualization showing only the center points of each region"""
    # Load original image
    if os.path.exists(original_image_path):
        img = cv2.imread(original_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Create visualization from mask
        img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        img[mask > 0] = [0, 255, 0]  # Green for mask
        print(f"  Warning: Original image not found, using mask visualization")
    
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
        for i, region in enumerate(regions):
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
    
    print(f"  ✓ Center points visualization saved: {output_path}")

def save_center_points_data(regions, output_path):
    """Save center points data to JSON and text files"""
    # Extract center points
    center_points = []
    for region in regions:
        center_point = {
            'region_id': int(region['id']),
            'center_x': float(region['center_x']),
            'center_y': float(region['center_y']),
            'area': float(region['width'] * region['height'])
        }
        center_points.append(center_point)
    
    # Save as JSON
    json_path = output_path.replace('.png', '_center_points.json')
    with open(json_path, 'w') as f:
        json.dump(center_points, f, indent=2)
    
    # Save as text
    txt_path = output_path.replace('.png', '_center_points.txt')
    with open(txt_path, 'w') as f:
        f.write("Region Center Points\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total regions: {len(regions)}\n")
        f.write("Method: Adaptive region splitting with center point extraction\n\n")
        
        for point in center_points:
            f.write(f"Region {point['region_id']}:\n")
            f.write(f"  Center: ({point['center_x']:.1f}, {point['center_y']:.1f})\n")
            f.write(f"  Area: {point['area']:.0f} pixels\n\n")
    
    print(f"  ✓ Center points data saved: {json_path}")
    print(f"  ✓ Center points data saved: {txt_path}")
    return json_path, txt_path

def main():
    print("Adaptive Region Splitting - Center Points Only")
    print("=" * 50)
    
    # Parse arguments
    image_path = "images/c_green.png"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    # Create outputs and masks directories if they don't exist
    outputs_dir = "outputs"
    masks_dir = "masks"
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    print(f"Input image: {image_path}")
    print(f"Output directory: {outputs_dir}")
    print(f"Masks directory: {masks_dir}\n")
    
    # Step 1: Generate mask from image
    print("Step 1: Generating mask from image...")
    if not os.path.exists(image_path):
        print(f"Error: Image file not found - {image_path}")
        sys.exit(1)
    
    mask = generate_mask_from_image(image_path)
    print(f"  ✓ Mask generated: {mask.shape}")
    
    # Step 1.5: Clean mask
    print("\nStep 1.5: Cleaning mask...")
    mask = clean_mask(mask)
    
    # Step 1.6: Save mask to masks folder
    print("\nStep 1.6: Saving mask...")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_output_path = os.path.join(masks_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_output_path, mask)
    print(f"  ✓ Mask saved: {mask_output_path}")
    
    # Step 2: Find contours
    print("\nStep 2: Finding contours...")
    contour = find_contours(mask)
    if contour is None:
        print("Error: No contours found")
        sys.exit(1)
    
    # Step 3: Extract boundary corner points
    print("\nStep 3: Extracting boundary corner points...")
    corner_points = extract_boundary_corners(contour, mask)
    
    # Step 4: Find mask boundary
    print("\nStep 4: Finding mask boundary...")
    boundary_lines = find_mask_boundary(mask)
    
    # Step 5: Create boundary-aware lines
    print("\nStep 5: Creating boundary-aware lines...")
    lines = create_boundary_aware_lines(corner_points, mask, boundary_lines)
    
    # Step 6: Find line intersections
    print("\nStep 6: Finding line intersections...")
    intersections = find_line_intersections(lines)
    
    # Step 7: Create regions
    print("\nStep 7: Creating regions...")
    regions = create_regions_from_boundary_aware_lines(lines, intersections, mask)
    
    # Step 8: Filter regions
    print("\nStep 8: Filtering regions...")
    filtered_regions = filter_regions_by_area(regions, min_area_threshold=1000)
    
    # Step 9: Create center points visualization
    print("\nStep 9: Creating center points visualization...")
    # Extract base name from image path and create output in outputs directory
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(outputs_dir, f"{base_name}_center_points.png")
    visualize_center_points_only(image_path, mask, filtered_regions, output_path)
    
    # Step 10: Save center points data
    print("\nStep 10: Saving center points data...")
    save_center_points_data(filtered_regions, output_path)
    
    print("\n" + "=" * 50)
    print("✓ Complete!")
    print(f"\nGenerated files:")
    print(f"  - Mask: {mask_output_path}")
    print(f"  - Center points visualization: {output_path}")
    print(f"  - Center points data: {output_path.replace('.png', '_center_points.json')}")
    print(f"  - Center points data: {output_path.replace('.png', '_center_points.txt')}")
    
    # Print final summary
    print(f"\nCenter Points Summary:")
    print(f"  Total regions: {len(filtered_regions)}")
    print(f"  Method: Adaptive region splitting with center point extraction")
    for region in filtered_regions:
        print(f"  Region {region['id']}: Center at ({region['center_x']:.1f}, {region['center_y']:.1f})")

if __name__ == "__main__":
    main()
