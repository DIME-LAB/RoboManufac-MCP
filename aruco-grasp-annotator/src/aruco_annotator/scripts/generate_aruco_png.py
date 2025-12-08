#!/usr/bin/env python3
"""
Generate individual PNG files for ArUco markers.
Uses the same specifications as the PDF generator (dimensions, border width, etc.).
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import sys
import os

# Simple ArUco generator without external dependencies
class SimpleArUcoGenerator:
    """Simple ArUco generator for PNG creation."""
    
    # Available ArUco dictionaries
    ARUCO_DICTS = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    }
    
    def __init__(self):
        self.detector = cv2.aruco.ArucoDetector()
    
    def generate_marker(self, dictionary: str, marker_id: int, size_pixels: int = 200) -> np.ndarray:
        """Generate an ArUco marker image."""
        if dictionary not in self.ARUCO_DICTS:
            raise ValueError(f"Unknown dictionary: {dictionary}")
        
        # Get the dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICTS[dictionary])
        
        # Generate marker
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_pixels)
        
        return marker_image


class ArUcoPNGGenerator:
    """Generate individual PNG files for ArUco markers."""
    
    def __init__(self, marker_size_mm=21, border_width_percent=5, 
                 black_border_mm=2, dpi=304.8):
        """
        Initialize the PNG generator.
        
        Args:
            marker_size_mm: Size of each marker in millimeters
            border_width_percent: Border width as percentage (0-50)
            black_border_mm: Width of the black border in mm
            dpi: Dots per inch for high resolution output
        """
        self.marker_size_mm = marker_size_mm
        self.border_width_percent = border_width_percent
        self.black_border_mm = black_border_mm
        self.dpi = dpi
        
        # Initialize ArUco generator
        self.aruco_generator = SimpleArUcoGenerator()
        
        # Calculate marker dimensions - use exact size specified
        # Convert mm to pixels using DPI (1 inch = 25.4 mm)
        pixels_per_mm = dpi / 25.4
        self.marker_size_pixels = int(marker_size_mm * pixels_per_mm)
        self.border_width_mm = marker_size_mm * (border_width_percent / 100)
        self.pattern_size_mm = marker_size_mm - 2 * self.border_width_mm
        
        # TODO: Change to conventional approach where pattern_size_mm is the inner pattern area
        # and white border is added OUTSIDE (not inside) the pattern size
        # TODO: Export border dimensions as absolute values in meters instead of percentages
        
        print(f"📏 Marker specifications:")
        print(f"   Total size: {marker_size_mm}mm")
        print(f"   Border width: {border_width_percent}% ({self.border_width_mm:.1f}mm)")
        print(f"   Pattern area: {self.pattern_size_mm:.1f}mm")
        print(f"   Resolution: {self.marker_size_pixels}x{self.marker_size_pixels} pixels")
        print(f"   DPI: {dpi}")
    
    def generate_marker_image(self, marker_id, dictionary="DICT_4X4_50", add_black_border=True):
        """Generate a single ArUco marker image with border."""
        # Generate the ArUco marker
        marker_image = self.aruco_generator.generate_marker(
            dictionary, marker_id, self.marker_size_pixels
        )
        
        # Create a larger image with white border
        total_size = self.marker_size_pixels
        bordered_image = np.ones((total_size, total_size), dtype=np.uint8) * 255  # White background
        
        # Calculate pattern area (center the pattern)
        pixels_per_mm = self.dpi / 25.4
        border_pixels = int(self.border_width_mm * pixels_per_mm)
        pattern_size = total_size - 2 * border_pixels
        
        # Resize the marker to fit the pattern area
        if pattern_size > 0:
            resized_marker = cv2.resize(marker_image, (pattern_size, pattern_size), 
                                      interpolation=cv2.INTER_NEAREST)
            
            # Place the resized marker in the center
            start_y = border_pixels
            end_y = start_y + pattern_size
            start_x = border_pixels
            end_x = start_x + pattern_size
            
            bordered_image[start_y:end_y, start_x:end_x] = resized_marker
        
        # Add black border if requested
        if add_black_border:
            border_pixels = int(self.black_border_mm * pixels_per_mm)
            bordered_size = self.marker_size_pixels + 2 * border_pixels
            
            # Create image with black border
            final_image = np.zeros((bordered_size, bordered_size), dtype=np.uint8)
            
            # Place white marker in center
            start_y = border_pixels
            end_y = start_y + self.marker_size_pixels
            start_x = border_pixels
            end_x = start_x + self.marker_size_pixels
            
            final_image[start_y:end_y, start_x:end_x] = bordered_image
        else:
            final_image = bordered_image
        
        return final_image
    
    def generate_png_files(self, output_dir, marker_ids, dictionary="DICT_4X4_50", 
                          add_black_border=True, filename_prefix="aruco_marker"):
        """
        Generate individual PNG files for ArUco markers.
        
        Args:
            output_dir: Directory to save the PNG files
            marker_ids: List of marker IDs to generate
            dictionary: ArUco dictionary to use
            add_black_border: Whether to add a black border around each marker
            filename_prefix: Prefix for the generated files
        """
        print(f"🎯 Generating PNG files for {len(marker_ids)} markers...")
        print(f"   Marker IDs: {marker_ids}")
        print(f"   Dictionary: {dictionary}")
        print(f"   Output directory: {output_dir}")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate each marker
        for marker_id in marker_ids:
            print(f"   Generating marker {marker_id}...")
            
            # Generate marker image
            marker_image = self.generate_marker_image(marker_id, dictionary, add_black_border)
            
            # Create filename
            filename = f"{filename_prefix}_{marker_id:03d}.png"
            filepath = output_path / filename
            
            # Save the image with high DPI metadata
            cv2.imwrite(str(filepath), marker_image)
            
            print(f"   ✅ Saved: {filepath}")
        
        print(f"✅ All {len(marker_ids)} PNG files generated successfully in {output_dir}")


def main():
    """Main function to generate ArUco marker PNG files."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    # Save to data/aruco/pngs folder (project root, not src/)
    default_output = script_dir.parent.parent.parent / "data" / "aruco" / "pngs"
    
    parser = argparse.ArgumentParser(description="Generate individual PNG files for ArUco markers")
    parser.add_argument("--output", "-o", default=str(default_output), 
                       help=f"Output directory for PNG files (default: {default_output})")
    parser.add_argument("--size", "-s", type=float, default=21.0,
                       help="Marker size in millimeters (default: 21)")
    parser.add_argument("--border", "-b", type=float, default=5.0,
                       help="Border width percentage (default: 5)")
    parser.add_argument("--ids", "-i", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5],
                       help="Marker IDs to generate (default: 0 1 2 3 4 5)")
    parser.add_argument("--dictionary", "-d", default="DICT_4X4_50",
                       help="ArUco dictionary (default: DICT_4X4_50)")
    parser.add_argument("--black-border", type=float, default=2.0,
                       help="Black border width in mm (default: 2)")
    parser.add_argument("--no-black-border", action="store_true",
                       help="Disable black border around markers")
    parser.add_argument("--dpi", type=float, default=304.8,
                       help="DPI for high resolution output (default: 304.8)")
    parser.add_argument("--prefix", default="aruco_marker",
                       help="Filename prefix (default: aruco_marker)")
    
    args = parser.parse_args()
    
    # Generate PNG files
    generator = ArUcoPNGGenerator(
        marker_size_mm=args.size,
        border_width_percent=args.border,
        black_border_mm=args.black_border,
        dpi=args.dpi
    )
    
    generator.generate_png_files(
        output_dir=args.output,
        marker_ids=args.ids,
        dictionary=args.dictionary,
        add_black_border=not args.no_black_border,
        filename_prefix=args.prefix
    )


if __name__ == "__main__":
    main()
