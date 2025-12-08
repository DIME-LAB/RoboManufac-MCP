#!/usr/bin/env python3
"""
Generate a PDF with ArUco markers for printing.
Uses the same specifications as the main app (dimensions, border width, etc.).
"""

import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm, inch
from reportlab.lib.colors import black, white
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics import renderPDF
import argparse
from pathlib import Path
import sys
import os

# Add the parent directory to the path to import aruco_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from aruco_annotator.utils.aruco_utils import ArUcoGenerator


class ArUcoPDFGenerator:
    """Generate PDF with ArUco markers for printing."""
    
    def __init__(self, marker_size_mm=30, border_width_percent=20, 
                 page_size='A4', margin_mm=20, markers_per_row=3):
        """
        Initialize the PDF generator.
        
        Args:
            marker_size_mm: Size of each marker in millimeters
            border_width_percent: Border width as percentage (0-50)
            page_size: Page size ('A4' or 'letter')
            margin_mm: Page margins in millimeters
            markers_per_row: Number of markers per row
        """
        self.marker_size_mm = marker_size_mm
        self.border_width_percent = border_width_percent
        self.margin_mm = margin_mm
        self.markers_per_row = markers_per_row
        
        # Set page size
        if page_size.lower() == 'a4':
            self.page_width, self.page_height = A4
        else:
            self.page_width, self.page_height = letter
            
        # Convert to mm for calculations
        self.page_width_mm = self.page_width / mm
        self.page_height_mm = self.page_height / mm
        
        # Initialize ArUco generator
        self.aruco_generator = ArUcoGenerator()
        
        # Calculate marker dimensions - use exact size specified
        self.marker_size_pixels = int(marker_size_mm * 12)  # 12 pixels per mm for high accuracy (304.8 DPI)
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
    
    def generate_marker_image(self, marker_id, dictionary="DICT_4X4_50"):
        """Generate a single ArUco marker image with border."""
        # Generate the ArUco marker
        marker_image = self.aruco_generator.generate_marker(
            dictionary, marker_id, self.marker_size_pixels
        )
        
        # Create a larger image with white border
        total_size = self.marker_size_pixels
        bordered_image = np.ones((total_size, total_size), dtype=np.uint8) * 255  # White background
        
        # Calculate pattern area (center the pattern)
        border_pixels = int(self.border_width_mm * 12)  # 12 pixels per mm
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
        
        return bordered_image
    
    def calculate_layout(self, num_markers):
        """Calculate the layout for the markers on the page."""
        # Calculate available space (account for header)
        header_space_mm = 50  # Space for title, specs, and margin
        available_width = self.page_width_mm - 2 * self.margin_mm
        available_height = self.page_height_mm - 2 * self.margin_mm - header_space_mm
        
        # Calculate spacing
        markers_per_row = min(self.markers_per_row, num_markers)
        num_rows = (num_markers + markers_per_row - 1) // markers_per_row
        
        # Calculate spacing between markers
        if markers_per_row > 1:
            horizontal_spacing = (available_width - markers_per_row * self.marker_size_mm) / (markers_per_row - 1)
        else:
            horizontal_spacing = 0
            
        if num_rows > 1:
            vertical_spacing = (available_height - num_rows * self.marker_size_mm) / (num_rows - 1)
        else:
            vertical_spacing = 0
        
        # Ensure minimum spacing
        horizontal_spacing = max(horizontal_spacing, 5)  # 5mm minimum
        vertical_spacing = max(vertical_spacing, 5)      # 5mm minimum
        
        print(f"📐 Layout calculation:")
        print(f"   Markers per row: {markers_per_row}")
        print(f"   Number of rows: {num_rows}")
        print(f"   Horizontal spacing: {horizontal_spacing:.1f}mm")
        print(f"   Vertical spacing: {vertical_spacing:.1f}mm")
        
        return markers_per_row, num_rows, horizontal_spacing, vertical_spacing
    
    def generate_pdf(self, output_path, marker_ids, dictionary="DICT_4X4_50", 
                    add_black_border=True, black_border_mm=2):
        """
        Generate PDF with ArUco markers.
        
        Args:
            output_path: Path to save the PDF
            marker_ids: List of marker IDs to generate
            dictionary: ArUco dictionary to use
            add_black_border: Whether to add a black border around each marker
            black_border_mm: Width of the black border in mm
        """
        print(f"🎯 Generating PDF with {len(marker_ids)} markers...")
        print(f"   Marker IDs: {marker_ids}")
        print(f"   Dictionary: {dictionary}")
        print(f"   Output: {output_path}")
        
        # Calculate layout
        markers_per_row, num_rows, h_spacing, v_spacing = self.calculate_layout(len(marker_ids))
        
        # Create PDF canvas
        c = canvas.Canvas(output_path, pagesize=(self.page_width, self.page_height))
        
        # Calculate header space (title + specs + margin)
        header_space_mm = 50  # Space for title, specs, and margin
        
        # Generate and place each marker
        for i, marker_id in enumerate(marker_ids):
            # Calculate position - arrange in sequential order from top to bottom
            row = i // markers_per_row
            col = i % markers_per_row
            
            x_mm = self.margin_mm + col * (self.marker_size_mm + h_spacing)
            y_mm = self.margin_mm + header_space_mm + row * (self.marker_size_mm + v_spacing)
            
            # Convert to points (1 mm = 2.834645669 points)
            x_pt = x_mm * mm
            y_pt = (self.page_height_mm - y_mm - self.marker_size_mm) * mm  # Flip Y coordinate
            
            print(f"   Placing marker {marker_id} at ({x_mm:.1f}, {y_mm:.1f}) mm")
            
            # Generate marker image
            marker_image = self.generate_marker_image(marker_id, dictionary)
            
            # Add black border if requested
            if add_black_border:
                border_pixels = int(black_border_mm * 12)  # 12 pixels per mm
                bordered_size = self.marker_size_pixels + 2 * border_pixels
                
                # Create image with black border
                final_image = np.zeros((bordered_size, bordered_size), dtype=np.uint8)
                
                # Place white marker in center
                start_y = border_pixels
                end_y = start_y + self.marker_size_pixels
                start_x = border_pixels
                end_x = start_x + self.marker_size_pixels
                
                final_image[start_y:end_y, start_x:end_x] = marker_image
                
                # Update size for PDF placement
                marker_size_for_pdf = (self.marker_size_mm + 2 * black_border_mm) * mm
            else:
                final_image = marker_image
                marker_size_for_pdf = self.marker_size_mm * mm
            
            # Save marker as temporary image
            temp_path = f"/tmp/aruco_marker_{marker_id}.png"
            cv2.imwrite(temp_path, final_image)
            
            # Add to PDF
            c.drawImage(temp_path, x_pt, y_pt, 
                       width=marker_size_for_pdf, height=marker_size_for_pdf)
            
            # Add marker ID label
            label_x = x_pt + marker_size_for_pdf / 2
            label_y = y_pt - 10  # 10 points below the marker
            
            c.setFont("Helvetica-Bold", 12)
            c.drawCentredString(label_x, label_y, f"ID: {marker_id}")
            
            # Clean up temporary file
            os.remove(temp_path)
        
        # Add title and specifications at the top
        title_y = self.page_height - 20
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(self.page_width / 2, title_y, "ArUco Markers for Printing")
        
        # Add specifications
        specs_y = title_y - 20
        c.setFont("Helvetica", 10)
        specs_text = (f"Dictionary: {dictionary} | "
                     f"Size: {self.marker_size_mm}mm | "
                     f"Border: {self.border_width_percent}% | "
                     f"Black border: {black_border_mm}mm")
        c.drawCentredString(self.page_width / 2, specs_y, specs_text)
        
        # Save PDF
        c.save()
        print(f"✅ PDF generated successfully: {output_path}")


def main():
    """Main function to generate ArUco marker PDF."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    default_output = script_dir / "aruco_markers.pdf"
    
    parser = argparse.ArgumentParser(description="Generate PDF with ArUco markers for printing")
    parser.add_argument("--output", "-o", default=str(default_output), 
                       help=f"Output PDF file path (default: {default_output})")
    parser.add_argument("--size", "-s", type=float, default=25.0,
                       help="Marker size in millimeters (default: 25)")
    parser.add_argument("--border", "-b", type=float, default=10.0,
                       help="Border width percentage (default: 10)")
    parser.add_argument("--ids", "-i", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5],
                       help="Marker IDs to generate (default: 0 1 2 3 4 5)")
    parser.add_argument("--dictionary", "-d", default="DICT_4X4_50",
                       help="ArUco dictionary (default: DICT_4X4_50)")
    parser.add_argument("--page-size", choices=["A4", "letter"], default="A4",
                       help="Page size (default: A4)")
    parser.add_argument("--margin", type=float, default=20.0,
                       help="Page margin in mm (default: 20)")
    parser.add_argument("--per-row", type=int, default=3,
                       help="Markers per row (default: 3)")
    parser.add_argument("--black-border", type=float, default=2.0,
                       help="Black border width in mm (default: 2)")
    parser.add_argument("--no-black-border", action="store_true",
                       help="Disable black border around markers")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate PDF
    generator = ArUcoPDFGenerator(
        marker_size_mm=args.size,
        border_width_percent=args.border,
        page_size=args.page_size,
        margin_mm=args.margin,
        markers_per_row=args.per_row
    )
    
    generator.generate_pdf(
        output_path=str(output_path),
        marker_ids=args.ids,
        dictionary=args.dictionary,
        add_black_border=not args.no_black_border,
        black_border_mm=args.black_border
    )


if __name__ == "__main__":
    main()
