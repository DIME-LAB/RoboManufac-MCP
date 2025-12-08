#!/usr/bin/env python3
"""
Main entry point for the Wireframe Exporter application.

This provides a unified interface to access all wireframe export and visualization tools.
"""

import argparse
import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from export_wireframe import main as export_main
from plot_wireframe_example import main as plot_main


def main():
    """Main entry point for the wireframe exporter application."""
    parser = argparse.ArgumentParser(
        description="Wireframe Exporter - Export and visualize 3D mesh wireframe data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch GUI application
  python main.py gui
  
  # Export wireframe data
  python main.py export model.obj --format json --output model_wireframe.json
  
  # Plot wireframe data
  python main.py plot model_wireframe.json
  
  # Get help for specific commands
  python main.py export --help
  python main.py plot --help
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # GUI subcommand
    gui_parser = subparsers.add_parser('gui', help='Launch GUI application')
    
    # Export subcommand
    export_parser = subparsers.add_parser('export', help='Export wireframe data from 3D mesh')
    export_parser.add_argument('input_file', help='Input 3D mesh file (STL, OBJ, PLY, etc.)')
    export_parser.add_argument('--format', '-f', choices=['json', 'csv', 'numpy', 'ply', 'obj'], 
                              default='json', help='Export format (default: json)')
    export_parser.add_argument('--output', '-o', help='Output file path (default: auto-generated)')
    export_parser.add_argument('--info', '-i', action='store_true', 
                              help='Show mesh information only')
    
    # Plot subcommand
    plot_parser = subparsers.add_parser('plot', help='Plot wireframe data')
    plot_parser.add_argument('wireframe_file', help='Exported wireframe file (JSON, NumPy, or CSV)')
    
    args = parser.parse_args()
    
    if args.command == 'gui':
        # Launch GUI application
        from gui.main_window import main as gui_main
        return gui_main()
    
    elif args.command == 'export':
        # Set up arguments for export_wireframe.py
        sys.argv = ['export_wireframe.py', args.input_file]
        if args.format:
            sys.argv.extend(['--format', args.format])
        if args.output:
            sys.argv.extend(['--output', args.output])
        if args.info:
            sys.argv.append('--info')
        
        return export_main()
    
    elif args.command == 'plot':
        # Set up arguments for plot_wireframe_example.py
        sys.argv = ['plot_wireframe_example.py', args.wireframe_file]
        
        return plot_main()
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())
