#!/usr/bin/env python3
"""
Launch the G4X 3D Reconstruction Web Interface.

Usage:
    python run_web.py [--port PORT] [--host HOST] [--debug]

Example:
    python run_web.py --port 8080
    python run_web.py --host 0.0.0.0 --port 5000 --debug
"""

import argparse
import webbrowser
import threading
import time


def open_browser(url):
    """Open browser after a short delay."""
    time.sleep(1.5)
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(
        description='G4X 3D Reconstruction Web Interface'
    )
    parser.add_argument(
        '--port', type=int, default=5000,
        help='Port to run the server on (default: 5000)'
    )
    parser.add_argument(
        '--host', type=str, default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1, use 0.0.0.0 for all interfaces)'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Run in debug mode'
    )
    parser.add_argument(
        '--no-browser', action='store_true',
        help='Do not open browser automatically'
    )

    args = parser.parse_args()

    # Import app here to avoid import errors if dependencies missing
    try:
        from app import app
    except ImportError as e:
        print(f"Error importing app: {e}")
        print("\nPlease install required dependencies:")
        print("  pip install -r requirements.txt")
        return 1

    url = f"http://{args.host}:{args.port}"

    print("=" * 60)
    print("G4X 3D Reconstruction Web Interface")
    print("=" * 60)
    print(f"\nStarting server at {url}")
    print("\nFeatures:")
    print("  - Upload section data (CSV, Parquet, ZIP)")
    print("  - Configure processing parameters")
    print("  - Interactive 3D visualization")
    print("  - Export results to various formats")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)

    # Open browser in background thread
    if not args.no_browser and args.host in ('127.0.0.1', 'localhost'):
        threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    # Run the Flask app
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
