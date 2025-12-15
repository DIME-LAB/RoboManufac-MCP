from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent

README = (ROOT / "README.md").read_text(encoding="utf-8")


setup(
    name="aruco-grasp-annotator",
    version="0.1.0",
    description="3D CAD annotation tool for placing ArUco markers on objects",
    long_description=README,
    long_description_content_type="text/markdown",
    author="ArUco Annotator",
    author_email="user@example.com",
    python_requires=">=3.9,<3.13",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "open3d>=0.17.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "trimesh>=4.0.0",
        "matplotlib>=3.7.0",
        "python-json-logger>=2.0.0",
        "opencv-contrib-python>=4.11.0.86",
        "reportlab>=4.0.0",
        "fastapi>=0.118.0",
        "uvicorn>=0.37.0",
        "python-multipart>=0.0.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "aruco-annotator=aruco_annotator.main:main",
            "assembly-app=assembly_app.main:main",
            "grasp-points-annotator=grasp_points_annotator.main:main",
            "symmetry-exporter=symmetry_exporter.main:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    include_package_data=True,
)
