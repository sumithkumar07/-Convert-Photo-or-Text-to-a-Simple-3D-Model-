# 3D Model Generation Pipeline

This project provides a comprehensive pipeline for generating 3D models from both text descriptions and images using state-of-the-art open-source models.

## Features

- Text-to-3D generation using Shap-E
- Image-to-3D conversion using MiDaS depth estimation
- Background removal using U²-Net
- 3D mesh generation and visualization
- Support for both CPU and GPU processing

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)
- 8GB+ RAM

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Image to 3D

1. Place your input image in the project root directory as `input_image.jpg`
2. Run the script:
```bash
python generate_3d.py
```
3. Find the generated files in the `output` directory:
   - `cleaned_image.png`: Image with background removed
   - `mesh_from_image.obj`: Generated 3D mesh
   - `visualization_image.png`: 3D visualization

### Text to 3D

The script automatically generates a 3D model from the text prompt "A small toy car". To modify the prompt, edit the `prompt` variable in the `main()` function of `generate_3d.py`.

Output files:
- `mesh_from_text.obj`: Generated 3D mesh
- `visualization_text.png`: 3D visualization

## Models Used

1. **Shap-E**: Fast and lightweight text-to-3D model
2. **MiDaS**: State-of-the-art monocular depth estimation
3. **U²-Net**: Precise background removal
4. **Open3D**: 3D mesh processing and visualization

## Performance

- Text-to-3D generation: ~15 seconds on GPU
- Image-to-3D conversion: ~5 seconds for depth estimation
- Background removal: ~1-2 seconds

## Output Format

All 3D models are saved in the standard .obj format, compatible with:
- Blender
- Unity
- 3D printers
- Most 3D viewers

## Notes

- For best results with image-to-3D, use images with clear object boundaries
- Text-to-3D works best with simple, concrete objects
- GPU acceleration is recommended for faster processing
- The generated meshes may require post-processing for optimal results

## License

This project uses open-source models and is released under the MIT License. 