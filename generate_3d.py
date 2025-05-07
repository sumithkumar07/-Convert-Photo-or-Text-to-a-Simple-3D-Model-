import os
import torch
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import open3d as o3d
import matplotlib.pyplot as plt
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.notebooks import decode_latent_mesh

# Define model cache directory
MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "shap_e_models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

class Model3DGenerator:
    def __init__(self):
        # Initialize MiDaS model
        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas.to(self.device)
        self.midas.eval()
        
        # Initialize Shap-E models with caching
        print("Loading Shap-E models (this may take a while on first run)...")
        self.xm = load_model('transmitter', device=self.device, cache_dir=MODEL_CACHE_DIR)
        self.model = load_model('text300M', device=self.device, cache_dir=MODEL_CACHE_DIR)
        self.diffusion = diffusion_from_config(load_config('diffusion', cache_dir=MODEL_CACHE_DIR))
        print("Models loaded successfully!")

    def remove_background(self, input_path, output_path):
        """Remove background from input image using U²-Net."""
        input_image = Image.open(input_path)
        output = remove(input_image)
        output.save(output_path)
        return output_path

    def estimate_depth(self, image_path):
        """Estimate depth map using MiDaS."""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Prepare input
        input_tensor = torch.from_numpy(img).float().to(self.device)
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Get depth map
        with torch.no_grad():
            depth_map = self.midas(input_tensor)
            depth_map = torch.nn.functional.interpolate(
                depth_map.unsqueeze(1),
                size=img.shape[:2],
                mode='bicubic',
                align_corners=False,
            ).squeeze()
        
        return depth_map.cpu().numpy()

    def generate_mesh_from_depth(self, depth_map, output_path):
        """Generate 3D mesh from depth map using Open3D."""
        # Convert depth map to Open3D format
        depth_image = o3d.geometry.Image(depth_map)
        
        # Create mesh from depth image
        mesh = o3d.geometry.TriangleMesh.create_from_depth_image(
            depth_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
            )
        )
        
        # Save mesh
        o3d.io.write_triangle_mesh(output_path, mesh)
        return mesh

    def generate_from_text(self, prompt, output_path):
        """Generate 3D model from text using Shap-E."""
        print(f"Generating 3D model for prompt: {prompt}")
        # Generate latents
        latents = sample_latents(
            batch_size=1,
            model=self.model,
            diffusion=self.diffusion,
            guidance_scale=15.0,
            model_kwargs=dict(texts=[prompt]),
            progress=True,
            use_fp16=True,
            device=self.device,
            clip_denoised=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0
        )
        
        print("Converting latents to mesh...")
        # Convert latent to mesh
        mesh = decode_latent_mesh(self.xm, latents[0]).tri_mesh()
        mesh.write_obj(output_path)
        print(f"Mesh saved to {output_path}")
        return mesh

    def visualize_mesh(self, mesh, output_path):
        """Visualize 3D mesh using matplotlib."""
        print(f"Generating visualization...")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=triangles,
            cmap='viridis'
        )
        
        plt.savefig(output_path)
        plt.close()
        print(f"Visualization saved to {output_path}")

def main():
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    generator = Model3DGenerator()
    
    # Example usage for image-to-3D
    if os.path.exists("input_image.jpg"):
        print("\nProcessing image-to-3D conversion...")
        # Remove background
        cleaned_image = generator.remove_background(
            "input_image.jpg",
            "output/cleaned_image.png"
        )
        
        # Generate depth map
        depth_map = generator.estimate_depth(cleaned_image)
        
        # Generate and save mesh
        mesh = generator.generate_mesh_from_depth(
            depth_map,
            "output/mesh_from_image.obj"
        )
        
        # Visualize mesh
        generator.visualize_mesh(mesh, "output/visualization_image.png")
    
    # Example usage for text-to-3D
    print("\nProcessing text-to-3D generation...")
    prompt = "A small toy car"
    mesh = generator.generate_from_text(prompt, "output/mesh_from_text.obj")
    generator.visualize_mesh(mesh, "output/visualization_text.png")

if __name__ == "__main__":
    main() 