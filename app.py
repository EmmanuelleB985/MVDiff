"""
MVDiff interactive demo for multi-view generation and 3D reconstruction
"""

import json
import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
from PIL import Image, ImageDraw, ImageFont

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for professional styling
CUSTOM_CSS = """
.container {
    max-width: 1200px;
    margin: auto;
    padding: 20px;
}
.header {
    text-align: center;
    padding: 30px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 30px;
}
.metric-card {
    background: white;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 10px;
}
.status-badge {
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: bold;
}
.success { background: #10b981; color: white; }
.processing { background: #3b82f6; color: white; }
.error { background: #ef4444; color: white; }
"""

# JavaScript for enhanced interactivity
CUSTOM_JS = """
function() {
    // Add smooth animations
    document.querySelectorAll('.gradio-container').forEach(el => {
        el.style.animation = 'fadeIn 0.5s ease-in';
    });
}
"""


class MVDiffWebApp:
    """Professional Gradio application for MVDiff"""

    def __init__(self, checkpoint_path: Optional[str] = None):
        """Initialize the web application"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path or self._get_default_checkpoint()
        self.model = None
        self.session_history = []
        self.generation_stats = {
            "total_generations": 0,
            "total_time": 0,
            "average_time": 0,
        }

        # Load example images
        self.example_images = self._prepare_examples()

        logger.info(f"MVDiff Web App initialized on {self.device}")

    def _get_default_checkpoint(self) -> str:
        """Get or create default checkpoint"""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        default_path = checkpoint_dir / "mvdiff_demo.pth"

        if not default_path.exists():
            # Create a demo checkpoint
            self._create_demo_checkpoint(default_path)

        return str(default_path)

    def _create_demo_checkpoint(self, path: Path):
        """Create a demo checkpoint for the app"""
        # Simplified model creation for demo
        checkpoint = {
            "model_state_dict": {},  # Would contain actual weights
            "config": {
                "img_size": 256,
                "num_diffusion_steps": 100,
                "model_type": "demo",
            },
            "metadata": {"created": datetime.now().isoformat(), "version": "1.0.0"},
        }
        torch.save(checkpoint, path)
        logger.info(f"Created demo checkpoint at {path}")

    def _prepare_examples(self) -> List[str]:
        """Prepare example images"""
        examples_dir = Path("examples")
        examples_dir.mkdir(exist_ok=True)

        example_paths = []

        # Create synthetic examples if they don't exist
        for name, color, shape in [
            ("chair", "#8B4513", "rect"),
            ("car", "#FF0000", "ellipse"),
            ("airplane", "#4169E1", "polygon"),
        ]:
            img_path = examples_dir / f"{name}.png"
            if not img_path.exists():
                img = self._create_example_image(name, color, shape)
                img.save(img_path)
            example_paths.append(str(img_path))

        return example_paths

    def _create_example_image(self, name: str, color: str, shape: str) -> Image.Image:
        """Create a synthetic example image"""
        img = Image.new("RGB", (256, 256), "#F0F0F0")
        draw = ImageDraw.Draw(img)

        if shape == "rect":
            draw.rectangle([64, 64, 192, 192], fill=color, outline="black", width=2)
        elif shape == "ellipse":
            draw.ellipse([64, 64, 192, 192], fill=color, outline="black", width=2)
        elif shape == "polygon":
            points = [(128, 64), (192, 192), (64, 192)]
            draw.polygon(points, fill=color, outline="black", width=2)

        # Add label
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        text_bbox = draw.textbbox((0, 0), name.upper(), font=font)
        text_width = text_bbox[2] - text_bbox[0]
        draw.text(((256 - text_width) // 2, 220), name.upper(), fill="black", font=font)

        return img

    def load_model(self):
        """Lazy load the model"""
        if self.model is None:
            logger.info("Loading MVDiff model...")
            # Here would load actual model
            # For demo, we'll use a mock model
            self.model = "MockModel"  # Placeholder
            logger.info("Model loaded successfully")
        return self.model

    def generate_views(
        self,
        input_image,
        num_views: int,
        num_steps: int,
        guidance_scale: float,
        elevation: float,
        azimuth_start: float,
        azimuth_end: float,
        radius: float,
        seed: int,
        progress=gr.Progress(),
    ) -> Tuple:
        """Generate multiple views from input image"""

        try:
            # Validate input
            if input_image is None:
                return None, None, None, None, "Please upload an image first"

            # Start timing
            start_time = time.time()

            # Update progress
            progress(0.1, desc="Initializing model...")
            model = self.load_model()

            # Convert input
            if isinstance(input_image, str):
                input_pil = Image.open(input_image)
            else:
                input_pil = Image.fromarray(input_image)

            # Set random seed for reproducibility
            if seed >= 0:
                torch.manual_seed(seed)
                np.random.seed(seed)

            # Generate views (mock for demo)
            progress(0.3, desc="Generating views...")
            views = []
            for i in range(num_views):
                progress(
                    0.3 + 0.5 * i / num_views,
                    desc=f"Generating view {i+1}/{num_views}...",
                )

                # Mock view generation (rotate image)
                angle = azimuth_start + (azimuth_end - azimuth_start) * i / max(
                    1, num_views - 1
                )
                view = input_pil.rotate(angle, fillcolor="white")
                views.append(np.array(view))
                time.sleep(0.1)  # Simulate processing

            # Create visualizations
            progress(0.8, desc="Creating visualizations...")
            grid_view = self._create_grid_view(input_pil, views)
            camera_plot = self._create_camera_plot(num_views, elevation, radius)
            mesh_preview = self._create_mesh_preview(views)

            # Generate statistics
            end_time = time.time()
            generation_time = end_time - start_time

            # Update stats
            self.generation_stats["total_generations"] += 1
            self.generation_stats["total_time"] += generation_time
            self.generation_stats["average_time"] = (
                self.generation_stats["total_time"]
                / self.generation_stats["total_generations"]
            )

            # Create metrics display
            metrics = self._create_metrics_display(
                num_views, num_steps, generation_time
            )

            # Success message
            status = f"""
            **Generation Complete!**
            - Generated {num_views} views in {generation_time:.2f} seconds
            - Average time per view: {generation_time/num_views:.3f}s
            - Model: {self.device.type.upper()}
            - Seed: {seed if seed >= 0 else 'Random'}
            """

            progress(1.0, desc="Complete!")

            # Add to session history
            self.session_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "num_views": num_views,
                    "generation_time": generation_time,
                }
            )

            return grid_view, camera_plot, mesh_preview, metrics, status

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None, None, None, None, f" Error: {str(e)}"

    def _create_grid_view(
        self, input_image: Image.Image, views: List[np.ndarray]
    ) -> Image.Image:
        """Create a grid visualization of all views"""
        n_views = len(views) + 1
        cols = int(np.ceil(np.sqrt(n_views)))
        rows = int(np.ceil(n_views / cols))

        cell_size = 200
        padding = 10

        # Create canvas
        width = cols * (cell_size + padding) + padding
        height = rows * (cell_size + padding) + padding + 40

        canvas = Image.new("RGB", (width, height), "#F5F5F5")
        draw = ImageDraw.Draw(canvas)

        # Add title
        try:
            title_font = ImageFont.truetype("arial.ttf", 24)
            label_font = ImageFont.truetype("arial.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()

        draw.text(
            (width // 2 - 100, 10),
            "Multi-View Generation",
            fill="#333333",
            font=title_font,
        )

        # Add images
        images = [input_image] + [Image.fromarray(v) for v in views]
        labels = ["Input"] + [f"View {i+1}" for i in range(len(views))]

        for idx, (img, label) in enumerate(zip(images, labels)):
            row = idx // cols
            col = idx % cols

            x = col * (cell_size + padding) + padding
            y = row * (cell_size + padding) + padding + 40

            # Resize and paste image
            img_resized = img.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            canvas.paste(img_resized, (x, y))

            # Add border
            draw.rectangle(
                [x - 1, y - 1, x + cell_size, y + cell_size], outline="#CCCCCC", width=2
            )

            # Add label
            label_bbox = draw.textbbox((0, 0), label, font=label_font)
            label_width = label_bbox[2] - label_bbox[0]
            draw.text(
                (x + (cell_size - label_width) // 2, y + cell_size + 2),
                label,
                fill="#666666",
                font=label_font,
            )

        return canvas

    def _create_camera_plot(
        self, num_views: int, elevation: float, radius: float
    ) -> go.Figure:
        """Create 3D plot of camera positions"""
        # Generate camera positions
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

        x = radius * np.cos(angles)
        y = np.ones(num_views) * elevation
        z = radius * np.sin(angles)

        # Create 3D scatter plot
        fig = go.Figure()

        # Add camera positions
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers+text",
                marker=dict(size=10, color="red"),
                text=[f"Cam {i+1}" for i in range(num_views)],
                textposition="top center",
                name="Cameras",
            )
        )

        # Add center object
        fig.add_trace(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode="markers",
                marker=dict(size=20, color="green"),
                text=["Object"],
                name="Object",
            )
        )

        # Add camera rays
        for i in range(num_views):
            fig.add_trace(
                go.Scatter3d(
                    x=[x[i], 0],
                    y=[y[i], 0],
                    z=[z[i], 0],
                    mode="lines",
                    line=dict(color="blue", width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Update layout
        fig.update_layout(
            title="Camera Configuration",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            showlegend=True,
            height=400,
        )

        return fig

    def _create_mesh_preview(self, views: List[np.ndarray]) -> Image.Image:
        """Create a preview of 3D reconstruction"""
        # Mock 3D reconstruction preview
        img = Image.new("RGB", (400, 400), "#E0E0E0")
        draw = ImageDraw.Draw(img)

        # Draw mock 3D wireframe
        center = (200, 200)
        size = 150

        # Draw cube wireframe
        points = [
            (center[0] - size // 2, center[1] - size // 2),
            (center[0] + size // 2, center[1] - size // 2),
            (center[0] + size // 2, center[1] + size // 2),
            (center[0] - size // 2, center[1] + size // 2),
        ]

        # Front face
        draw.polygon(points, outline="#4A90E2", width=2)

        # Back face (offset)
        offset = 30
        back_points = [(p[0] + offset, p[1] - offset) for p in points]
        draw.polygon(back_points, outline="#7AB8F5", width=2)

        # Connect faces
        for i in range(4):
            draw.line([points[i], back_points[i]], fill="#7AB8F5", width=2)

        # Add label
        draw.text((200 - 60, 350), "3D Reconstruction", fill="#333333")
        draw.text((200 - 40, 370), "(Preview)", fill="#666666")

        return img

    def _create_metrics_display(
        self, num_views: int, num_steps: int, generation_time: float
    ) -> str:
        """Create metrics display HTML"""
        metrics_html = f"""
        <div style='display: flex; justify-content: space-around; padding: 20px;'>
            <div class='metric-card'>
                <h4 style='color: #4A90E2;'>Views Generated</h4>
                <p style='font-size: 24px; font-weight: bold;'>{num_views}</p>
            </div>
            <div class='metric-card'>
                <h4 style='color: #4A90E2;'>Diffusion Steps</h4>
                <p style='font-size: 24px; font-weight: bold;'>{num_steps}</p>
            </div>
            <div class='metric-card'>
                <h4 style='color: #4A90E2;'>Generation Time</h4>
                <p style='font-size: 24px; font-weight: bold;'>{generation_time:.2f}s</p>
            </div>
            <div class='metric-card'>
                <h4 style='color: #4A90E2;'>Time per View</h4>
                <p style='font-size: 24px; font-weight: bold;'>{generation_time/num_views:.3f}s</p>
            </div>
        </div>
        """
        return metrics_html

    def export_results(self, grid_view, views_data) -> str:
        """Export generation results"""
        try:
            # Create export directory
            export_dir = Path("exports") / datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir.mkdir(parents=True, exist_ok=True)

            # Save grid view
            if grid_view is not None:
                grid_path = export_dir / "grid_view.png"
                Image.fromarray(grid_view).save(grid_path)

            # Save metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "generation_stats": self.generation_stats,
                "session_history": self.session_history,
            }

            metadata_path = export_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            return f" Results exported to {export_dir}"
        except Exception as e:
            return f" Export failed: {str(e)}"

    def get_session_stats(self) -> str:
        """Get session statistics"""
        if not self.session_history:
            return "No generations yet"

        stats = f"""
        **Session Statistics**
        - Total Generations: {len(self.session_history)}
        - Average Generation Time: {self.generation_stats['average_time']:.2f}s
        - Total Processing Time: {self.generation_stats['total_time']:.2f}s
        - Device: {self.device.type.upper()}
        """
        return stats


def create_interface():
    """Create the professional Gradio interface"""

    app = MVDiffWebApp()

    with gr.Blocks(
        title="MVDiff - Multi-View Generation", theme=gr.themes.Soft(), css=CUSTOM_CSS
    ) as demo:

        # Header
        gr.HTML(
            """
        <div class='header'>
            <h1> MVDiff: Multi-View Diffusion</h1>
            <p>State-of-the-art single image to multi-view generation and 3D reconstruction</p>
            <p>
                <a href='https://arxiv.org/abs/2405.03894' target='_blank'>📄 Paper</a> | 
                <a href='https://github.com/mvdiff' target='_blank'>💻 Code</a> 
            </p>
        </div>
        """
        )

        with gr.Tabs():
            # Generation Tab
            with gr.TabItem(" Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Input Section
                        gr.Markdown("### Input")
                        input_image = gr.Image(
                            label="Upload Image", type="numpy", elem_id="input-image"
                        )

                        # Basic Settings
                        gr.Markdown("### Basic Settings")
                        num_views = gr.Slider(
                            minimum=4,
                            maximum=24,
                            value=8,
                            step=1,
                            label="Number of Views",
                            info="How many views to generate",
                        )
                        num_steps = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=30,
                            step=5,
                            label="Diffusion Steps",
                            info="Higher = better quality, slower",
                        )
                        guidance_scale = gr.Slider(
                            minimum=1.0,
                            maximum=10.0,
                            value=3.0,
                            step=0.5,
                            label="Guidance Scale",
                            info="Higher = more faithful to input",
                        )

                        # Advanced Settings
                        with gr.Accordion(" Advanced Settings", open=False):
                            elevation = gr.Slider(
                                minimum=-90,
                                maximum=90,
                                value=15,
                                step=5,
                                label="Camera Elevation (degrees)",
                            )
                            azimuth_start = gr.Slider(
                                minimum=0,
                                maximum=360,
                                value=0,
                                step=15,
                                label="Azimuth Start (degrees)",
                            )
                            azimuth_end = gr.Slider(
                                minimum=0,
                                maximum=360,
                                value=360,
                                step=15,
                                label="Azimuth End (degrees)",
                            )
                            radius = gr.Slider(
                                minimum=1.0,
                                maximum=5.0,
                                value=2.5,
                                step=0.1,
                                label="Camera Distance",
                            )
                            seed = gr.Number(
                                value=-1,
                                precision=0,
                                label="Random Seed (-1 for random)",
                            )

                        # Action Buttons
                        with gr.Row():
                            generate_btn = gr.Button(
                                "Generate Views",
                                variant="primary",
                                elem_id="generate-btn",
                            )
                            clear_btn = gr.Button("Clear", variant="secondary")

                        # Examples
                        gr.Markdown("### Example Images")
                        gr.Examples(
                            examples=app.example_images,
                            inputs=input_image,
                            label="Click to load",
                        )

                    with gr.Column(scale=2):
                        # Output Section
                        gr.Markdown("### Generated Views")
                        output_grid = gr.Image(
                            label="Multi-View Grid", elem_id="output-grid"
                        )

                        with gr.Row():
                            with gr.Tab(" Camera Plot"):
                                camera_plot = gr.Plot(label="Camera Positions")

                            with gr.Tab(" 3D Preview"):
                                mesh_preview = gr.Image(
                                    label="3D Reconstruction Preview"
                                )

                        # Metrics Display
                        metrics_display = gr.HTML(label="Generation Metrics")

                        # Status Display
                        status_text = gr.Markdown(label="Status", elem_id="status")

            # Gallery Tab
            with gr.TabItem("Gallery"):
                gr.Markdown(
                    """
                ### Recent Generations
                Browse your recent multi-view generations
                """
                )
                gallery = gr.Gallery(
                    label="Generation History",
                    show_label=False,
                    elem_id="gallery",
                    columns=3,
                    rows=2,
                    height="auto",
                )

                with gr.Row():
                    refresh_btn = gr.Button(" Refresh Gallery")
                    export_btn = gr.Button(" Export All")

                export_status = gr.Textbox(label="Export Status", interactive=False)

            # Settings Tab
            with gr.TabItem(" Settings"):
                gr.Markdown(
                    """
                ### Model Configuration
                Configure model and inference settings
                """
                )

                with gr.Row():
                    with gr.Column():
                        model_dropdown = gr.Dropdown(
                            choices=["MVDiff-Base", "MVDiff-Large", "MVDiff-CO3D"],
                            value="MVDiff-Base",
                            label="Model Selection",
                        )

                        device_radio = gr.Radio(
                            choices=["CPU", "CUDA", "MPS"],
                            value="CUDA" if torch.cuda.is_available() else "CPU",
                            label="Device",
                        )

                        precision_radio = gr.Radio(
                            choices=["fp32", "fp16", "int8"],
                            value="fp32",
                            label="Precision",
                        )

                    with gr.Column():
                        batch_size = gr.Slider(
                            minimum=1, maximum=8, value=1, step=1, label="Batch Size"
                        )

                        cache_checkbox = gr.Checkbox(
                            value=True, label="Enable Model Caching"
                        )

                        stats_display = gr.Markdown(value=app.get_session_stats())

            # About Tab
            with gr.TabItem(" About"):
                gr.Markdown(
                    """
                ## About MVDiff
                
                **MVDiff** is a state-of-the-art diffusion model for generating consistent multi-view images from single-view inputs.
                
                ### Key Features
                -  **High Quality**: Generates photorealistic multi-view images
                -  **3D Consistent**: Maintains geometric consistency across views
                -  **Fast Inference**: Optimized for real-time generation
                -  **Flexible**: Works with various object categories
                
                ### Citation
                ```bibtex
                @article{bourigault2024mvdiff,
                  title={MVDiff: Scalable and Flexible Multi-View Diffusion},
                  author={Bourigault, E. and Bourigault, P.},
                  journal={arXiv:2405.03894},
                  year={2024}
                }
                ```
                
                ### License
                This project is licensed under the MIT License.
                
                """
                )

        # Wire up the interface
        generate_btn.click(
            fn=app.generate_views,
            inputs=[
                input_image,
                num_views,
                num_steps,
                guidance_scale,
                elevation,
                azimuth_start,
                azimuth_end,
                radius,
                seed,
            ],
            outputs=[
                output_grid,
                camera_plot,
                mesh_preview,
                metrics_display,
                status_text,
            ],
        )

        clear_btn.click(
            fn=lambda: (None, None, None, None, "Ready for input"),
            inputs=[],
            outputs=[
                output_grid,
                camera_plot,
                mesh_preview,
                metrics_display,
                status_text,
            ],
        )

        export_btn.click(
            fn=app.export_results,
            inputs=[output_grid, gallery],
            outputs=[export_status],
        )

        # Add keyboard shortcuts
        demo.load(js=CUSTOM_JS)

    return demo


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="MVDiff Web Application")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model checkpoint"
    )
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create and launch interface
    print(" " * 15 + "MVDiff Web Application")
    print(f"\n Launching on port {args.port}...")
    print(f" Local URL: http://localhost:{args.port}")

    if args.share:
        print(" Creating public link...")

    demo = create_interface()

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        show_api=False,
        favicon_path=None,
    )


if __name__ == "__main__":
    main()
