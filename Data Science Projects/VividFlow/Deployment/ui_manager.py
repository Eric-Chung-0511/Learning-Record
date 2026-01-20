import gradio as gr
from PIL import Image
from typing import Tuple, Optional, Dict, Any
import os
import logging

from FlowFacade import FlowFacade
from BackgroundEngine import BackgroundEngine
from style_transfer import StyleTransferEngine
from scene_templates import SceneTemplateManager
from css_style import DELTAFLOW_CSS
from prompt_examples import PROMPT_EXAMPLES

try:
    import spaces
    SPACES_AVAILABLE = True
except ImportError:
    SPACES_AVAILABLE = False

logger = logging.getLogger(__name__)


class UIManager:
    def __init__(self, facade: FlowFacade, background_engine: BackgroundEngine, style_engine: StyleTransferEngine):
        self.facade = facade
        self.background_engine = background_engine
        self.style_engine = style_engine
        self.template_manager = SceneTemplateManager()

    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(
            theme=gr.themes.Soft(),
            css=DELTAFLOW_CSS,
            title="VividFlow - AI Image Enhancement & Video Generation"
        ) as interface:

            # Header
            gr.HTML("""
                <div class="header-container">
                    <h1 class="header-title">🌊 VividFlow</h1>
                    <p class="header-subtitle">
                        AI-Powered Image Enhancement & Video Generation<br>
                        Transform images with background replacement, then bring them to life with AI
                    </p>
                </div>
            """)

            # Main Tabs
            with gr.Tabs() as main_tabs:

                # Tab 1: Image to Video
                with gr.Tab("🎬 Image to Video"):
                    self._create_i2v_tab()

                # Tab 2: Background Generation
                with gr.Tab("🎨 Background Generation"):
                    self._create_background_tab()

                # Tab 3: AI Style Transfer
                with gr.Tab("✨ Style Transfer"):
                    self._create_3d_tab()

            # Footer
            gr.HTML("""
                <div class="footer">
                    <p>Powered by Wan2.2-I2V-A14B, SDXL, and OpenCLIP | Built with Gradio</p>
                    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e0e0e0;">
                        <p style="font-size: 0.9rem; color: #6c757d; margin-bottom: 0.75rem;">
                            💡 Curious about the technical details?
                        </p>
                        <a href="https://github.com/Eric-Chung-0511/Learning-Record/tree/main/Data%20Science%20Projects/VividFlow"
                           target="_blank"
                           style="display: inline-block; padding: 10px 24px; background: linear-gradient(135deg, #24292e 0%, #1a1e22 100%);
                                  color: white; text-decoration: none; border-radius: 8px; font-size: 0.9rem; font-weight: 500;
                                  transition: all 0.3s ease; box-shadow: 0 2px 8px rgba(36, 41, 46, 0.2);">
                            <span style="margin-right: 8px;">⭐</span>
                            Explore the technical docs on GitHub
                            <span style="margin-left: 8px;">→</span>
                        </a>
                    </div>
                </div>
            """)

        return interface

    def _create_i2v_tab(self):
        """Create Image to Video tab (original VividFlow functionality)"""
        with gr.Row():
            # Left Panel: Input
            with gr.Column(scale=1, elem_classes="input-card"):
                gr.Markdown("### 📤 Input")

                image_input = gr.Image(
                    label="Upload Image (any type: photo, art, cartoon, etc.)",
                    type="pil",
                    elem_classes="image-upload",
                    height=320
                )

                resolution_info = gr.Markdown(
                    value="",
                    visible=False,
                    elem_classes="info-text"
                )

                prompt_input = gr.Textbox(
                    label="Motion Instruction",
                    placeholder="Describe camera movements and subject actions...",
                    lines=3,
                    max_lines=6
                )

                category_dropdown = gr.Dropdown(
                    choices=list(PROMPT_EXAMPLES.keys()),
                    label="💡 Quick Prompt Category",
                    value="💃 Fashion / Beauty (Facial Only)",
                    interactive=True
                )

                example_dropdown = gr.Dropdown(
                    choices=PROMPT_EXAMPLES["💃 Fashion / Beauty (Facial Only)"],
                    label="Example Prompts (click to use)",
                    value=None,
                    interactive=True
                )

                gr.HTML("""
                    <div class="quality-banner">
                        <strong>💡 Choose the Right Prompt Category:</strong><br>
                        • <strong>💃 Facial Only:</strong> Safe for headshots without visible hands<br>
                        • <strong>🙌 Hands Visible Required:</strong> Only use if hands are fully visible<br>
                        • <strong>🌄 Scenery/Objects:</strong> For landscapes, products, abstract content
                    </div>
                """)

                gr.HTML("""
                    <div class="patience-banner">
                        <strong>⏱️ First-time loading may take a moment!</strong><br>
                        Subsequent runs will be much faster.
                    </div>
                """)

                generate_btn = gr.Button(
                    "🎬 Generate Video",
                    variant="primary",
                    elem_classes="primary-button",
                    size="lg"
                )

                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    duration_slider = gr.Slider(
                        minimum=0.5,
                        maximum=5.0,
                        value=3.0,
                        step=0.5,
                        label="Video Duration (seconds)"
                    )

                    steps_slider = gr.Slider(
                        minimum=4,
                        maximum=25,
                        value=4,
                        step=1,
                        label="Quality Steps (4=Lightning Fast, 8-25=Higher Quality)"
                    )

                    fps_slider = gr.Slider(
                        minimum=8,
                        maximum=24,
                        value=16,
                        step=1,
                        label="Frames Per Second"
                    )

                    expand_prompt = gr.Checkbox(
                        label="AI Prompt Expansion (experimental)",
                        value=False
                    )

                    randomize_seed = gr.Checkbox(
                        label="Randomize Seed",
                        value=True
                    )

                    seed_input = gr.Number(
                        label="Manual Seed (if not randomized)",
                        value=42,
                        precision=0
                    )

            # Right Panel: Output
            with gr.Column(scale=1, elem_classes="output-card"):
                gr.Markdown("### 🎥 Output")

                video_output = gr.Video(
                    label="Generated Video",
                    elem_classes="video-player"
                )

                final_prompt_output = gr.Textbox(
                    label="Final Prompt Used",
                    interactive=False,
                    lines=2
                )

                seed_output = gr.Number(
                    label="Seed Used",
                    interactive=False,
                    precision=0
                )

        # Event handlers for I2V tab
        def update_resolution_display(img):
            if img is None:
                return gr.update(visible=False)
            w, h = img.size
            new_w = (w // 16) * 16
            new_h = (h // 16) * 16
            return gr.update(
                value=f"📐 **Resolution:** Input: {w}×{h} → Output: {new_w}×{new_h}",
                visible=True
            )

        def category_changed(category):
            if category in PROMPT_EXAMPLES:
                return gr.update(choices=PROMPT_EXAMPLES[category], value=None)
            return gr.update()

        def example_selected(example):
            return example if example else ""

        image_input.change(
            fn=update_resolution_display,
            inputs=[image_input],
            outputs=[resolution_info]
        )

        category_dropdown.change(
            fn=category_changed,
            inputs=[category_dropdown],
            outputs=[example_dropdown]
        )

        example_dropdown.change(
            fn=example_selected,
            inputs=[example_dropdown],
            outputs=[prompt_input]
        )

        generate_btn.click(
            fn=self._generate_video_handler,
            inputs=[
                image_input, prompt_input, duration_slider,
                steps_slider, fps_slider, expand_prompt,
                randomize_seed, seed_input
            ],
            outputs=[video_output, final_prompt_output, seed_output]
        )

    def _generate_video_handler(
        self,
        image: Image.Image,
        prompt: str,
        duration: float,
        steps: int,
        fps: int,
        expand_prompt: bool,
        randomize_seed: bool,
        seed: int
    ) -> Tuple[str, str, int]:
        """Handler for video generation"""
        if image is None:
            return None, "Please upload an image", 0

        if not prompt.strip():
            return None, "Please provide a motion prompt", 0

        try:
            video_path, final_prompt, seed_used = self.facade.generate_video_from_image(
                image=image,
                user_instruction=prompt,
                duration_seconds=duration,
                num_inference_steps=steps,
                enable_prompt_expansion=expand_prompt,
                randomize_seed=randomize_seed,
                seed=seed
            )
            return video_path, final_prompt, seed_used

        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            return None, f"Error: {str(e)}", 0


    def _create_background_tab(self):
        """Create Background Generation tab (SceneWeaver functionality)"""
        with gr.Row():
            # Left Panel: Input
            with gr.Column(scale=1, elem_classes="feature-card"):
                gr.Markdown("### 📸 Upload & Configure")

                gr.HTML("""
                    <div class="quality-banner">
                        <strong>💡 Best Results Tips:</strong><br>
                        • Clean portrait photos with simple backgrounds work best<br>
                        • Complex scenes (e.g., pets with grass) may need parameter adjustments<br>
                        • Use Advanced Options below to fine-tune edge blending
                    </div>
                """)

                bg_image_input = gr.Image(
                    label="Upload Your Image",
                    type="pil",
                    height=280
                )

                # Scene Template Selector
                template_dropdown = gr.Dropdown(
                    label="Scene Templates (24 curated scenes A-Z)",
                    choices=[""] + self.template_manager.get_template_choices_sorted(),
                    value="",
                    info="Optional: Select a preset or describe your own",
                    elem_classes=["template-dropdown"]
                )

                bg_prompt_input = gr.Textbox(
                    label="Background Scene Description",
                    placeholder="Select a template above or describe your own scene...",
                    lines=3
                )

                combination_mode = gr.Dropdown(
                    label="Composition Mode",
                    choices=["center", "left_half", "right_half", "full"],
                    value="center",
                    info="center=Smart Center | full=Full Image"
                )

                focus_mode = gr.Dropdown(
                    label="Focus Mode",
                    choices=["person", "scene"],
                    value="person",
                    info="person=Tight Crop | scene=Include Surrounding"
                )

                with gr.Accordion("Advanced Options", open=False):
                    gr.HTML("""
                        <div style="padding: 8px; background: #f0f4ff; border-radius: 6px; margin-bottom: 12px; font-size: 13px;">
                            <strong>💡 When to Adjust:</strong><br>
                            • <strong>Enhance Dark Edges:</strong> Enable for images with dark/black backgrounds where foreground parts get lost.<br>
                            • <strong>Feather Radius:</strong> Use 5-10 for complex scenes with fine details (hair, fur, foliage). 0 = sharp edges for clean portraits.<br>
                            • <strong>Mask Preview:</strong> Check the "Mask Preview" tab after generation. White = kept, Black = replaced.
                        </div>
                    """)

                    enhance_dark_edges = gr.Checkbox(
                        label="🌙 Enhance Dark Edges",
                        value=False,
                        info="Enable if dark foreground parts blend into dark backgrounds"
                    )
                    gr.HTML("""
                        <div style="padding: 6px 8px; background: #fff3cd; border-radius: 4px; font-size: 11px; margin-bottom: 12px;">
                            <strong>When to use:</strong> If mask preview shows gray areas where foreground should be white (e.g., dark hair/clothing on dark background).
                            Auto-detection is enabled by default, but this toggle forces stronger enhancement.
                        </div>
                    """)

                    feather_radius_slider = gr.Slider(
                        label="Feather Radius (Edge Softness)",
                        minimum=0,
                        maximum=20,
                        value=0,
                        step=1,
                        info="Softens mask edges. Try 5-10 if edges look harsh."
                    )

                    bg_negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="blurry, low quality, distorted, people, characters",
                        lines=2,
                        info="Prevents unwanted elements in background"
                    )

                    bg_steps_slider = gr.Slider(
                        label="Quality Steps",
                        minimum=15,
                        maximum=50,
                        value=25,
                        step=5,
                        info="Higher = better quality but slower"
                    )

                    bg_guidance_slider = gr.Slider(
                        label="Guidance Scale",
                        minimum=5.0,
                        maximum=15.0,
                        value=7.5,
                        step=0.5,
                        info="How strictly to follow prompt"
                    )

                generate_bg_btn = gr.Button(
                    "🎨 Generate Background",
                    variant="primary",
                    elem_classes="primary-button",
                    size="lg"
                )

            # Right Panel: Output
            with gr.Column(scale=2, elem_classes="feature-card"):
                gr.Markdown("### 🎭 Results Gallery")

                gr.HTML("""
                    <div class="patience-banner">
                        <strong>⏱️ First-time users:</strong> Initial model loading takes 30-60 seconds.
                        Subsequent generations are much faster (~30s).
                    </div>
                """)

                with gr.Tabs():
                    with gr.TabItem("Final Result"):
                        bg_combined_output = gr.Image(
                            label="Your Generated Image",
                            elem_classes=["result-gallery"]
                        )
                    with gr.TabItem("Background"):
                        bg_generated_output = gr.Image(
                            label="Generated Background",
                            elem_classes=["result-gallery"]
                        )
                    with gr.TabItem("Original"):
                        bg_original_output = gr.Image(
                            label="Processed Original",
                            elem_classes=["result-gallery"]
                        )
                    with gr.TabItem("Mask Preview"):
                        gr.HTML("""
                            <div style="padding: 8px; background: #f0f4ff; border-radius: 6px; margin-bottom: 8px; font-size: 13px;">
                                <strong>📐 How to Read:</strong> White = Original kept | Black = Background replaced<br>
                                Use this to diagnose edge quality. If edges are too harsh, increase Feather Radius.
                            </div>
                        """)
                        bg_mask_output = gr.Image(
                            label="Blending Mask",
                            elem_classes=["result-gallery"]
                        )

                bg_status_output = gr.Textbox(
                    label="Status",
                    value="Ready to create! Upload an image and describe your vision.",
                    interactive=False,
                    elem_classes=["status-panel"]
                )

                with gr.Row():
                    clear_bg_btn = gr.Button(
                        "Clear All",
                        elem_classes=["secondary-button"]
                    )
                    memory_btn = gr.Button(
                        "Clean Memory",
                        elem_classes=["secondary-button"]
                    )

                # Touch Up Section for manual artifact removal
                with gr.Accordion("🖌️ Touch Up (Remove Artifacts)", open=False) as touchup_accordion:
                    gr.HTML("""
                        <div style="padding: 10px; background: #e8f4fd; border-radius: 6px; margin-bottom: 12px; font-size: 13px;">
                            <strong>✨ How to Use Touch Up:</strong><br>
                            1. After generating, if you see unwanted artifacts (gray edges, leftover objects)<br>
                            2. Click "Load Result for Touch Up" to load the image<br>
                            3. Use the brush to paint over areas you want to remove<br>
                            4. Click "Remove & Fill" to replace painted areas with background
                        </div>
                    """)

                    # State to store the current result and prompt
                    touchup_source_image = gr.State(value=None)
                    touchup_background_prompt = gr.State(value="")

                    load_touchup_btn = gr.Button(
                        "📥 Load Result for Touch Up",
                        elem_classes=["secondary-button"]
                    )

                    touchup_editor = gr.ImageEditor(
                        label="Draw on areas to remove (use brush tool)",
                        type="pil",
                        height=400,
                        brush=gr.Brush(
                            colors=["#FF0000"],
                            default_color="#FF0000",
                            default_size=20
                        ),
                        layers=False,
                        interactive=True,
                        visible=True
                    )

                    with gr.Row():
                        brush_size_slider = gr.Slider(
                            label="Brush Size",
                            minimum=5,
                            maximum=50,
                            value=20,
                            step=5,
                            scale=2
                        )
                        touchup_strength = gr.Slider(
                            label="Fill Strength",
                            minimum=0.8,
                            maximum=1.0,
                            value=0.99,
                            step=0.01,
                            scale=2,
                            info="Higher = more complete replacement"
                        )

                    remove_fill_btn = gr.Button(
                        "🎨 Remove & Fill",
                        variant="primary",
                        elem_classes="primary-button"
                    )

                    touchup_result = gr.Image(
                        label="Touch Up Result",
                        elem_classes=["result-gallery"]
                    )

                    touchup_status = gr.Textbox(
                        label="Touch Up Status",
                        value="Load an image to start touch up.",
                        interactive=False
                    )

        # Event handlers for Background Generation tab
        def apply_template(display_name: str, current_negative: str) -> Tuple[str, str, float]:
            if not display_name:
                return "", current_negative, 7.5

            template_key = self.template_manager.get_template_key_from_display(display_name)
            if not template_key:
                return "", current_negative, 7.5

            template = self.template_manager.get_template(template_key)
            if template:
                prompt = template.prompt
                negative = self.template_manager.get_negative_prompt_for_template(
                    template_key, current_negative
                )
                guidance = template.guidance_scale
                return prompt, negative, guidance

            return "", current_negative, 7.5

        template_dropdown.change(
            fn=apply_template,
            inputs=[template_dropdown, bg_negative_prompt],
            outputs=[bg_prompt_input, bg_negative_prompt, bg_guidance_slider]
        )

        generate_bg_btn.click(
            fn=self._generate_background_handler,
            inputs=[
                bg_image_input, bg_prompt_input, combination_mode,
                focus_mode, bg_negative_prompt, bg_steps_slider, bg_guidance_slider,
                feather_radius_slider, enhance_dark_edges
            ],
            outputs=[
                bg_combined_output, bg_generated_output,
                bg_original_output, bg_mask_output, bg_status_output
            ]
        )

        clear_bg_btn.click(
            fn=lambda: (None, None, None, None, "Ready to create!"),
            outputs=[
                bg_combined_output, bg_generated_output,
                bg_original_output, bg_mask_output, bg_status_output
            ]
        )

        memory_btn.click(
            fn=lambda: self.background_engine._memory_cleanup() or "Memory cleaned!",
            outputs=[bg_status_output]
        )

        # Touch Up event handlers
        def load_for_touchup(combined_image, prompt):
            """Load the generated result into touch up editor"""
            if combined_image is None:
                return None, None, "", "Please generate a background first!"
            return combined_image, combined_image, prompt, "✓ Image loaded! Use brush to paint areas to remove."

        load_touchup_btn.click(
            fn=load_for_touchup,
            inputs=[bg_combined_output, bg_prompt_input],
            outputs=[touchup_editor, touchup_source_image, touchup_background_prompt, touchup_status]
        )

        remove_fill_btn.click(
            fn=self._touchup_inpaint_handler,
            inputs=[touchup_editor, touchup_background_prompt, touchup_strength],
            outputs=[touchup_result, touchup_status]
        )

    def _touchup_inpaint_handler(
        self,
        editor_data: dict,
        background_prompt: str,
        strength: float
    ) -> Tuple[Optional[Image.Image], str]:
        """Handler for touch up inpainting"""
        if editor_data is None:
            return None, "Please load an image first!"

        try:
            # Extract image and mask from editor
            # Gradio ImageEditor returns a dict with 'background', 'layers', 'composite'
            if isinstance(editor_data, dict):
                base_image = editor_data.get("background") or editor_data.get("composite")
                layers = editor_data.get("layers", [])

                if base_image is None:
                    return None, "No image found in editor!"

                # Create mask from drawn layers (red brush strokes)
                mask = self._extract_mask_from_editor(base_image, layers)

                if mask is None or not self._has_painted_area(mask):
                    return None, "Please draw on areas you want to remove!"

            else:
                # Fallback for PIL Image
                return None, "Invalid editor data format!"

            # Apply ZeroGPU decorator if available
            if SPACES_AVAILABLE:
                inpaint_fn = spaces.GPU(duration=60)(self._touchup_inpaint_core)
            else:
                inpaint_fn = self._touchup_inpaint_core

            result = inpaint_fn(base_image, mask, background_prompt, strength)

            if result["success"]:
                return result["inpainted_image"], "✓ Touch up completed!"
            else:
                return None, f"Error: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Touch up failed: {e}")
            return None, f"Error: {str(e)}"

    def _extract_mask_from_editor(self, base_image: Image.Image, layers: list) -> Optional[Image.Image]:
        """Extract painted mask from ImageEditor layers"""
        import numpy as np

        if not layers:
            return None

        # Create blank mask
        width, height = base_image.size
        mask_array = np.zeros((height, width), dtype=np.uint8)

        for layer in layers:
            if layer is None:
                continue

            # Convert layer to numpy array
            if isinstance(layer, Image.Image):
                layer_array = np.array(layer.convert('RGBA'))
            else:
                continue

            # Find non-transparent pixels (painted areas)
            # The alpha channel indicates where user drew
            if layer_array.shape[2] >= 4:
                alpha = layer_array[:, :, 3]
                # Also check for red color (our brush color)
                red = layer_array[:, :, 0]
                # Painted areas have high alpha and red channel
                painted = (alpha > 50) | (red > 100)
                mask_array[painted] = 255

        return Image.fromarray(mask_array, mode='L')

    def _has_painted_area(self, mask: Image.Image) -> bool:
        """Check if mask has any painted area"""
        import numpy as np
        mask_array = np.array(mask)
        return np.sum(mask_array > 127) > 100  # At least 100 white pixels

    def _touchup_inpaint_core(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        strength: float
    ) -> dict:
        """Core inpainting function"""
        # Use the background prompt to fill in the masked areas
        inpaint_prompt = f"{prompt}, seamless, natural continuation, no artifacts" if prompt else "natural background, seamless continuation"

        return self.background_engine.inpaint_region(
            image=image,
            mask=mask,
            prompt=inpaint_prompt,
            negative_prompt="blurry, artifacts, seams, inconsistent, unnatural",
            num_inference_steps=20,
            guidance_scale=7.5,
            strength=float(strength)
        )

    def _generate_background_handler(
        self,
        image: Image.Image,
        prompt: str,
        combination_mode: str,
        focus_mode: str,
        negative_prompt: str,
        steps: int,
        guidance: float,
        feather_radius: int,
        enhance_dark_edges: bool = False
    ) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[Image.Image], Optional[Image.Image], str]:
        """Handler for background generation"""
        if image is None:
            return None, None, None, None, "Please upload an image to get started!"

        if not prompt.strip():
            return None, None, None, None, "Please describe the background scene you'd like!"

        try:
            # Apply ZeroGPU decorator if available
            if SPACES_AVAILABLE:
                generate_fn = spaces.GPU(duration=60)(self._background_generate_core)
            else:
                generate_fn = self._background_generate_core

            result = generate_fn(
                image, prompt, combination_mode, focus_mode,
                negative_prompt, steps, guidance, feather_radius, enhance_dark_edges
            )

            if result["success"]:
                return (
                    result["combined_image"],
                    result["generated_scene"],
                    result["original_image"],
                    result["mask"],
                    "Image created successfully!"
                )
            else:
                error_msg = result.get("error", "Something went wrong")
                return None, None, None, None, f"Error: {error_msg}"

        except Exception as e:
            logger.error(f"Background generation failed: {e}")
            return None, None, None, None, f"Error: {str(e)}"

    def _background_generate_core(
        self,
        image: Image.Image,
        prompt: str,
        combination_mode: str,
        focus_mode: str,
        negative_prompt: str,
        steps: int,
        guidance: float,
        feather_radius: int,
        enhance_dark_edges: bool = False
    ) -> Dict[str, Any]:
        """Core background generation with models"""
        if not self.background_engine.is_initialized:
            logger.info("Loading background generation models...")
            self.background_engine.load_models()

        result = self.background_engine.generate_and_combine(
            original_image=image,
            prompt=prompt,
            combination_mode=combination_mode,
            focus_mode=focus_mode,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            enable_prompt_enhancement=True,
            feather_radius=int(feather_radius),
            enhance_dark_edges=enhance_dark_edges
        )

        return result

    def _create_3d_tab(self):
        """Create Style Transfer tab - converts images to various artistic styles"""
        with gr.Row():
            # Left Panel: Input & Settings
            with gr.Column(scale=1, elem_classes="feature-card"):
                gr.Markdown("### 🎨 AI Style Transfer")

                # How It Works Guide
                gr.HTML("""
                    <div class="quality-banner">
                        <strong>📖 Transform Your Photos</strong><br><br>
                        Convert your images into <strong>stunning artistic styles</strong>!<br><br>
                        <strong>🎨 Single Styles:</strong> Pure artistic transformations<br>
                        <strong>🎭 Style Blends:</strong> Unique combinations for distinctive looks<br><br>
                        <strong>💡 Tips:</strong><br>
                        • Use <strong>Seed</strong> to recreate the exact same result<br>
                        • Try different blends for unique artistic effects
                    </div>
                """)

                # Step 1: Upload
                gr.Markdown("#### Step 1: Upload Image")
                style3d_image_input = gr.Image(
                    label="Upload Your Image",
                    type="pil",
                    height=280
                )

                # Step 2: Choose Style
                gr.Markdown("#### Step 2: Choose Style")

                # Hidden state to track which mode is active (updated by tab selection)
                is_blend_mode = gr.State(value=False)

                with gr.Tabs() as style_tabs:
                    with gr.TabItem("🎨 Single Styles", id="single_tab") as single_tab:
                        style_dropdown = gr.Dropdown(
                            choices=self.style_engine.get_style_choices(),
                            value="🎬 3D Cartoon",
                            label="Art Style",
                            info="Select a single artistic style"
                        )

                        style_strength = gr.Slider(
                            label="Style Strength",
                            minimum=0.3,
                            maximum=0.7,
                            value=0.50,
                            step=0.05,
                            info="Lower = keep more original | Higher = stronger style (0.45-0.55 recommended)"
                        )

                    with gr.TabItem("🎭 Style Blends", id="blend_tab") as blend_tab:
                        blend_dropdown = gr.Dropdown(
                            choices=self.style_engine.get_blend_choices(),
                            value=self.style_engine.get_blend_choices()[0] if self.style_engine.get_blend_choices() else None,
                            label="Blend Preset",
                            info="Pre-configured style combinations"
                        )
                        gr.HTML("""
                            <div style="padding: 8px; background: #f0f4ff; border-radius: 6px; font-size: 12px; margin-top: 8px;">
                                <strong>Available Blends:</strong><br>
                                • 🎭 3D Anime Fusion - 3D + Anime linework<br>
                                • 🌈 Dreamy Watercolor - Fantasy + Watercolor<br>
                                • 📖 Anime Storybook - Anime + Fantasy<br>
                                • 👑 Renaissance Portrait - Classical oil painting<br>
                                • 🕹️ Retro Game Art - Enhanced pixel art
                            </div>
                        """)

                # Face Restore option for identity preservation
                face_restore = gr.Checkbox(
                    label="🛡️ Face Restore (Preserve Identity)",
                    value=False,
                    info="Enable to better preserve facial features and prevent identity changes"
                )
                gr.HTML("""
                    <div style="padding: 6px 8px; background: #fff3cd; border-radius: 4px; font-size: 11px; margin-top: 4px;">
                        <strong>💡 When to use:</strong> Enable if the style changes the person's face, age, or ethnicity too much.
                        Auto-reduces strength to preserve original features.
                    </div>
                """)

                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=5.0,
                        maximum=12.0,
                        value=7.5,
                        step=0.5,
                        info="How closely to follow the style"
                    )

                    num_steps = gr.Slider(
                        label="Quality Steps",
                        minimum=20,
                        maximum=50,
                        value=30,
                        step=5,
                        info="More steps = better quality but slower"
                    )

                    custom_prompt = gr.Textbox(
                        label="Additional Description (optional)",
                        placeholder="e.g., smiling, dramatic lighting, vibrant colors...",
                        lines=2
                    )

                    gr.Markdown("##### 🎲 Seed Control")
                    randomize_seed = gr.Checkbox(
                        label="Randomize Seed",
                        value=True,
                        info="Uncheck to use manual seed for reproducible results"
                    )

                    seed_input = gr.Number(
                        label="Manual Seed",
                        value=42,
                        precision=0,
                        info="Use same seed to reproduce exact results"
                    )

                # Step 3: Generate
                gr.Markdown("#### Step 3: Generate")

                gr.HTML("""
                    <div class="patience-banner">
                        <strong>⏱️ Generation Time:</strong> ~20-30 seconds.
                        First-time model loading may take 30-60 seconds.
                    </div>
                """)

                generate_style_btn = gr.Button(
                    "🎨 Transform Image",
                    variant="primary",
                    elem_classes="primary-button",
                    size="lg"
                )

            # Right Panel: Output
            with gr.Column(scale=1, elem_classes="feature-card"):
                gr.Markdown("### 📤 Results")

                with gr.Tabs():
                    with gr.TabItem("Stylized Result"):
                        style3d_output = gr.Image(
                            label="Stylized Result",
                            elem_classes=["result-gallery"]
                        )

                    with gr.TabItem("Original"):
                        style3d_original = gr.Image(
                            label="Original Image",
                            elem_classes=["result-gallery"]
                        )

                    with gr.TabItem("Comparison"):
                        with gr.Row():
                            style3d_compare_original = gr.Image(
                                label="Before",
                                elem_classes=["result-gallery"]
                            )
                            style3d_compare_result = gr.Image(
                                label="After",
                                elem_classes=["result-gallery"]
                            )

                with gr.Row():
                    style3d_status_output = gr.Textbox(
                        label="Status",
                        value="Ready! Upload an image and select a style to transform.",
                        interactive=False,
                        elem_classes=["status-panel"],
                        scale=3
                    )
                    seed_output = gr.Number(
                        label="Seed Used",
                        value=0,
                        interactive=False,
                        precision=0,
                        scale=1
                    )

                with gr.Row():
                    clear_style_btn = gr.Button(
                        "Clear All",
                        elem_classes=["secondary-button"]
                    )
                    memory_style_btn = gr.Button(
                        "Clean Memory",
                        elem_classes=["secondary-button"]
                    )

        # Event handlers - detect mode from TAB selection (not just dropdown)
        single_tab.select(
            fn=lambda: False,  # Single Styles tab clicked -> is_blend = False
            inputs=[],
            outputs=[is_blend_mode]
        )

        blend_tab.select(
            fn=lambda: True,  # Style Blends tab clicked -> is_blend = True
            inputs=[],
            outputs=[is_blend_mode]
        )

        generate_style_btn.click(
            fn=self._generate_3d_style_handler,
            inputs=[
                style3d_image_input, style_dropdown, blend_dropdown, is_blend_mode,
                style_strength, guidance_scale, num_steps, custom_prompt,
                randomize_seed, seed_input, face_restore
            ],
            outputs=[
                style3d_output, style3d_original,
                style3d_compare_original, style3d_compare_result,
                style3d_status_output, seed_output
            ]
        )

        clear_style_btn.click(
            fn=lambda: (None, None, None, None, "Ready! Upload an image and select a style to transform.", 0),
            outputs=[
                style3d_output, style3d_original,
                style3d_compare_original, style3d_compare_result,
                style3d_status_output, seed_output
            ]
        )

        memory_style_btn.click(
            fn=self._cleanup_3d_memory,
            outputs=[style3d_status_output]
        )

    def _generate_3d_style_handler(
        self,
        image: Image.Image,
        style_choice: str,
        blend_choice: str,
        is_blend_mode: bool,
        strength: float,
        guidance_scale: float,
        num_steps: int,
        custom_prompt: str,
        randomize_seed: bool,
        manual_seed: int,
        face_restore: bool = False
    ) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[Image.Image], Optional[Image.Image], str, int]:
        """Handler for style transfer generation"""
        if image is None:
            return None, None, None, None, "Please upload an image first!", 0

        try:
            # Determine style key based on mode (detected from last dropdown interaction)
            if is_blend_mode:
                style_key = self.style_engine.get_blend_key_from_choice(blend_choice)
                is_blend = True
            else:
                style_key = self.style_engine.get_style_key_from_choice(style_choice)
                is_blend = False

            # Handle seed
            seed = -1 if randomize_seed else int(manual_seed)

            if SPACES_AVAILABLE:
                generate_fn = spaces.GPU(duration=120)(self._3d_style_generate_core)
            else:
                generate_fn = self._3d_style_generate_core

            result = generate_fn(
                image, style_key, is_blend, strength,
                guidance_scale, num_steps, custom_prompt, seed, face_restore
            )

            if result["success"]:
                stylized = result["stylized_image"]
                style_name = result.get("style_name", "Style")
                seed_used = result.get("seed_used", 0)
                return (
                    stylized,
                    image,
                    image,
                    stylized,
                    f"✓ {style_name} completed! (seed: {seed_used})",
                    seed_used
                )
            else:
                error_msg = result.get("error", "Unknown error")
                return None, None, None, None, f"Error: {error_msg}", 0

        except Exception as e:
            logger.error(f"Style generation failed: {e}")
            return None, None, None, None, f"Error: {str(e)}", 0

    def _3d_style_generate_core(
        self,
        image: Image.Image,
        style_key: str,
        is_blend: bool,
        strength: float,
        guidance_scale: float,
        num_steps: int,
        custom_prompt: str,
        seed: int,
        face_restore: bool = False
    ) -> dict:
        """Core style transfer generation"""
        return self.style_engine.generate_all_outputs(
            image=image,
            style_key=style_key,
            strength=float(strength),
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_steps),
            custom_prompt=custom_prompt if custom_prompt else "",
            seed=seed,
            is_blend=is_blend,
            face_restore=face_restore
        )

    def _cleanup_3d_memory(self) -> str:
        """Clean up 3D engine memory"""
        self.style_engine.unload_model()
        return "Memory cleaned!"

