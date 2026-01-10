import gradio as gr
from PIL import Image
from typing import Tuple, Optional, Dict, Any
import os
import logging

from FlowFacade import FlowFacade
from BackgroundEngine import BackgroundEngine
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
    def __init__(self, facade: FlowFacade, background_engine: BackgroundEngine):
        self.facade = facade
        self.background_engine = background_engine
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
                
                # Tab 1: Image to Video (Original Functionality)
                with gr.Tab("🎬 Image to Video"):
                    self._create_i2v_tab()
                
                # Tab 2: Background Generation (New Feature)
                with gr.Tab("🎨 Background Generation"):
                    self._create_background_tab()

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
                            • <strong>Feather Radius:</strong> Use 5-10 for complex scenes with fine details (hair, fur, foliage). 0 = sharp edges for clean portraits.<br>
                            • <strong>Mask Preview:</strong> Check the "Mask Preview" tab after generation. White = kept, Black = replaced. Helps diagnose edge issues.
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
                        <strong>⏱️ First-time users:</strong> Initial model loading takes 1-2 minutes.
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
                feather_radius_slider
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

    def _generate_background_handler(
        self,
        image: Image.Image,
        prompt: str,
        combination_mode: str,
        focus_mode: str,
        negative_prompt: str,
        steps: int,
        guidance: float,
        feather_radius: int
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
                negative_prompt, steps, guidance, feather_radius
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
        feather_radius: int
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
            feather_radius=int(feather_radius)
        )

        return result

