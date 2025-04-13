import torch
if not hasattr(torch, 'float8_e4m3fn'):
    torch.float8_e4m3fn = torch.float32

try:
    import huggingface_patch
    print("huggingface_hub patch has been applied")
except ImportError:
    print("Warning: Failed to import the huggingface_patch module")

from PIL import Image, ImageEnhance
import numpy as np
import gradio as gr
from diffusers import StableDiffusionImg2ImgPipeline
import time
import os
import base64
import spaces
from io import BytesIO

class DogStyleTransfer:
    """
    Class for handling dog image style transfer using Stable Diffusion.
    This class manages model loading, image preprocessing, and style transfer operations.
    """
    def __init__(self):
        self.models = {}
        self.device = 'cpu'

        # Check xformers availability
        self.xformers_available = False
        try:
            import xformers
            self.xformers_available = True
            print(f"xformers {xformers.__version__} is available and will be used for memory-efficient attention")
        except ImportError:
            print("xformers not found - will use default attention mechanism")
        except Exception as e:
            print(f"Error checking xformers: {str(e)} - will use default attention mechanism")

        # Define style to model mapping based on availability
        if self.device == "cuda":
            self.style_model_mapping = {
                "Japanese Anime Style": "Linaqruf/anything-v3.0",
                "Classic Cartoon": "nitrosocke/mo-di-diffusion",
                "Oil Painting": "runwayml/stable-diffusion-v1-5",
                "Watercolor": "dreamlike-art/dreamlike-photoreal-2.0",
                "Cyberpunk": "dreamlike-art/dreamlike-diffusion-1.0"
            }
        else:
            # Lightweight models for CPU mode
            self.style_model_mapping = {
                "Japanese Anime Style": "runwayml/stable-diffusion-v1-5",
                "Classic Cartoon": "runwayml/stable-diffusion-v1-5",
                "Oil Painting": "runwayml/stable-diffusion-v1-5",
                "Watercolor": "runwayml/stable-diffusion-v1-5",
                "Cyberpunk": "runwayml/stable-diffusion-v1-5"
            }

        # style prompts with each feature
        self.style_prompts = {
            "Japanese Anime Style": "masterpiece, highest quality, genuine anime style illustration of a (dog:1.5), (bold anime aesthetics:1.5), (vibrant saturated colors:1.3), clean distinct lineart, stylized simplified features, expressive anime eyes, (preserve exact animal species:1.8), (maintain original animal breed:1.7), distinctive animal characteristics, (iconic anime art style:1.4), dramatic shading, flat color areas with highlight accents, simplified background elements, characteristic anime proportions, retain animal identity while stylizing, professional anime production quality, no watermarks, no signatures, (do not change animal species:1.8)",
            "Classic Cartoon": "masterpiece, highest quality classic cartoon illustration of a dog, (golden age animation style:1.3), hand-drawn cel animation quality, bold clean outlines, (vibrant solid color fills:1.2), exaggerated expressive features, playful animated poses, classic Disney/Pixar influenced design, professional animation studio quality, simplified but expressive details, perfect smooth linework, rounded stylized forms, cheerful color palette, dynamic motion lines, classic cartoon physics, expressive oversized eyes, joyful personality captured, squash and stretch principles applied, classic cartoon proportions, professional character design, perfect animation keyframe quality, appealing character expression, masterful use of simple shapes, iconic cartoon aesthetic, no watermarks, no signatures",
            "Oil Painting": "masterpiece, museum quality oil painting of a dog, (impasto technique:1.3), visible textured brushstrokes, layered oil pigments, rich depth of color, classical composition, (dramatic chiaroscuro lighting:1.2), Renaissance painting technique, glazing layers, sophisticated color harmony, warm and cool tones balance, expert painterly details, canvas texture visible, traditional realistic portrait style, fine art quality, gallery exhibition standard, rich shadows and highlights, volumetric form definition, atmospheric perspective, professional oil painting techniques, traditional varnished finish, color complexity with subtle undertones, expertly captured fur textures, strong compositional focus, emotional depth, timeless artistic quality, no watermarks, no signatures",
            "Watercolor": "masterpiece, highest quality watercolor painting of a dog, (wet-on-wet technique:1.3), flowing color blends, translucent paint layers, visible paper texture, (controlled paint blooms:1.2), delicate color washes, spontaneous paint flow, preserved white spaces, soft color bleeding effects, subtle granulation textures, feathered edges, luminous transparency, loose expressive brushwork, artistic color pooling, gradient color transitions, minimalist background, playful splatter accents, artistic negative space usage, light-filled composition, watercolor paper grain visible, atmospheric color diffusion, professional traditional watercolor techniques, delicate brush details combined with flowing textures, no watermarks, no signatures",
            "Cyberpunk": "masterpiece, highest quality, hyper-detailed cyberpunk digital art of a dog, (advanced technological integration:1.4), holographic collar interface, bionic limb enhancements, neural implant visuals, data visualization overlay, augmented reality HUD elements, (neon light reflections:1.3), wet street reflections, volumetric fog effects, urban dystopian background, megacity skyline, glowing circuitry details, optical fiber accents, synthetic materials, dramatic neon-lit contrast, cybernetic enhancements, high tech visors, digital distortion effects, information flow visualization, glitchy textures, metallic surfaces with advanced patina, dark atmospheric tone with vibrant neon accents, electrical energy effects, retro-futuristic design elements, near-future technology aesthetic, no watermarks, no signatures"
        }

        # Feature preservation prompts with weighted emphasis
        self.feature_preservation = {
            "common": "faithful representation of original animal species:(1.6), preserve original animal face structure:(1.5), maintain exact species characteristics:(1.4), accurate distinctive features:(1.3), consistent anatomical structure:(1.2), recognizable animal identity",
            "Japanese Anime Style": "anime style dog with preserved realistic proportions, distinctive dog breed characteristics maintained, dog facial features clearly recognizable",
            "Classic Cartoon": "cartoon style with accurate dog proportions, characteristic breed features preserved, recognizable dog expressions",
            "Oil Painting": "oil painting technique while maintaining anatomical accuracy, realistic dog proportions, distinctive breed characteristics",
            "Watercolor": "watercolor aesthetic with precise breed representation, accurate dog anatomy, distinctive dog features preserved",
            "Cyberpunk": "cyberpunk elements while maintaining accurate dog proportions, recognizable breed features, true-to-life dog expression"
        }

        # Negative prompts
        self.negative_prompts = {
            "common": "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limbs, missing limbs, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, watermark, signature, text, change of species, wrong animal species, incorrect animal type, different animal, human features",
            "dog_specific": "human face, human features, anthropomorphic, humanoid, human-like features, cartoon eyes, unrealistic eyes",
            "Japanese Anime Style": "photorealistic, 3d render, western cartoon style, pixar style, realistic textured skin",
            "Classic Cartoon": "anime style, manga, realistic, detailed skin texture, painterly, sketch, watercolor style",
            "Oil Painting": "flat colors, digital art, cartoon, cell shaded, smooth texture, anime style",
            "Watercolor": "digital art, 3d render, vector art, perfect linework, hard edges, bold lines",
            "Cyberpunk": "watercolor paint, oil painting, natural scene, traditional art, vintage style, soft colors",
            "species_preservation": "species transformation, change of animal type, incorrect animal features, wrong animal proportions, mixed animal characteristics"
        }

        # Style descriptions for UI display
        self.style_descriptions = {
            "Japanese Anime Style": "Characterized by vibrant colors, large expressive eyes, and stylized features common in Japanese animation.",
            "Classic Cartoon": "Friendly, rounded features with bold outlines and bright colors typical of classic animated films.",
            "Oil Painting": "Rich textures and depth created through visible brushstrokes and layered color application.",
            "Watercolor": "Soft, transparent washes of color with flowing transitions and subtle color blending.",
            "Cyberpunk": "Futuristic sci-fi aesthetic with neon colors, high contrast, and technological elements."
        }

        # Set model cache path
        self.model_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "dog_style_transfer")
        os.makedirs(self.model_cache_dir, exist_ok=True)

        # Display system info for debugging
        self._print_system_info()

    def _print_system_info(self):
        """Print system information for debugging purposes"""
        print("\n===== System Information =====")
        print(f"Device: {self.device}")
        print(f"PyTorch version: {torch.__version__}")

        if self.device == "cuda":
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'Unknown'}")
            print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB" if torch.cuda.is_available() else "Not available")

        print(f"xformers available: {self.xformers_available}")
        print("============================\n")

    @spaces.GPU
    def load_model(self, style_name):
        """Load the appropriate model based on style, handling xformers compatibility"""

        if not hasattr(self, '_cuda_initialized'):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._cuda_initialized = True

        # Get model ID for the style
        model_id = self.style_model_mapping.get(style_name, "runwayml/stable-diffusion-v1-5")

        # Check if model is already loaded
        if model_id not in self.models:
            print(f"Loading model {model_id} for {style_name} style...")

            try:
                # Load model with cache directory
                model = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_id,
                    cache_dir=self.model_cache_dir,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,  # Remove safety checker to improve speed
                )

                if self.device == "cuda":
                    model = model.to("cuda")
                    # Enable memory optimization
                    model.enable_attention_slicing()

                    # Try to enable xformers
                    try:
                        if hasattr(model, 'enable_xformers_memory_efficient_attention'):
                            print("Attempting to enable xformers memory efficient attention...")
                            model.enable_xformers_memory_efficient_attention()
                            print("xformers memory efficient attention enabled successfully!")
                    except Exception as e:
                        print(f"Warning: Could not enable xformers memory efficient attention: {e}")
                        print("Proceeding without xformers optimization - this may use more memory but should still work.")

                # Store model
                self.models[model_id] = model
                print(f"Model {model_id} loaded successfully!")
            except Exception as e:
                print(f"Error loading model {model_id}: {str(e)}")
                # Fall back to basic model if specific model fails
                if model_id != "runwayml/stable-diffusion-v1-5":
                    print("Falling back to default model...")
                    return self.load_model("Oil Painting")  # Use generic model as fallback
                raise

        return self.models[model_id]

    def preprocess_image(self, image, animal_type='dog'):
        """Enhanced preprocessing for dog images before style transfer"""
        # Convert to PIL image if needed
        if isinstance(image, np.ndarray):
            # Handle RGBA images by converting to RGB
            if image.shape[2] == 4:
                image = image[:, :, :3]
            image = Image.fromarray(np.uint8(image))

        # Resize while maintaining aspect ratio
        width, height = image.size
        max_size = 512  # SD models typically use 512x512 input
        scaling_factor = min(max_size / width, max_size / height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        image = image.resize((new_width, new_height), Image.LANCZOS)

        # Enhance contrast to emphasize dog features
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)  # Slightly enhance contrast

        # Sharpen to improve detail
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.3)  # Enhance sharpness

        # Pad if not 512x512, instead of cropping
        if new_width != 512 or new_height != 512:
            new_img = Image.new("RGB", (512, 512), (255, 255, 255))
            # Center the resized image
            offset = ((512 - new_width) // 2, (512 - new_height) // 2)
            new_img.paste(image, offset)
            image = new_img

        if animal_type != 'dog':
            self.feature_preservation['common'] = 'strict preservation of original animal species:(1.8),' + self.feature_preservation["common"]

        return image

    @spaces.GPU
    def transform_style(self, image, style_name, strength=0.75, guidance_scale=7.5):
        """
        Transform image to selected style with improved prompts and parameters
        Args:
            image: Input image
            style_name: Name of the style to apply
            strength: Style transformation strength (0-1)
            guidance_scale: Guidance scale for stable diffusion
        Returns:
            tuple: (transformed_image, error_message)
        """
        try:
            if image is None:
                return None, "Please upload a dog image first!"

            start_time = time.time()
            print(f"Starting style transfer: {style_name}")

            # Adjust parameters based on style
            if style_name == "Japanese Anime Style":
                guidance_scale = 9.0  # Higher guidance for anime style
                strength = 0.8
                num_steps = 50
            elif style_name == "Classic Cartoon":
                guidance_scale = 8.0
                strength = 0.75
                num_steps = 40
            elif style_name == "Oil Painting" or style_name == "Watercolor":
                guidance_scale = 8.0  # Medium guidance for art styles
                strength = 0.85
                num_steps = 50
            elif style_name == "Cyberpunk":
                guidance_scale = 10.0  # Very high guidance for cyberpunk
                strength = 0.85
                num_steps = 50
            else:
                num_steps = 40

            # Load model for style
            try:
                pipe = self.load_model(style_name)
            except Exception as e:
                print(f"Failed to load specific model for {style_name}: {str(e)}")
                # Fall back to default model
                pipe = self.load_model("Oil Painting")

            # Enhanced image preprocessing
            pil_image = self.preprocess_image(image)

            # Get style prompt and add feature preservation
            base_prompt = self.style_prompts.get(style_name, "digital art style, a dog")

            # Feature preservation prompts - combining common and style-specific
            feature_preservation = f"{self.feature_preservation['common']}, {self.feature_preservation.get(style_name, '')}"

            # Enhanced positive prompt with feature preservation
            prompt = f"{base_prompt}, {feature_preservation}, (high quality, detailed, sharp focus, professional photography):(1.2)"

            # Use negative prompt - combining common and style-specific
            negative_prompt = f"{self.negative_prompts['common']}, {self.negative_prompts['dog_specific']}, {self.negative_prompts.get(style_name, '')}"

            print(f"Using prompt: {prompt}")
            print(f"Using negative prompt: {negative_prompt}")
            print(f"Transformation parameters - Strength: {strength}, Guidance Scale: {guidance_scale}, Steps: {num_steps}")

            # Limit steps if too large to avoid memory issues
            if num_steps > 60 and self.device == "cuda":
                print("Reducing inference steps to save memory")
                num_steps = 60

            try:
                # Generate transformed image
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=pil_image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps
                ).images[0]

            except RuntimeError as e:
                # Handle CUDA out of memory errors
                if "CUDA out of memory" in str(e):
                    print("CUDA out of memory error, trying with reduced parameters")
                    # Retry with lower settings
                    return self._retry_with_lower_settings(pipe, prompt, negative_prompt, pil_image, strength, guidance_scale)
                else:
                    # Try without negative prompt
                    print(f"Error with negative prompt, retrying without it: {str(e)}")
                    try:
                        result = pipe(
                            prompt=prompt,
                            image=pil_image,
                            strength=strength,
                            guidance_scale=guidance_scale,
                            num_inference_steps=30  # Reduce steps
                        ).images[0]
                    except Exception as retry_error:
                        print(f"Retry also failed: {str(retry_error)}")
                        raise

            proc_time = time.time() - start_time
            print(f"Style transfer completed in {proc_time:.2f} seconds")

            return np.array(result), None

        except Exception as e:
            error_message = str(e)
            # Provide user-friendly error messages
            if "xformers" in error_message.lower():
                print(f"xformers related error: {error_message}")
                return None, "Style transfer error: xformers optimization unavailable, but functionality not affected. Please click 'Transform Style' button again to continue."
            elif "CUDA out of memory" in error_message:
                print(f"CUDA memory error: {error_message}")
                return None, "GPU memory insufficient. Try reducing parameters or using a smaller image."
            else:
                print(f"Error during style transfer: {error_message}")
                return None, f"Style transfer error: {error_message}"


    def _retry_with_lower_settings(self, pipe, prompt, negative_prompt, image, strength, guidance_scale):
        """Retry with lower settings when memory is insufficient"""
        try:
            # First attempt: Reduce inference steps
            print("Attempting with lower settings (steps=20)...")
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=20  # Significantly reduce steps
            ).images[0]
            return np.array(result), None

        except Exception as first_error:
            # Log first failure
            print(f"First retry attempt failed: {str(first_error)}")

            # Second attempt: Minimum settings
            try:
                print("Attempting with minimum settings (steps=15, strength=0.6)...")
                result = pipe(
                    prompt=prompt,
                    image=image,
                    strength=0.6,  # Lower strength
                    guidance_scale=7.0,  # Use standard setting
                    num_inference_steps=15  # Minimum steps
                ).images[0]
                return np.array(result), None

            except Exception as second_error:
                # Log all failures
                print(f"Second retry attempt also failed: {str(second_error)}")
                print("All retry attempts failed")

                # Return clear error message
                error_msg = f"Unable to complete style transfer, even with minimal settings: {str(second_error)}"
                return None, error_msg

    def get_available_styles(self):
        """Get all available style options"""
        return list(self.style_prompts.keys())

    def get_style_description(self, style_name):
        """Get description for a specific style"""
        return self.style_descriptions.get(style_name, "")

    def get_model_info(self, style_name):
        """Get the model information for a specific style"""
        model_id = self.style_model_mapping.get(style_name, "runwayml/stable-diffusion-v1-5")
        return f"Powered by: {model_id}"

    def get_image_download_link(self, image):
        """
        Generate a data URL for downloading the image
        Args:
            image: PIL Image or numpy array
        Returns:
            str: Base64 encoded data URL
        """
        if image is None:
            return None

        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(np.uint8(image))

        # Save image to bytes buffer
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/png;base64,{img_str}"

def create_style_transfer_tab(dog_style_transfer):
    """Create style transfer tab with UI components"""

    with gr.Column():
        gr.Markdown("""
        # 🎨 Dog Style Transformation
        Transform your dog photos into different artistic styles! Upload a dog picture, choose your preferred style, and create unique artwork.
        """)

        gr.HTML("""
            <div style="
                text-align: center;
                padding: 16px;
                margin: 16px 0;
                background: linear-gradient(to right, rgba(66, 153, 225, 0.1), rgba(72, 187, 120, 0.1));
                border-radius: 10px;
                border: 1px solid rgba(66, 153, 225, 0.2);
                box-shadow: 0 3px 10px rgba(0,0,123,0.1);
            ">
                <h3 style="color: #333;">🐶 Upload a dog photo and select an artistic style</h3>
                <p>After uploading your dog photo, the system will transform it into your chosen artistic style. Try different styles to create stunning effects!</p>
                <p>The system uses specialized models for each style to ensure the best results.</p>
                <p style="margin-top: 10px; padding: 8px; background-color: #fff9e6; border-left: 4px solid #ffd966; border-radius: 4px;"><b>⏱️ Patience is a virtue!</b> While AI is working its magic, your dog might have time to learn a new trick or two. The transformation can take up to 30 seconds, depending on how photogenic your furry friend is! 🐾</p>
                <p style="margin-top: 10px; padding: 8px; background-color: #e6f9ff; border-left: 4px solid #66c2ff; border-radius: 4px;"><b>🤫 A Little Secret:</b> Although I designed this tool for dogs, it can actually transform any photo! Portraits, landscapes, even your favorite teddy bear — feel free to try them all! Just don't tell the other dogs… they might get jealous! 😉</p>
                <p style="margin-top: 10px; padding: 8px; background-color: #e6f9e6; border-left: 4px solid #66cc77; border-radius: 4px;"><b>✨ Unlimited Creativity!</b> Sometimes, AI might surprise you with unexpected creative interpretations, adding unique colors or features to your image. ✨</p>
            </div>
            """)

        with gr.Row():
            with gr.Column(scale=1):
                # Upload image component
                input_image = gr.Image(
                    label="Upload Dog Photo",
                    type="numpy"
                )

                style_dropdown = gr.Dropdown(
                    choices=dog_style_transfer.get_available_styles(),
                    value=dog_style_transfer.get_available_styles()[0],
                    label="Select Artistic Style"
                )

                # Display style description
                style_description = gr.Markdown(
                    dog_style_transfer.get_style_description(dog_style_transfer.get_available_styles()[0])
                )

                # Display model info
                model_info = gr.Markdown(
                    dog_style_transfer.get_model_info(dog_style_transfer.get_available_styles()[0])
                )

                with gr.Row():
                    strength_slider = gr.Slider(
                        minimum=0.3,
                        maximum=0.9,
                        value=0.75,
                        step=0.05,
                        label="Style Intensity (lower values preserve more original details)"
                    )

                # customize Transform style buttom
                style_button = gr.Button("Transform Style", variant="primary")

                gr.Markdown("""
                <style>
                button.primary {
                    background: linear-gradient(90deg, #ff6b6b, #ffa36b, #ffd56b) !important;
                    color: white !important;
                    font-weight: 600 !important;
                    text-shadow: 0 1px 1px rgba(0,0,0,0.2) !important;
                    border: none !important;
                }
                button.primary:hover {
                    background: linear-gradient(90deg, #ff5b5b, #ff936b, #ffcf6b) !important;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
                }
                </style>
                """)

                # Progress indicator
                status_indicator = gr.Textbox(
                    label="Status",
                    value="Upload an image and press 'Transform Style' to begin",
                    interactive=False
                )

                error_output = gr.Textbox(
                    visible=False,
                    label="Error Message"
                )

            with gr.Column(scale=1):
                # Output image component
                output_image = gr.Image(
                    label="Style Transformation Result"
                )

                # Hidden component to store the download link
                download_link = gr.HTML(visible=False)

                # HTML component for actual download
                download_html = gr.HTML(visible=False)

                gr.Markdown("""
                ### Tips for Best Results
                - Use images with clear dog features and good lighting
                - For best results, use images where the dog is the main subject
                - Different styles work better with different dog breeds
                - Lower the style intensity to preserve more original details
                """)

        gr.HTML("""
            <style>
            .style-box {
                background: linear-gradient(145deg, #ffffff, #f5f7fa);
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                padding: 25px 30px;
                margin: 30px 0;
                border: 1px solid rgba(0,0,0,0.05);
                position: relative;
            }
            .style-box::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 6px;
                height: 100%;
                background: linear-gradient(to bottom, #ff6b6b, #ffa36b, #ffd56b);
                border-radius: 6px 0 0 6px;
            }
            .style-box h2 {
                color: #333;
                font-size: 24px;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #f0f0f0;
            }
            .style-name {
                font-weight: bold;
                color: #333;
            }
            .style-desc {
                margin-bottom: 15px;
                padding-bottom: 15px;
                border-bottom: 1px solid #f0f0f0;
            }
            .style-desc:last-child {
                margin-bottom: 0;
                padding-bottom: 0;
                border-bottom: none;
            }
            </style>
            <div class="style-box">
                <h2>Style Effect Descriptions</h2>
                <p>Each style transforms your dog photo in a unique way:</p>
                <div class="style-desc">
                    <p><span class="style-name">Japanese Anime Style:</span> Vibrant artwork with fluid animation qualities, expressive features, and dramatic lighting effects. Features soft color gradients, detailed line work, and emotional depth.</p>
                </div>
                <div class="style-desc">
                    <p><span class="style-name">Classic Cartoon:</span> Traditional animation style with bold outlines, solid color fills, and playful character design. Displays exaggerated expressions, simplified forms, and dynamic poses.</p>
                </div>
                <div class="style-desc">
                    <p><span class="style-name">Oil Painting:</span> Classical art technique with visible textured brushstrokes and layered color application. Shows rich depth, dramatic lighting contrast, and sophisticated color harmony.</p>
                </div>
                <div class="style-desc">
                    <p><span class="style-name">Watercolor:</span> Delicate painting style with flowing color blends and translucent layers. Features soft edges, color bleeding effects, and visible paper texture elements.</p>
                </div>
                <div class="style-desc">
                    <p><span class="style-name">Cyberpunk:</span> High-tech futuristic aesthetic with advanced technological elements and neon accents. Incorporates holographic interfaces, digital effects, and urban dystopian elements.</p>
                </div>
            </div>
            """)

        # Setup event triggers
        def update_progress(value, desc):
            """Update progress bar and description"""
            return gr.update(value=value), gr.update(value=desc)

        def process_style_transfer(image, style, strength):
            """Process style transfer and prepare download options"""
            if image is None:
                return (
                    None,
                    gr.update(visible=True, value="Please upload a dog image first!"),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(value="Upload an image and press 'Transform Style' to begin")
                )

            # Display processing status
            status_message = "Processing your image... This may take a moment."

            # Perform style transfer
            result, error = dog_style_transfer.transform_style(
                image,
                style,
                strength
            )

            if error:
                return (
                    None,
                    gr.update(visible=True, value=error),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(value="Error occurred. Please try again.")
                )

            # Generate download link for the image
            if result is not None:
                pil_image = Image.fromarray(result)
                download_data = dog_style_transfer.get_image_download_link(pil_image)
                download_html_content = f"""
                <style>
                .download-btn {{
                    display: inline-block;
                    background: linear-gradient(90deg, #3498db, #2ecc71);
                    color: white !important; /* 確保文字為白色 */
                    text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important; /* 增強文字陰影使白色更突出 */
                    font-weight: 700 !important; /* 加粗字體 */
                    padding: 12px 24px;
                    text-align: center;
                    text-decoration: none;
                    font-size: 16px;
                    border-radius: 25px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    border: none;
                    box-shadow: 0 3px 6px rgba(0,0,0,0.16);
                    letter-spacing: 0.5px; /* 字母間距，提高可讀性 */
                }}
                .download-btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 5px 10px rgba(0,0,0,0.25);
                    background: linear-gradient(90deg, #2980b9, #27ae60);
                }}
                .download-btn:active {{
                    transform: translateY(0);
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                </style>
                <a href="{download_data}" download="dog_{style.replace(' ', '_')}.png" class="download-btn">
                    Download Transformed Image
                </a>
                """

                # Store download data in a hidden element
                # We'll make this invisible to avoid showing base64 encoded data
                hidden_download_data = download_data

                return (
                    result,
                    gr.update(visible=False),
                    gr.update(visible=False, value=hidden_download_data),  # Keep the data hidden
                    gr.update(visible=True, value=download_html_content),  # Show the HTML button
                    gr.update(value="Transform Completed! You can download the image")
                )
            else:
                # Handle the case where result is None but no error was returned
                return (
                    None,
                    gr.update(visible=True, value="Style transfer failed with no specific error. Please try again."),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(value="Something went wrong. Please try again.")
                )

        # Update style description and model info
        def update_style_info(style):
            return dog_style_transfer.get_style_description(style), dog_style_transfer.get_model_info(style)

        style_button.click(
            fn=process_style_transfer,
            inputs=[input_image, style_dropdown, strength_slider],
            outputs=[
                output_image,
                error_output,
                download_link,
                download_html,
            ]
        )

        style_dropdown.change(
            fn=update_style_info,
            inputs=[style_dropdown],
            outputs=[style_description, model_info]
        )


        # Add example images
        example_dogs = [
            ["Border_Collie.jpg", "Japanese Anime Style"],
            ["Golden_Retriever.jpeg", "Classic Cartoon"],
            ["Saint_Bernard.jpeg", "Oil Painting"],
            ["Samoyed.jpeg", "Watercolor"],
            ["French_Bulldog.jpeg", "Cyberpunk"]
        ]

        # Check if Examples feature is available
        try:
            gr.Examples(
                examples=example_dogs,
                inputs=[input_image, style_dropdown]
            )
        except Exception as e:
            print(f"Note: Examples feature not available in your Gradio version: {e}")

        gr.HTML("""
            <style>
            .attribution-box {
                font-size: 0.85em;
                color: #666;
                margin-top: 20px;
                padding: 18px;
                border-radius: 8px;
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                font-style: italic;
            }
            .attribution-box h4 {
                margin-top: 0;
                color: #495057;
                font-style: normal;
                font-weight: 600;
                margin-bottom: 12px;
            }
            .attribution-box p {
                margin: 8px 0;
                line-height: 1.5;
            }
            </style>
            <div class="attribution-box">
                <h4>Attribution</h4>
                <p>This application uses pre-trained diffusion models from Hugging Face for image style transfer. All models are used according to their respective open source licenses for educational and non-commercial purposes.</p>
                <p>Powered by the open source Diffusers library from Hugging Face.</p>
            </div>
            """)

    return input_image, style_dropdown, style_button, output_image
