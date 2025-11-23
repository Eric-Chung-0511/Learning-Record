# 🎨 SceneWeaver - AI-Powered Background Generation & Image Composition

> **Professional studio-quality image composition at your fingertips.**

SceneWeaver is an intelligent background generation system that transforms ordinary portraits into professional studio shots through seamless foreground-background integration. Whether you're a photographer, content creator, or digital artist, SceneWeaver eliminates the need for manual editing while delivering results that rival professional photo studios.

---

## ✨ Overview

Creating professional-quality portraits typically requires expensive studio equipment, skilled photographers, and hours of post-production work. SceneWeaver democratizes this process by leveraging cutting-edge AI technology to generate photorealistic backgrounds and blend them naturally with your subjects. The system handles the complexity of foreground extraction, background generation, and seamless composition automatically.

### What Makes SceneWeaver Special?

SceneWeaver addresses a critical challenge in image composition: achieving clean, professional-looking edges where foreground meets background. Traditional background replacement tools often produce visible halos, color bleeding, and unnatural transitions that immediately betray digital manipulation. SceneWeaver solves these problems through sophisticated multi-model architecture that combines deep learning segmentation with advanced color processing algorithms.

The system employs **BiRefNet** and **U²-Net** for precise foreground extraction, even handling difficult cases like dark clothing and cartoon characters that confound simpler approaches. **Stable Diffusion XL** generates high-quality, photorealistic backgrounds tailored to your creative vision. Custom Lab color space processing eliminates color contamination at edges through adaptive spill suppression and chroma correction. **OpenCLIP** provides intelligent image analysis that enhances prompt engineering for contextually appropriate backgrounds.

This integrated approach creates a system that doesn't just swap backgrounds—it understands lighting, color harmony, and visual aesthetics to produce compositions that look naturally photographed rather than digitally assembled. SceneWeaver supports both automated processing with 24 curated scene templates and custom background generation through natural language descriptions.

---

## 🎯 Core Capabilities

**Intelligent Foreground Detection** forms the foundation of SceneWeaver's processing pipeline. The system employs a three-tier detection strategy that prioritizes quality while ensuring robustness. BiRefNet provides state-of-the-art segmentation accuracy with clean edge detection for most images. When BiRefNet encounters challenging scenarios, the system automatically falls back to U²-Net through rembg integration, which offers broader compatibility across image types. For maximum reliability, traditional computer vision methods using gradient-based edge detection serve as the final fallback layer. This hierarchical approach guarantees successful foreground extraction regardless of image complexity.

**Advanced Edge Processing** represents SceneWeaver's most sophisticated technical achievement. The system recognizes that edge quality determines whether composition appears professional or amateurish. Lab color space manipulation enables perceptual color correction that accounts for how humans actually perceive color relationships. Adaptive spill suppression removes contamination from original backgrounds without affecting genuine foreground colors. Multi-scale edge refinement processes boundaries at different resolutions to handle both fine details and broader transitions. Guided filtering preserves edge sharpness while smoothing artifacts. Alpha channel binarization eliminates semi-transparent halos that create visible seams. Core foreground protection prevents background influences from affecting faces and bodies.

**AI Scene Generation** leverages Stable Diffusion XL to create photorealistic backgrounds that match your creative vision. The system goes beyond simple text-to-image generation by incorporating intelligent prompt enhancement. OpenCLIP analyzes uploaded images to understand color temperature, brightness levels, and subject types. This analysis automatically enriches user prompts with appropriate lighting descriptors, atmospheric elements, and quality modifiers. Twenty-four curated scene templates span professional offices, natural landscapes, urban environments, artistic styles, and seasonal themes. Each template includes optimized prompts, negative prompts, and guidance scale parameters tuned for specific aesthetic results. Users can also describe custom backgrounds in natural language for unlimited creative possibilities.

**Production-Grade Blending** ensures foreground and background integrate seamlessly through multiple processing stages. Mask erosion removes contaminated edge pixels where original background colors have bled into foreground boundaries. Chroma vector deprojection mathematically removes background color influence from semi-transparent edge regions. Luminance matching adapts foreground lighting to harmonize with background illumination. Multiple correction passes address stubborn color contamination that resists single-pass processing. Inpainting repair handles remaining artifacts through context-aware pixel reconstruction. The result is a composition where viewers cannot identify the join between foreground and background elements.

**Memory-Optimized Architecture** enables deployment on resource-constrained environments like Google Colab. The system employs aggressive memory management strategies including lazy model loading that defers initialization until needed, LRU caching for model reuse across generations, ultra-aggressive cleanup between operations, sequential CPU offloading for low-memory scenarios, and progressive image processing that builds results incrementally. These optimizations allow SceneWeaver to run on consumer-grade GPUs while maintaining generation quality.

---

## 🏗️ System Architecture

SceneWeaver's architecture reflects a modular design where specialized components handle distinct aspects of the image composition pipeline. Each module encapsulates specific functionality while maintaining clean interfaces that enable testing, debugging, and future enhancement.

### The Multi-Component System

**MaskGenerator** serves as the intelligent foreground extraction engine. The component implements a three-tier priority system where BiRefNet attempts initial segmentation, U²-Net provides robust fallback through rembg integration, and traditional gradient-based methods ensure universal compatibility. The module includes specialized processing for cartoon characters that distinguishes between black line art and genuine transparency. Guided filter integration enables edge-preserving smoothing that maintains boundary sharpness while reducing noise. Trimap generation defines foreground, background, and uncertain regions to guide subsequent processing. Scene focus modes allow either tight cropping around subjects or wider inclusion of surrounding objects.

**ImageBlender** handles the sophisticated color processing that distinguishes SceneWeaver from simpler background replacement tools. The component operates in Lab color space where perceptual uniformity enables more effective color correction than RGB processing. Multi-scale edge refinement analyzes boundaries at different resolutions to optimize both fine details and broader transitions. Adaptive strength mapping applies correction intensity based on pixel-specific contamination levels rather than uniform processing. Background color contamination removal identifies and replaces pixels that match original background colors within foreground regions. Core foreground protection ensures faces and bodies retain original colors without background influence. Edge cleanup passes eliminate residual semi-transparent artifacts through targeted binarization.

**SceneWeaverCore** orchestrates the entire generation and composition workflow. The component manages Stable Diffusion XL for background generation with DPM solver scheduling for efficient inference. OpenCLIP integration enables intelligent prompt enhancement through automatic analysis of color temperature, brightness, and subject type. Scene template management provides curated presets with optimized generation parameters. Memory management implements ultra-aggressive cleanup strategies that maintain stability on constrained hardware. Quality checking validates results through automated assessment of mask coverage, edge continuity, and color harmony. Progress tracking provides detailed feedback during lengthy generation operations.

**QualityChecker** provides automated validation of composition results. The component assesses mask coverage to ensure adequate foreground area. Edge continuity analysis detects gaps, discontinuities, and jagged boundaries. Color harmony evaluation measures perceptual compatibility between foreground and background using Lab space deltaE calculations. The quality scoring system produces numerical ratings that enable automated decision-making about result acceptability.

**UIManager** delivers a professional Gradio interface optimized for both desktop and mobile access. The component implements scene template selection through searchable dropdowns. Quick preview functionality provides fast foreground visualization using lightweight traditional methods. Advanced options expose generation parameters for expert users while maintaining simplicity for casual use. Results gallery organizes outputs through tabbed views of final composition, generated background, and processed original. Download functionality enables immediate access to completed images. Memory cleanup buttons provide manual control over resource management.

---

## 🚀 Try It Live

Experience SceneWeaver directly through the deployed Hugging Face Space without any installation required. Simply upload your portrait and describe your desired background to see AI-powered composition in action.

➡️ **[Try SceneWeaver on Hugging Face Spaces](https://huggingface.co/spaces/DawnC/SceneWeaver)**

---

## 💡 Use Cases

**Photography Studios** leverage SceneWeaver to expand creative options without requiring physical sets or extensive post-production. Portrait photographers generate diverse background options from a single shoot, offering clients variety without additional session costs. Product photographers create consistent backdrops that highlight merchandise while maintaining brand aesthetics. Event photographers provide attendees with fun background variations that enhance keepsake photos.

**Content Creation** represents a major application domain for the system. Social media influencers maintain visual consistency across posts by standardizing backgrounds while keeping subjects varied. YouTubers generate custom thumbnails with eye-catching backgrounds that improve click-through rates. Digital artists explore creative compositions by combining hand-drawn characters with AI-generated environments. Marketing teams produce professional-looking campaign imagery without expensive studio rentals.

**E-Commerce Applications** employ SceneWeaver to improve product presentation. Online retailers standardize product photos with consistent, professional backgrounds that enhance perceived quality. Marketplace sellers create appealing listings without photography expertise or equipment investment. Fashion brands showcase clothing in lifestyle contexts by compositing garment photos onto appropriate backgrounds.

**Personal Projects** benefit from the system's accessibility and ease of use. Families create holiday cards with seasonal backgrounds from casual snapshots. Pet owners generate adorable portraits of their animals in whimsical or elegant settings. Hobbyist photographers experiment with creative compositions without mastering complex editing software.

---

## 🌱 Future Directions

Several enhancement opportunities exist for extending SceneWeaver's capabilities while maintaining its core strengths. ControlNet Inpainting integration would enable context-aware background generation that respects composition guidelines. The system could analyze foreground pose and perspective to generate backgrounds with matching camera angles and lighting directions, producing more photorealistic results.

Multi-style batch generation would allow users to explore creative options efficiently. The system could produce multiple variations with different artistic styles, color palettes, or atmospheric conditions from a single input. This diversity mode would accelerate creative exploration by presenting a range of options rather than requiring iterative regeneration.

Enhanced memory optimization would support larger resolution outputs without sacrificing quality. Progressive rendering could build high-resolution images through multi-pass processing. Intelligent tiling could split large images into manageable sections while maintaining edge consistency across boundaries.

Cloud deployment on Hugging Face Spaces with ZeroGPU acceleration would make SceneWeaver accessible to broader audiences. The serverless architecture would eliminate installation requirements while providing adequate computational resources for high-quality generation. Queue management would handle concurrent users efficiently during peak demand periods.

---

## 🔧 Technical Stack

**Core Models:**
- Stable Diffusion XL (stabilityai/stable-diffusion-xl-base-1.0) - Background generation
- BiRefNet (ZhengPeng7/BiRefNet) - Precise foreground segmentation
- U²-Net via rembg - Robust backup segmentation
- OpenCLIP (ViT-B-32) - Image understanding and prompt enhancement

**Processing Libraries:**
- PyTorch 2.6.0 - Deep learning framework
- OpenCV - Computer vision operations
- Pillow - Image manipulation
- NumPy - Numerical processing

**Interface & Deployment:**
- Gradio 5.33.1 - Web interface
- Hugging Face Diffusers - Model management
- xformers - Memory optimization

---

## 📜 License

![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-lightgray)

© 2025 Eric Chung. This project is licensed under the Apache License 2.0, a permissive open source license that enables broad usage while ensuring proper attribution to the original author.

For detailed terms and conditions, please refer to the [LICENSE](https://github.com/Eric-Chung-0511/Learning-Record/blob/main/LICENSE.md) file.

---

## 🙏 Acknowledgements

Gratitude to **Stability AI** for **Stable Diffusion XL**, the foundation model enabling photorealistic background generation. Learn more at the [Stability AI Repository](https://github.com/Stability-AI/stablediffusion).

Thanks to **Zheng Peng** for **BiRefNet**, providing state-of-the-art salient object detection with clean edge extraction. Visit the [BiRefNet Repository](https://github.com/ZhengPeng7/BiRefNet).

Appreciation to the **OpenCLIP** team for enabling semantic image understanding through vision-language models. Explore more at the [OpenCLIP Repository](https://github.com/mlfoundations/open_clip).

Thanks to **Daniel Gatis** for **rembg**, delivering robust background removal capabilities. Check out the [rembg Repository](https://github.com/danielgatis/rembg).

Recognition to **Xuebin Qin et al.** for **U²-Net**, enabling reliable foreground-background segmentation across diverse image types.

Thanks to the **PyTorch**, **Gradio**, and **Hugging Face** teams for providing the foundational frameworks that make this project possible.

---
