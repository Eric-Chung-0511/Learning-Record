# 🛰️ Vision Scout 🔍

Welcome aboard **Vision Scout**! 🚀 This isn't just another tool that spots objects in pictures. Vision Scout digs deeper, aiming to **understand the whole story an image tells**. I blend the sharp eyes of **YOLOv8** for object detection, the contextual smarts of **OpenAI's CLIP** for semantic understanding, the scene classification power of **Places365**, and the advanced narrative capabilities of **Meta's Llama 3.2** to bring it all together. The result? A system that can figure out the scene type (is it a crossroads or a quiet bedroom?), gauge the lighting, map out how things are arranged, guess what might be happening, and even point out things to watch out for, all presented in a cohesive, human-like story. It's all wrapped up in a friendly Gradio web app.

**Why Vision Scout?**

While many tools detect objects, Vision Scout offers more:
* **Multi-Modal Intelligent Fusion:** It doesn't just *see* objects (YOLOv8), it understands the *context* and *vibe* (CLIP), establishes environmental baseline with comprehensive scene classification (Places365), and smartly combines all three perspectives for superior scene identification. This triple-layer approach ensures robust understanding whether the scene is defined by specific objects, overall atmosphere, or fundamental environmental characteristics.
* **Rich Narratives, Elevated by LLM:** Forget simple tags. Vision Scout, enhanced by **Llama 3.2**, crafts detailed, human-like descriptions that truly tell the story of the scene, making complex visual information accessible and engaging.
* **Spatial Smarts:** It figures out where things are in relation to each other, identifying functional zones like 'dining areas' or 'workspaces'.
* **From Snapshots to Stories Over Time**: Beyond single images, Vision Scout now handles videos, tracking objects and analyzing scene evolution across frames. Whether it's people moving through a plaza or traffic flowing at an intersection, Vision Scout connects the dots over time, not just space. The integration of Llama also paves the way for more sophisticated video summaries and event highlighting in the future.

---

## ⚠️ Important Notice
**Due to a current GitHub issue with rendering Jupyter notebooks (missing 'state' key in metadata.widgets), the notebook code and outputs may not display properly in this repository.**

For the complete notebook with all outputs and visualizations, please access the project via this Google Colab link:  
👉 [View Complete Project in Google Colab](https://colab.research.google.com/drive/1YTgAYIO9mWxxNY82-shf6wYK7AerrnR2?usp=sharing)

The issue is being tracked by GitHub and will be resolved as soon as possible. Thank you for your understanding!

---

## 🎯 What Vision Scout Offers

Think of Vision Scout as your AI companion for analyzing images. Here's a glimpse of what it brings to the table:

* **Sharp Object Spotting:** Uses **YOLOv8** (pick 'n', 'm', or 'x' for speed vs. accuracy) to find objects quickly and reliably. Furthermore, you can control the confidence threshold and filter *for* specific object types (like people or vehicles), meaning *only* those selected objects will be displayed in the results. This filtering can be helpful when focusing on particular elements in a scene.

* **Clear Statistics & Visualization:** Provides clear statistics and a visual bar chart summarizing what objects were detected and how many of each type were found.

* **Scene Foundation with Places365:** The **Places365** model provides the environmental baseline, classifying scenes across 365 categories from parks to kitchens to streets. This ResNet50-based foundation gives the system its initial understanding of the setting, which then guides and enhances all other analytical components.

* **Understanding the Vibe with CLIP:** This is where Vision Scout truly shines. **CLIP** grasps the image's overall meaning by comparing it to descriptions covering everything from 'city street' vs 'living room' to 'daytime clear' vs 'neon night', understanding context objects alone can't provide. It's also employed for zero-shot landmark recognition, identifying famous locations and architectural features that might not be caught by standard object detection.

* **Smarter Scene Classification (Multi-Modal Fusion):** Vision Scout intelligently **combines** YOLOv8's object data, CLIP's contextual understanding, and Places365's scene classification through dynamic weight fusion. This *hybrid scoring* improves scene identification by cleverly weighting evidence—leaning more on YOLO for object-defined scenes (like kitchens) and more on CLIP for atmosphere-defined scenes (like night markets), while using Places365 as the foundational environmental context.

* **Telling the Story - Rich & Refined Descriptions with LLM:** Instead of just a list, the `EnhancedSceneDescriber` crafts an initial detailed narrative. Then, the **`LLMEnhancer` (powered by Llama 3.2)** takes this further, refining the story for superior fluency and contextual depth. It weaves together the scene type, objects, lighting, viewpoint, functional zones, and even cultural hints into a highly readable and insightful paragraph, ensuring factual consistency with the visual evidence.

* **Mapping the Scene - Spatial Awareness:** The `SpatialAnalyzer` looks at object positions, maps them to image regions (top-left, bottom-right, etc.), and identifies **functional zones** (like spotting chairs and a table as a 'dining area').

* **Reading the Light - Enhanced Lighting Analysis:** Bright day, dim room, or neon night? The `LightingAnalyzer` employs a sophisticated approach, analyzing brightness, colors, and contrast while incorporating guidance from Places365's indoor/outdoor classification to determine precise **lighting conditions**. This Places365-guided enhancement ensures more accurate lighting assessment by considering the environmental context.

* **Reading Between the Lines - Activity, Safety & LLM Verification:** Based on the scene and objects (like people near traffic or sharp items in a kitchen), the system suggests **likely activities** and flags potential **safety points**. Furthermore, if conflicting interpretations arise between object detection and semantic analysis, the **LLM** can be optionally invoked to **verify and reconcile** these differences, offering an additional layer of intelligent assessment to the scene understanding.

* **Video Understanding**: Upload a video, and the system will extract frames at your chosen interval and run object detection on each frame. It provides frame-by-frame statistics and an overall summary in structured JSON format, showing which object classes appeared and how often. While object tracking and scene description updates are not yet implemented, the current architecture is designed to support future enhancements such as multi-frame tracking, dynamic scene understanding, and LLM-based video summarization.

* **An Interface for Exploration:** All this analysis is presented through a **Gradio web app**. Easily upload images, tweak settings, and view results in organized tabs (annotated image, stats, full report). The LLM-enhanced descriptions are clearly marked, allowing you to appreciate the AI's advanced narrative capabilities.

---
## 🧠 How It All Comes Together: The Vision Scout Flow

Vision Scout works like a team of AI specialists, each examining your image from their unique perspective before coming together to tell the complete story. It's a collaborative process where different components analyze in parallel, then intelligently combine their insights. The flowchart below shows how this all unfolds, followed by a detailed walkthrough:

<p align="center">
  <img src="https://github.com/Eric-Chung-0511/Learning-Record/blob/main/Data%20Science%20Projects/VisionScout/Process_diagram_02.svg" width="800">
</p>

**The Complete Journey**

1.  **Image Input and Preprocessing**
    When you upload an image, Vision Scout first ensures it's in the right format for analysis. Whether you provide a PIL Image, Numpy array, or other format, the system converts and temporarily stores it, setting the stage for all the analytical work ahead.

2.  **Multi-Modal Perception: The Initial Assessment**
    This is where the magic starts. Your image gets sent down multiple analytical pathways simultaneously, like having several experts examine the same scene at once:

    * **Places365 Scene Classification** jumps in first with a ResNet50 model that's been trained on 365 different scene categories. Think of it as the generalist who takes one look and says "this feels like a kitchen" or "looks like a park to me." It provides that crucial first impression along with a confidence score, plus an early guess about whether we're looking at an indoor or outdoor scene.
    * At the same time, **YOLOv8 Object Detection** gets busy with the detailed inventory work. It's scanning the image to identify specific objects and exactly where they're located, complete with bounding boxes and confidence levels for each detection. This gives us the concrete "what's actually in this picture" foundation that everything else builds on.

3.  **Feature Enhancement and Deep Analysis**
    Now things get really interesting as the system layers on more sophisticated analysis:

    * The **Lighting Analysis** component dives deep into the visual characteristics, examining brightness levels, color distributions, and textures to figure out the lighting conditions. Is this a sunny day? A dimly lit indoor room? Night scene with artificial lighting? It's smart enough to consider that initial indoor/outdoor hint from Places365 to refine its assessment.
    * **Spatial Object Mapping** takes the object locations from YOLO and maps them onto a spatial grid, understanding how things are arranged across different regions of the image. This spatial awareness becomes crucial for identifying functional zones later on.
    * **Landmark Recognition** uses CLIP's zero-shot classification capabilities to spot famous landmarks. It's particularly clever about this, sometimes examining areas that YOLO missed or had low confidence about, and it can even perform multi-scale analysis when needed.
    * **Semantic Scene Analysis** employs CLIP to create a unique "fingerprint" of your image, then compares this against hundreds of text descriptions to understand the overall meaning and context. This captures things like atmosphere, cultural context, and subtle visual cues that pure object detection might miss.

4.  **Multi-Dimensional Scoring Engine**
    Here's where the different analytical streams start coming together. The system runs three parallel scoring mechanisms:

    * **YOLO-based scoring** focuses on object-driven scene assessment, **CLIP-based scoring** emphasizes semantic and contextual understanding, while **Places365-based scoring** provides classification confidence. Each approach brings its own perspective to the table.

5.  **Dynamic Weight Fusion: The Intelligent Decision**
    The system acts like a smart conductor, intelligently combining all these different viewpoints. It doesn't just average the scores; instead, it dynamically adjusts the importance of each analytical component based on the scene type. For example, a kitchen scene heavily relies on YOLO detecting specific appliances, while a bustling market scene might lean more on CLIP's understanding of atmosphere and cultural context.

6.  **Context Reasoning Engine**
    With the main scene identified, Vision Scout adds layers of contextual intelligence:

    * **Functional Zone Identification** analyzes the spatial layout to identify specific areas within the scene. In an office, it might spot workstation areas where desks and chairs cluster together. In a street scene, it could delineate pedestrian walkways from vehicle traffic zones.
    * **Activity and Safety Inference** consults the system's knowledge base to infer probable activities and flag potential safety concerns. This draws from templates and definitions that understand what typically happens in different environments and what risks might be present.

7.  **Natural Language Generation Pipeline**
    This is where all the analytical insights transform into human-readable narrative:

    * **Template-Based Scene Description** serves as the initial storyteller, gathering all the insights about scene type, key objects, lighting conditions, spatial layout, inferred activities, and safety notes. It weaves these elements together using sophisticated templates to create a comprehensive initial description.
    * **Factual Verification System** performs consistency and accuracy checks, ensuring that the generated content aligns with the actual analytical findings and resolving any conflicts between different components.
    * **LLM Enhancement Process** uses Llama 3.2 to refine and polish the narrative, making it more fluent and insightful while strictly maintaining factual accuracy. The LLM receives specific instructions to only reference objects that were actually detected and to respect quantities and other verified facts.

8.  **Structured Output Assembly and Presentation**
    Finally, the complete analysis gets organized and presented through the Gradio Interface. You'll see your image with bounding boxes around detected objects, detailed statistics about what was found, and the rich, refined scene understanding report. The interface clearly indicates when descriptions have been LLM-enhanced and often provides options to compare with the original template-based description.

The beauty of this system lies in how all these components work together, each contributing their specialty while the intelligent fusion process ensures you get the most accurate and insightful understanding of your image possible.

---

## 🛠️ Under the Hood: Key Models & Components

This project relies on a mix of cutting-edge models and custom logic:

* **YOLOv8 (Ultralytics):** Our object detection powerhouse, known for its speed/accuracy balance. It identifies objects, draws bounding boxes, assigns COCO class labels, and gives confidence scores. Handled by `DetectionModel`.

* **Places365 (MIT):** The environmental foundation of our system, employing a ResNet50 model trained on 365 distinct scene categories. This component provides the crucial baseline scene classification that guides all subsequent analysis. The `Places365Model` delivers initial scene labels with confidence scores and indoor/outdoor probability estimates, serving as the environmental anchor that helps refine lighting analysis and contextual understanding throughout the processing pipeline.

* **CLIP (OpenAI):** The core of our semantic understanding. Learns from image-text pairs, embedding images and text into a shared space. This allows measuring similarity between an image and descriptions (like "a photo of a busy city street at night"). The `CLIPAnalyzer` uses this with prompts from `clip_prompts.py` for zero-shot reasoning about scenes, lighting, and contextual elements. Additionally, CLIP powers our zero-shot landmark recognition through the `CLIPZeroShotClassifier`, enabling identification of famous landmarks and architectural features that traditional object detection might miss, with sophisticated multi-scale analysis and adaptive confidence thresholds.

* **Llama 3.2 (Meta):** Our advanced language processing and reasoning engine, specifically `meta-llama/Llama-3.2-3B-Instruct`. The `LLMEnhancer` leverages this Large Language Model (LLM) to significantly upgrade the final scene description. It takes structured analytical data and the initial template-based narrative, then refines it for superior fluency, contextual richness, and human-like articulation. Crucially, it's also employed to verify and reconcile potential discrepancies between YOLO's object-centric view, CLIP's semantic interpretation, and Places365's environmental classification, ensuring a more robust and accurate final understanding.

* **`LightingAnalyzer`:** Custom module analyzing pixel statistics (brightness, color distribution, contrast) to determine lighting conditions ('day_clear', 'indoor_dim') and indoor/outdoor probability. This component intelligently incorporates guidance from Places365's environmental classification to enhance accuracy, using the scene context to refine lighting assessment and resolve ambiguous cases where visual characteristics alone might be insufficient.

* **`SpatialAnalyzer`:** Custom component mapping YOLO detection boxes to a spatial grid system, analyzing object density, distribution patterns, and co-occurrence relationships to infer **functional zones** within scenes. This spatial intelligence enables identification of specific areas like workstations, dining spaces, or traffic zones based on object clustering and arrangement patterns.

* **`SceneAnalyzer`:** The central coordination hub managing the entire analytical workflow. It orchestrates parallel processing streams, calculates scoring from multiple sources (YOLO object-based, CLIP semantic-based, and Places365 classification-based), performs the crucial **multi-modal fusion** with dynamic adaptive weighting, and determines the final scene classification through intelligent combination of all analytical perspectives.

* **`EnhancedSceneDescriber`:** Transforms structured analytical data into compelling natural language narratives using sophisticated template systems (from `scene_detail_templates.py`, `object_template_fillers.py`, and related files). This component incorporates all analytical dimensions including scene type, object inventories, lighting conditions, spatial arrangements, functional zones, and contextual elements to generate the comprehensive initial narrative before LLM enhancement.

* **Knowledge Base (`.py` files):** A comprehensive collection of Python files (`scene_type.py`, `object_categories.py`, `clip_prompts.py`, `landmark_data.py`, various `*_templates.py`) that serves as the system's knowledge repository and operational rulebook. These files define scene taxonomies, object categorizations, CLIP prompt libraries for semantic analysis, landmark databases for recognition, and template systems for narrative generation and activity/safety inference. This modular knowledge architecture ensures system adaptability and enables sophisticated contextual reasoning across diverse scenarios.

---

## 🚀 Try It Online 

The easiest way to try Vision Scout is via the deployed Hugging Face Space:

➡️ **[Try Vision Scout Live on HuggingFace Spaces](https://huggingface.co/spaces/DawnC/VisionScout)**

---

## 🌱 Future Directions

There's always more to explore! Here are some ideas for where Vision Scout could go next:

* **Dynamic and Temporal Analysis**: Extend beyond static analysis by implementing more advanced video understanding—such as temporal consistency checks, motion pattern recognition, multi-frame activity inference, and higher-level object interactions over time. This would enable the system to not just describe scenes frame-by-frame, but to understand evolving stories, like detecting group behavior, spotting anomalies, or summarizing key events from a full video.

* **Domain Specialization:**  Fine-tuning the core models (YOLOv8, potentially CLIP) on domain-specific data could significantly boost performance for targeted applications like indoor navigation assistance or retail shelf analysis.
  
* **Broader Knowledge and Context:** Continuously expand the internal "Knowledge Base" (the `.py` definition and template files). This involves adding definitions for a wider variety of scene types, recognizing more diverse objects, understanding more complex activities, and incorporating richer cultural contexts to make the analysis more globally applicable and robust.
  
* **Expanded Object Recognition:** Improve core object detection capabilities by training or fine-tuning YOLOv8 on custom datasets. This would allow Vision Scout to recognize specific objects relevant to niche domains that go beyond the standard 80 COCO classes, leading to more accurate and specialized analyses where needed.

---

## 🌍 The Vision Behind Vision Scout

Vision Scout started with a simple thought:  
**What if AI could help people understand the world they can’t see?**

For those who are visually impaired, whether from birth or due to illness or injury, daily life can be filled with unseen challenges. Vision is often called "the window to the soul," and I began to wonder if technology could offer a different kind of window — one that communicates.

I wanted to see if AI could go further than just recognizing objects.  
Could it understand what’s happening in a scene, how things are arranged, what the atmosphere feels like, and express that in clear, human-like language?

At the same time, I saw value in helping fields like public safety.  
Imagine if AI could notice details people might miss, or help investigators piece together the story behind a place just by analyzing how it's structured and lit.

Vision Scout is my way of bringing these ideas together.  
It’s not just about detecting *what’s there*, but understanding *what it means*.

---


## 📜 License
![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-lightgray)

© 2025 Eric Chung. This project is licensed under the Apache License 2.0, a permissive open source license that enables broad usage while ensuring proper attribution to the original author.

For detailed terms and conditions, please refer to the [LICENSE](./LICENSE.md) file.

---

## 🙏 Acknowledgements

* Thanks to **Ultralytics** for the powerful and easy-to-use **YOLOv8** object detection model. More information and the implementation can be found at the [Ultralytics YOLOv8 Repository](https://github.com/ultralytics/ultralytics).

* Big thanks to **OpenAI** for the revolutionary **CLIP** model which enables the semantic understanding capabilities of this project. Paper: Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020).

* A huge shout out to **Meta AI** for the **Llama 3.2** model (`meta-llama/Llama-3.2-3B-Instruct`), which significantly elevates the natural language understanding and generation in Vision Scout. This powerful LLM allows for more nuanced, fluent, and contextually accurate scene descriptions. More information about the Llama family of models can typically be found on the [Meta AI Blog](https://ai.meta.com/blog/) or their official model cards on [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
