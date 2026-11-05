# Project Requirements: Image to Structured SVG Converter

## 1. Project Overview

**Goal:** Develop a tool that converts uploaded raster images (PNG, WEBP, JPG, JPEG) into SVG files that follow a predefined SVG structure and formatting standard.

The generated SVG must follow the same structure, layer organization, class naming conventions, and color separation logic as the provided reference SVG files.

The primary objective is not simply to convert an image into SVG. The objective is to generate an SVG that is compatible with our existing system, which uses specific layer IDs and CSS classes to control different colors and patterns.

### Important Technical Requirements
- Do not use AI or external AI APIs for image conversion.
- Build our own image-to-SVG conversion pipeline.
- Use free and open-source tools and libraries only.
- No paid services, subscriptions, or API dependencies.
- The conversion process should run programmatically.
- The generated SVG must be editable and compatible with our existing product customization system.

---

## 2. Input and Output

### Input
Users should be able to upload an image in the following formats:
- PNG
- WEBP
- JPG
- JPEG

### Output
The tool should generate a structured SVG file that follows the predefined SVG format.

The output filename should match the original filename, with the extension changed to `.svg`.

### Example

**Example 1:**
- Input: `C:\Users\Giriraj-Yash\Downloads\S35-19.webp`
- Output: `C:\Users\Giriraj-Yash\Downloads\S35-19.svg`

**Example 2:**
- Input: `C:\Users\Giriraj-Yash\Downloads\S46-15.webp`
- Output: `C:\Users\Giriraj-Yash\Downloads\s46-15.svg`

The original image dimensions and aspect ratio should be preserved.

---

## 3. Reference Files

The following files will be provided to the development AI:

1. Original PNG/WEBP image to convert.
2. Three existing SVG files with similar designs.
3. A generated SVG example (`Generated.svg`) for reference.

The AI/developer must inspect all reference SVG files before implementing the conversion logic.

Analyze the reference files to understand:
- SVG structure and hierarchy.
- Layer organization.
- Group IDs.
- CSS classes.
- Color separation.
- Solid color elements.
- Pattern elements.
- Stroke and fill properties.
- Reusable SVG definitions.
- Layer visibility rules.

The existing SVG files should be treated as the source of truth for the required output format.

Do not simply copy an existing SVG and replace its image. The generated SVG must represent the actual uploaded image.

---

## 4. Required SVG Structure

The generated SVG must follow the following layer structure.

### Layer 1 – Color 1

```xml
<g id="Layer_1">
    <!-- Color 1 solid elements -->
</g>
```

- Contains all solid elements belonging to Color 1.
- Elements must use classes following the `st{0-6}_1` naming convention.

### Layer 3 – Color 2 Pattern + Color 3 Solid

```xml
<g id="Layer_3">
    <g id="Layer_3_2">
        <!-- Color 2 pattern elements -->
    </g>

    <g id="Layer_3_3">
        <!-- Color 3 solid elements -->
    </g>
</g>
```

Visibility rules:
- Show this layer when Color 2 is configured as a pattern.
- Hide this layer when Color 2 is configured as a solid color.

### Layer 2 – Color 2 Solid + Color 3 Solid

```xml
<g id="Layer_2">
    <g id="Layer_2_1_2">
        <!-- Color 2 solid elements -->
    </g>

    <g id="Layer_2_1_3">
        <!-- Color 3 solid elements -->
    </g>
</g>
```

Visibility rules:
- Show this layer when Color 2 is configured as a solid color.
- Hide this layer when Color 2 is configured as a pattern.

### Layer Rose

```xml
<g id="Layer_rose">
    <!-- Reusable rose elements -->
</g>
```

- Reuse the `Layer_rose` implementation from one of the provided reference SVG files.
- Preserve its structure and required definitions.
- Do not unnecessarily recreate or modify it.

---

## 5. CSS Class Naming Convention

Each SVG element must use the appropriate class according to its color and rendering type.

The class structure is:

`st{0-6}_{color}`

The first digit can be any number from 0 to 6. It does not have a specific functional meaning in our system.

| Class | Purpose |
|---|---|
| `st0_1` to `st6_1` | Color 1 solid elements |
| `st0_2_pattern` to `st6_2_pattern` | Color 2 pattern elements |
| `st0_2` to `st6_2` | Color 2 solid elements |
| `st0_3` to `st6_3` | Color 3 solid elements |

Important:
- Use `_2_pattern` only for Color 2 pattern elements.
- Use `_2` for Color 2 solid elements.
- Use `_1` for Color 1 solid elements.
- Use `_3` for Color 3 solid elements.
- Every element must be assigned to the correct class and parent layer.
- Do not assign classes randomly.

---

## 6. Image Processing and SVG Conversion Logic

The tool must implement a programmatic image-processing pipeline.

The conversion should involve the following steps:

### Step 1: Image Preprocessing
- Read the uploaded image.
- Normalize the image where necessary.
- Preserve transparency.
- Identify the image dimensions.
- Remove unnecessary background noise where appropriate.
- Prepare the image for segmentation and vectorization.

### Step 2: Color Detection and Separation

Identify and separate the image into three primary color categories:

- Color 1
- Color 2
- Color 3

The processing logic should:
- Detect dominant colors.
- Identify different color regions.
- Group visually connected regions where appropriate.
- Distinguish between solid color regions and pattern regions.
- Handle anti-aliased edges and color variations.
- Avoid treating every slightly different pixel color as a separate SVG color.

The system should support images containing multiple disconnected regions of the same color.

### Step 3: Pattern Detection

Identify regions that should be treated as Color 2 patterns.

The pattern-detection logic should consider:
- Repeated shapes.
- Repeated motifs.
- Textures.
- Pattern-filled regions.
- Differences between solid and patterned areas.

Pattern elements must be assigned to `Layer_3_2` and use the `_2_pattern` class naming convention.

Color 2 solid elements must instead be assigned to `Layer_2_1_2`.

### Step 4: Vectorization

Convert raster image regions into vector paths.

Requirements:
- Generate clean SVG paths.
- Avoid unnecessary anchor points.
- Reduce excessive path complexity.
- Preserve important curves and boundaries.
- Maintain the original design proportions.
- Avoid jagged edges.
- Preserve small decorative details as much as reasonably possible.
- Avoid overlapping or duplicated paths.

### Step 5: SVG Layer Generation

After vectorization, organize all generated paths into the required SVG groups.

Each element must be assigned to:
- The correct color.
- The correct solid/pattern category.
- The correct SVG group.
- The correct CSS class.

### Step 6: SVG Cleanup

Before generating the final file:
- Remove unnecessary metadata.
- Remove unused definitions.
- Remove redundant groups.
- Remove unnecessary attributes.
- Preserve required IDs and classes.
- Preserve required definitions and reusable elements.
- Ensure valid XML and SVG syntax.

---

## 7. Framework and Technology Flexibility

There are no restrictions on which programming languages, frameworks, libraries, or technologies can be used to develop this tool.

The development team / AI agent is free to choose any suitable technology stack based on:
- Conversion accuracy and SVG output quality.
- Performance and processing speed.
- Ease of development and maintenance.
- Scalability and future improvements.
- Compatibility with the required SVG structure.

You can use any frontend framework, backend framework, programming language, image-processing library, or SVG-generation library.

Examples include:
- **Frontend:** React, Next.js, Vue.js, Angular, Svelte, or any other suitable framework.
- **Backend:** Python (FastAPI, Flask, Django), Node.js (Express, NestJS), or any other suitable backend.
- **Image Processing:** OpenCV, Pillow, scikit-image, or other open-source libraries.
- **SVG Generation:** Potrace, VTracer, svgwrite, or custom implementations.

These are examples only, not mandatory technologies.

**Important restrictions:**
1. All technologies, frameworks, and libraries must be free to use.
2. Prefer open-source solutions with licenses compatible with commercial use.
3. No paid APIs, subscriptions, or external AI services.
4. No AI/ML-based image generation or conversion is required.
5. The primary focus must remain on generating accurate SVG files that follow our predefined structure.

Choose whichever technology stack delivers the best results. Do not restrict the implementation to Python, React, or any specific framework.

---

## 8. User Interface Requirements

Build a simple and user-friendly interface.

The interface should include:

1. Image upload area.
2. Image preview.
3. Convert to SVG button.
4. Processing status indicator.
5. Generated SVG preview.
6. Download SVG button.

### Optional but Recommended Features
- Before/after comparison.
- SVG source code preview.
- Display detected colors.
- Display detected layers.
- Ability to regenerate the SVG after adjusting color assignments.
- Ability to manually move elements between layers.
- Ability to change an element's color category.
- Conversion error messages.

The interface should make it easy to verify whether the generated SVG matches the original image.

---

## 9. SVG Validation Requirements

Every generated SVG must pass the following checks before download:

- Valid SVG/XML syntax.
- Correct SVG dimensions and viewBox.
- Required layer IDs are present.
- Correct class naming conventions.
- Correct parent-child layer hierarchy.
- All Color 1 elements are in `Layer_1`.
- Color 2 pattern elements are in `Layer_3_2`.
- Color 3 elements in the pattern configuration are in `Layer_3_3`.
- Color 2 solid elements are in `Layer_2_1_2`.
- Color 3 elements in the solid configuration are in `Layer_2_1_3`.
- `Layer_rose` is included.
- No unnecessary metadata.
- No broken references or missing definitions.
- No unintended duplicate paths.

The generated SVG must also be tested in a standard SVG viewer and our existing website.

---

## 10. Reference-Based Compatibility Testing

The generated SVG must be tested against the provided sample SVG files.

Compare:
- SVG structure.
- Layer IDs.
- CSS classes.
- Color separation.
- Pattern handling.
- Path quality.
- Rendering accuracy.
- Compatibility with the existing product customization system.

The final output must not just look correct in a browser. It must also behave correctly when our website changes the colors and toggles the pattern/solid layers.

---

## 11. Testing with Real Products

After the conversion functionality is implemented:

1. Convert the supplied sample images.
2. Generate SVG files using the required structure.
3. Add the generated SVG to a test product on our website.
4. Test Color 1, Color 2 and Color 3 independently.
5. Test Color 2 in solid color mode.
6. Test Color 2 in pattern mode.
7. Verify that Color 3 remains correct in both modes.
8. Check for missing, duplicated or incorrectly colored elements.
9. Make any necessary adjustments to the conversion logic.

Manual corrections may be needed initially, but the objective is to make the conversion process increasingly automated and consistent.

---

## 12. Expected Deliverables

The development should deliver:

1. A working image-to-SVG conversion tool.
2. Complete source code.
3. Image preprocessing and color-separation logic.
4. Pattern detection and handling logic.
5. SVG vectorization functionality.
6. Structured SVG generation functionality.
7. SVG validation functionality.
8. A simple upload and download interface.
9. Documentation explaining the conversion pipeline.
10. Generated SVG files for the supplied sample images.
11. Testing results demonstrating compatibility with our website.

---

## 13. Important Development Instructions

- First inspect all provided reference SVG files and understand their structure.
- Do not start implementation before understanding the SVG layer and class requirements.
- Do not use AI models, AI APIs, or paid image conversion services.
- Do not use a basic raster-to-SVG conversion and consider the task complete.
- The main challenge is correctly identifying colors, patterns and individual design elements, then placing them into the required SVG structure.
- Keep the conversion logic modular so that color detection, pattern detection, vectorization and SVG generation can be improved independently.
- Prioritize SVG compatibility with our existing website over simply producing visually similar SVG files.
- Make the tool work with different uploaded designs, not just the supplied sample images.

**Final Goal:** Upload any supported raster image and automatically generate a clean, editable, properly layered SVG file that follows our exact predefined SVG structure and works correctly with our existing product customization system.