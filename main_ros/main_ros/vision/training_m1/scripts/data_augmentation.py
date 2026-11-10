#!/usr/bin/env python3

"""
============================================================
PUZZLEBOT - CONFIGURABLE DATA AUGMENTATION
============================================================

Input:
    frames/
        logo_1_train/
        logo_1_val/
        logo_2_train/
        logo_2_val/
        logo_3_train/
        logo_3_val/

Output:
    frames_augmentation/
        logo_1_train/
        logo_1_val/
        logo_2_train/
        logo_2_val/
        logo_3_train/
        logo_3_val/

Features:
    - Configurable strength for every augmentation
    - Configurable probability for every augmentation
    - No rotations
    - No horizontal flips
    - Original training images preserved
    - Validation images copied without augmentation
    - Reproducible random seed
    - Exactly 800 training images per class by default
    - CSV manifest for dataset tracking

Dependencies:
    numpy
    opencv-python
"""

from pathlib import Path
import argparse
import random
import shutil
import csv

import cv2
import numpy as np


# ============================================================
# 1. GENERAL CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "frames"
OUTPUT_DIR = BASE_DIR / "frames_augmentation"

# Final number of training images per logo.
TRAIN_TARGET = 800

# Reproducibility.
RANDOM_SEED = 42

# JPEG quality for augmented images.
JPEG_OUTPUT_QUALITY = 95

# Number of augmentation effects per generated image.
MIN_EFFECTS = 2
MAX_EFFECTS = 4

# Supported source image formats.
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

FOLDERS = [
    "logo_1_train",
    "logo_1_val",
    "logo_2_train",
    "logo_2_val",
    "logo_3_train",
    "logo_3_val",
]


# ============================================================
# 2. AUGMENTATION STRENGTH
# ============================================================

"""
Strength range:

    0.00 = Disabled
    0.25 = Mild
    0.50 = Moderate
    0.75 = Strong
    1.00 = Very strong

Recommended starting values:

    0.30 - 0.50

These values control the intensity of each transformation.
"""

BRIGHTNESS_STRENGTH = 0.50

CONTRAST_STRENGTH = 0.45

GAMMA_STRENGTH = 0.40

MOTION_BLUR_STRENGTH = 0.40

GAUSSIAN_BLUR_STRENGTH = 0.35

NOISE_STRENGTH = 0.35

SHADOW_STRENGTH = 0.45

COLOR_TEMPERATURE_STRENGTH = 0.40

PERSPECTIVE_STRENGTH = 0.30

ZOOM_STRENGTH = 0.35

JPEG_COMPRESSION_STRENGTH = 0.35


# ============================================================
# 3. AUGMENTATION PROBABILITIES
# ============================================================

"""
Probability controls how often each effect is selected.

Examples:

    0.00 = Never
    0.25 = Low chance
    0.50 = Medium chance
    0.75 = High chance
    1.00 = Always eligible

The pipeline selects up to MAX_EFFECTS effects per image.
These probabilities are independent selection chances,
not guarantees of application frequency.
"""

BRIGHTNESS_PROBABILITY = 0.65

CONTRAST_PROBABILITY = 0.60

GAMMA_PROBABILITY = 0.45

MOTION_BLUR_PROBABILITY = 0.45

GAUSSIAN_BLUR_PROBABILITY = 0.40

NOISE_PROBABILITY = 0.50

SHADOW_PROBABILITY = 0.50

COLOR_TEMPERATURE_PROBABILITY = 0.45

PERSPECTIVE_PROBABILITY = 0.35

ZOOM_PROBABILITY = 0.40

JPEG_COMPRESSION_PROBABILITY = 0.35


# ============================================================
# 4. UTILITY FUNCTIONS
# ============================================================

def clamp_strength(value):
    """Ensure strength is between 0 and 1."""

    return max(
        0.0,
        min(1.0, float(value)),
    )


def clip_uint8(image):
    """Convert floating-point image to valid uint8."""

    return np.clip(
        image,
        0,
        255,
    ).astype(np.uint8)


def get_images(folder):
    """Return sorted image paths."""

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def odd_kernel(value):
    """Convert a value to an odd kernel size >= 3."""

    value = max(
        3,
        int(round(value)),
    )

    if value % 2 == 0:
        value += 1

    return value


# ============================================================
# 5. BRIGHTNESS
# ============================================================

def augment_brightness(image, rng, strength):

    """
    Simulate brighter or darker environments.
    """

    strength = clamp_strength(strength)

    if strength == 0:
        return image

    # Maximum brightness change at strength 1.0: +/- 65.
    max_change = 65.0 * strength

    change = rng.uniform(
        -max_change,
        max_change,
    )

    result = (
        image.astype(np.float32)
        + change
    )

    return clip_uint8(result)


# ============================================================
# 6. CONTRAST
# ============================================================

def augment_contrast(image, rng, strength):

    """
    Change image contrast.
    """

    strength = clamp_strength(strength)

    if strength == 0:
        return image

    # At strength 1.0:
    # Contrast multiplier ranges approximately 0.45 - 1.55.
    variation = 0.55 * strength

    alpha = rng.uniform(
        1.0 - variation,
        1.0 + variation,
    )

    result = (
        (image.astype(np.float32) - 127.5)
        * alpha
        + 127.5
    )

    return clip_uint8(result)


# ============================================================
# 7. GAMMA EXPOSURE
# ============================================================

def augment_gamma(image, rng, strength):

    """
    Simulate camera exposure differences.
    """

    strength = clamp_strength(strength)

    if strength == 0:
        return image

    max_log_gamma = 1.0 * strength

    log_gamma = rng.uniform(
        -max_log_gamma,
        max_log_gamma,
    )

    gamma = np.exp(log_gamma)

    normalized = (
        image.astype(np.float32)
        / 255.0
    )

    result = (
        np.power(normalized, gamma)
        * 255.0
    )

    return clip_uint8(result)


# ============================================================
# 8. MOTION BLUR
# ============================================================

def augment_motion_blur(image, rng, strength):

    """
    Simulate camera movement while the robot drives.
    """

    strength = clamp_strength(strength)

    if strength == 0:
        return image

    # Kernel size:
    # Strength 0.5 -> maximum approximately 9 pixels.
    # Strength 1.0 -> maximum approximately 15 pixels.
    max_kernel = odd_kernel(
        3 + 12 * strength
    )

    kernel_sizes = list(
        range(3, max_kernel + 1, 2)
    )

    kernel_size = rng.choice(
        kernel_sizes
    )

    kernel = np.zeros(
        (kernel_size, kernel_size),
        dtype=np.float32,
    )

    direction = rng.choice([
        "horizontal",
        "vertical",
        "diagonal",
        "anti_diagonal",
    ])

    if direction == "horizontal":

        kernel[kernel_size // 2, :] = 1.0

    elif direction == "vertical":

        kernel[:, kernel_size // 2] = 1.0

    elif direction == "diagonal":

        np.fill_diagonal(
            kernel,
            1.0,
        )

    else:

        np.fill_diagonal(
            np.fliplr(kernel),
            1.0,
        )

    kernel /= kernel.sum()

    return cv2.filter2D(
        image,
        -1,
        kernel,
    )


# ============================================================
# 9. GAUSSIAN BLUR
# ============================================================

def augment_gaussian_blur(image, rng, strength):

    """
    Simulate slightly out-of-focus images.
    """

    strength = clamp_strength(strength)

    if strength == 0:
        return image

    max_sigma = 0.5 + 3.5 * strength

    sigma = rng.uniform(
        0.5,
        max_sigma,
    )

    kernel_size = odd_kernel(
        3 + 8 * strength
    )

    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=sigma,
    )


# ============================================================
# 10. GAUSSIAN NOISE
# ============================================================

def augment_noise(image, rng, strength):

    """
    Simulate sensor noise.
    """

    strength = clamp_strength(strength)

    if strength == 0:
        return image

    # At strength 0.5:
    # Noise sigma approximately 4 - 17.5.
    max_sigma = 35.0 * strength

    sigma = rng.uniform(
        max(0.5, max_sigma * 0.25),
        max_sigma,
    )

    noise = rng.np_rng.normal(
        0,
        sigma,
        image.shape,
    )

    result = (
        image.astype(np.float32)
        + noise
    )

    return clip_uint8(result)


# ============================================================
# 11. SHADOWS
# ============================================================

def augment_shadows(image, rng, strength):

    """
    Simulate uneven warehouse lighting.

    Creates a soft shadow across part of the image.
    """

    strength = clamp_strength(strength)

    if strength == 0:
        return image

    height, width = image.shape[:2]

    shadow_mask = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    # Generate a random diagonal shadow boundary.
    x_top = rng.randint(
        0,
        max(0, width - 1),
    )

    x_bottom = rng.randint(
        0,
        max(0, width - 1),
    )

    polygon = np.array([
        [0, 0],
        [x_top, 0],
        [x_bottom, height - 1],
        [0, height - 1],
    ], dtype=np.int32)

    cv2.fillPoly(
        shadow_mask,
        [polygon],
        1.0,
    )

    # Blur boundary.
    blur_size = odd_kernel(
        max(3, min(height, width) * 0.08)
    )

    shadow_mask = cv2.GaussianBlur(
        shadow_mask,
        (blur_size, blur_size),
        0,
    )

    # Strength 0.5 -> up to 35% darker.
    max_darkness = 0.70 * strength

    darkness = rng.uniform(
        max_darkness * 0.5,
        max_darkness,
    )

    lighting = (
        1.0 - shadow_mask * darkness
    )

    result = (
        image.astype(np.float32)
        * lighting[:, :, None]
    )

    return clip_uint8(result)


# ============================================================
# 12. COLOR TEMPERATURE
# ============================================================

def augment_color_temperature(image, rng, strength):

    """
    Simulate warm or cold artificial lighting.

    OpenCV format: BGR.
    """

    strength = clamp_strength(strength)

    if strength == 0:
        return image

    max_shift = 0.45 * strength

    shift = rng.uniform(
        max_shift * 0.4,
        max_shift,
    )

    result = image.astype(
        np.float32
    )

    warm = rng.choice([
        True,
        False,
    ])

    if warm:

        # Increase red, decrease blue.
        result[:, :, 2] *= 1.0 + shift
        result[:, :, 0] *= 1.0 - shift

    else:

        # Increase blue, decrease red.
        result[:, :, 0] *= 1.0 + shift
        result[:, :, 2] *= 1.0 - shift

    return clip_uint8(result)


# ============================================================
# 13. PERSPECTIVE DISTORTION
# ============================================================

def augment_perspective(image, rng, strength):

    """
    Simulate slightly different camera viewpoints.

    No explicit image rotation is used.
    """

    strength = clamp_strength(strength)

    if strength == 0:
        return image

    height, width = image.shape[:2]

    # Maximum displacement at strength 1.0: 10%.
    margin_x = width * 0.10 * strength
    margin_y = height * 0.10 * strength

    source = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ])

    destination = source.copy()

    for point in destination:

        point[0] += rng.uniform(
            -margin_x,
            margin_x,
        )

        point[1] += rng.uniform(
            -margin_y,
            margin_y,
        )

    matrix = cv2.getPerspectiveTransform(
        source,
        destination,
    )

    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


# ============================================================
# 14. ZOOM
# ============================================================

def augment_zoom(image, rng, strength):

    """
    Simulate the robot moving closer to a logo.

    Zoom-in only, no rotations.
    """

    strength = clamp_strength(strength)

    if strength == 0:
        return image

    height, width = image.shape[:2]

    max_zoom = 0.35 * strength

    scale = rng.uniform(
        1.0,
        1.0 + max_zoom,
    )

    new_width = max(
        width,
        int(round(width * scale)),
    )

    new_height = max(
        height,
        int(round(height * scale)),
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    start_x = (
        new_width - width
    ) // 2

    start_y = (
        new_height - height
    ) // 2

    return resized[
        start_y:start_y + height,
        start_x:start_x + width,
    ]


# ============================================================
# 15. JPEG COMPRESSION
# ============================================================

def augment_jpeg_compression(image, rng, strength):

    """
    Simulate compressed camera images.
    """

    strength = clamp_strength(strength)

    if strength == 0:
        return image

    # At strength 1.0, minimum quality is 20.
    # At strength 0.5, minimum quality is ~58.
    minimum_quality = int(
        95 - 75 * strength
    )

    quality = rng.randint(
        minimum_quality,
        95,
    )

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            quality,
        ],
    )

    if not success:
        return image

    decoded = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR,
    )

    if decoded is None:
        return image

    return decoded


# ============================================================
# 16. RANDOM GENERATOR
# ============================================================

class RandomGenerator:

    def __init__(self, seed):

        self.py_rng = random.Random(seed)

        self.np_rng = np.random.default_rng(seed)

    def uniform(self, a, b):

        return self.py_rng.uniform(a, b)

    def randint(self, a, b):

        return self.py_rng.randint(a, b)

    def choice(self, values):

        return self.py_rng.choice(values)

    def sample(self, values, count):

        return self.py_rng.sample(
            values,
            count,
        )

    def random(self):

        return self.py_rng.random()


# ============================================================
# 17. AUGMENTATION REGISTRY
# ============================================================

def get_augmentation_registry():

    """
    Each augmentation has:

        name
        function
        strength
        probability

    Change the global variables at the top of the file
    to adjust the pipeline.
    """

    return [
        {
            "name": "brightness",
            "function": augment_brightness,
            "strength": BRIGHTNESS_STRENGTH,
            "probability": BRIGHTNESS_PROBABILITY,
        },
        {
            "name": "contrast",
            "function": augment_contrast,
            "strength": CONTRAST_STRENGTH,
            "probability": CONTRAST_PROBABILITY,
        },
        {
            "name": "gamma",
            "function": augment_gamma,
            "strength": GAMMA_STRENGTH,
            "probability": GAMMA_PROBABILITY,
        },
        {
            "name": "motion_blur",
            "function": augment_motion_blur,
            "strength": MOTION_BLUR_STRENGTH,
            "probability": MOTION_BLUR_PROBABILITY,
        },
        {
            "name": "gaussian_blur",
            "function": augment_gaussian_blur,
            "strength": GAUSSIAN_BLUR_STRENGTH,
            "probability": GAUSSIAN_BLUR_PROBABILITY,
        },
        {
            "name": "noise",
            "function": augment_noise,
            "strength": NOISE_STRENGTH,
            "probability": NOISE_PROBABILITY,
        },
        {
            "name": "shadows",
            "function": augment_shadows,
            "strength": SHADOW_STRENGTH,
            "probability": SHADOW_PROBABILITY,
        },
        {
            "name": "color_temperature",
            "function": augment_color_temperature,
            "strength": COLOR_TEMPERATURE_STRENGTH,
            "probability": COLOR_TEMPERATURE_PROBABILITY,
        },
        {
            "name": "perspective",
            "function": augment_perspective,
            "strength": PERSPECTIVE_STRENGTH,
            "probability": PERSPECTIVE_PROBABILITY,
        },
        {
            "name": "zoom",
            "function": augment_zoom,
            "strength": ZOOM_STRENGTH,
            "probability": ZOOM_PROBABILITY,
        },
        {
            "name": "jpeg_compression",
            "function": augment_jpeg_compression,
            "strength": JPEG_COMPRESSION_STRENGTH,
            "probability": JPEG_COMPRESSION_PROBABILITY,
        },
    ]


# ============================================================
# 18. AUGMENTATION PIPELINE
# ============================================================

def augment_image(image, rng, registry):

    """
    Apply between MIN_EFFECTS and MAX_EFFECTS.

    Effects are sampled according to their probabilities.

    At least one effect is applied when any enabled
    augmentation exists.

    No rotations.
    No horizontal flips.
    """

    enabled = [
        operation
        for operation in registry
        if operation["strength"] > 0
        and operation["probability"] > 0
    ]

    if not enabled:
        return image.copy(), []

    # First, select effects using their probabilities.
    selected = [
        operation
        for operation in enabled
        if rng.random() < operation["probability"]
    ]

    # Ensure enough effects when possible.
    minimum = min(
        MIN_EFFECTS,
        len(enabled),
    )

    if len(selected) < minimum:

        remaining = [
            operation
            for operation in enabled
            if operation not in selected
        ]

        needed = min(
            minimum - len(selected),
            len(remaining),
        )

        selected.extend(
            rng.sample(remaining, needed)
        )

    # Limit number of effects.
    maximum = min(
        MAX_EFFECTS,
        len(selected),
    )

    if len(selected) > maximum:

        selected = rng.sample(
            selected,
            maximum,
        )

    rng.py_rng.shuffle(selected)

    augmented = image.copy()

    applied_names = []

    for operation in selected:

        augmented = operation["function"](
            augmented,
            rng,
            operation["strength"],
        )

        applied_names.append(
            operation["name"]
        )

    return augmented, applied_names


# ============================================================
# 19. DATASET PROCESSING
# ============================================================

def process_folder(
    folder_name,
    rng,
    registry,
    train_target,
):

    input_folder = INPUT_DIR / folder_name
    output_folder = OUTPUT_DIR / folder_name

    print("\n" + "=" * 60)
    print(f"Processing: {folder_name}")
    print("=" * 60)

    if not input_folder.is_dir():

        raise FileNotFoundError(
            f"Input folder not found: {input_folder}"
        )

    images = get_images(
        input_folder
    )

    if not images:

        raise RuntimeError(
            f"No images found in {input_folder}"
        )

    is_training = folder_name.endswith("_train")

    real_count = len(images)

    if is_training and real_count > train_target:

        raise ValueError(
            f"{folder_name} contains {real_count} images, "
            f"which exceeds the target of {train_target}."
        )

    # Load and validate all source images before modifying
    # the output folder.
    source_images = []

    for image_path in images:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise RuntimeError(
                f"Cannot read image: {image_path}"
            )

        source_images.append(
            (image_path, image)
        )

    # Regenerate this output folder.
    # The original frames/ folder is never modified.
    if output_folder.exists():

        shutil.rmtree(
            output_folder
        )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = []

    # --------------------------------------------------------
    # Copy original images.
    # --------------------------------------------------------

    for index, (image_path, _) in enumerate(source_images):

        filename = (
            f"real_{index:06d}"
            f"{image_path.suffix.lower()}"
        )

        destination = output_folder / filename

        shutil.copy2(
            image_path,
            destination,
        )

        manifest_rows.append({
            "filename": filename,
            "source": image_path.name,
            "type": "real",
            "effects": "",
        })

    print(f"Original images: {real_count}")

    # --------------------------------------------------------
    # Training augmentation.
    # --------------------------------------------------------

    augmented_needed = 0

    if is_training:

        augmented_needed = (
            train_target - real_count
        )

        print(
            f"Augmented images needed: "
            f"{augmented_needed}"
        )

        # Cycle through shuffled source images.
        # This avoids augmenting only a few originals.
        source_indices = list(
            range(real_count)
        )

        rng.py_rng.shuffle(
            source_indices
        )

        for index in range(augmented_needed):

            if index > 0 and index % real_count == 0:

                rng.py_rng.shuffle(
                    source_indices
                )

            source_index = source_indices[
                index % real_count
            ]

            source_path, source_image = source_images[
                source_index
            ]

            augmented, effects = augment_image(
                source_image,
                rng,
                registry,
            )

            filename = (
                f"aug_{index:06d}.jpg"
            )

            destination = output_folder / filename

            success = cv2.imwrite(
                str(destination),
                augmented,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_OUTPUT_QUALITY,
                ],
            )

            if not success:

                raise RuntimeError(
                    f"Failed to write image: {destination}"
                )

            manifest_rows.append({
                "filename": filename,
                "source": source_path.name,
                "type": "augmented",
                "effects": "|".join(effects),
            })

            if (index + 1) % 100 == 0:

                print(
                    f"Generated "
                    f"{index + 1}/"
                    f"{augmented_needed}"
                )

    else:

        print(
            "Validation folder: "
            "original images only."
        )

    # --------------------------------------------------------
    # Save manifest.
    # --------------------------------------------------------

    manifest_path = (
        output_folder / "manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "source",
                "type",
                "effects",
            ],
        )

        writer.writeheader()
        writer.writerows(
            manifest_rows
        )

    final_count = len(
        get_images(output_folder)
    )

    print(f"Final images: {final_count}")

    return real_count, augmented_needed


# ============================================================
# 20. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "PuzzleBot configurable logo "
            "data augmentation."
        )
    )

    parser.add_argument(
        "--train-target",
        type=int,
        default=TRAIN_TARGET,
        help="Total training images per class.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed.",
    )

    args = parser.parse_args()

    if args.train_target <= 0:

        parser.error(
            "--train-target must be positive"
        )

    if not INPUT_DIR.is_dir():

        parser.error(
            f"Input directory does not exist: {INPUT_DIR}"
        )

    if MIN_EFFECTS < 0 or MAX_EFFECTS < MIN_EFFECTS:

        parser.error(
            "Invalid MIN_EFFECTS / MAX_EFFECTS settings."
        )

    registry = get_augmentation_registry()

    for operation in registry:

        if not 0.0 <= operation["strength"] <= 1.0:

            parser.error(
                f"Invalid strength for {operation['name']}: "
                f"{operation['strength']}"
            )

        if not 0.0 <= operation["probability"] <= 1.0:

            parser.error(
                f"Invalid probability for {operation['name']}: "
                f"{operation['probability']}"
            )

    rng = RandomGenerator(
        args.seed
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print("PUZZLEBOT - CONFIGURABLE DATA AUGMENTATION")
    print("=" * 60)

    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Training target: {args.train_target}")
    print(f"Seed: {args.seed}")

    print("\nAugmentation settings:")

    for operation in registry:

        print(
            f"  {operation['name']:<20} "
            f"strength={operation['strength']:.2f} "
            f"probability={operation['probability']:.2f}"
        )

    total_real = 0
    total_augmented = 0

    for folder_name in FOLDERS:

        real_count, augmented_count = process_folder(
            folder_name=folder_name,
            rng=rng,
            registry=registry,
            train_target=args.train_target,
        )

        total_real += real_count
        total_augmented += augmented_count

    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)

    print(f"Original images: {total_real}")
    print(f"Augmented images: {total_augmented}")
    print(
        f"Total images: "
        f"{total_real + total_augmented}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()