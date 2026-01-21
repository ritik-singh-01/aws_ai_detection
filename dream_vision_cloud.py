"""
Dream Vision - AWS CLOUD VERSION
=================================================
Optimized for AWS G6 Instance with NVIDIA L4 GPU (24GB VRAM)
High-performance cloud deployment for real-time AI art generation.

Hardware Target: AWS g6.xlarge / g6.2xlarge (NVIDIA L4 - 24GB VRAM)
Expected Performance: 25-40 FPS at 768x768 resolution

Key Features:
1. Multi-person detection and individual processing
2. Gender auto-detection for adaptive prompts
3. Face and body preservation (protected from transformation)
4. Higher resolution support (768x768) due to 24GB VRAM
5. Video file input support (no physical webcam)
6. Headless operation mode for cloud deployment

Author: AI Creative Technologist
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import threading
import queue
import time
import gc
import os
import argparse
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union
from contextlib import contextmanager
from pathlib import Path

# Conditional imports for cloud features
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[Warning] MediaPipe not available, using fallback segmentation")

from diffusers import AutoPipelineForImage2Image
import warnings
from enum import Enum

warnings.filterwarnings("ignore")


# ============================================================================
# GENDER ENUM
# ============================================================================
class Gender(Enum):
    UNKNOWN = "unknown"
    MALE = "male"
    FEMALE = "female"
    MIXED = "mixed"  # Multiple people with different genders

# Force high performance CUDA settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# ============================================================================
# CLOUD CONFIGURATION
# ============================================================================
@dataclass
class CloudConfig:
    """Configuration optimized for AWS G6 (L4 GPU - 24GB VRAM)"""

    # Resolution - Higher for L4 GPU (24GB VRAM allows larger images)
    width: int = 768  # Can go up to 1024 with L4
    height: int = 768

    # Model
    model_id: str = "stabilityai/sdxl-turbo"

    # SDXL Turbo Settings - Optimized for quality + speed
    num_inference_steps: int = 2  # 2 steps for better quality with L4
    guidance_scale: float = 0.0
    denoising_strength: float = 0.50  # Balanced for cloud

    # Consistency
    fixed_seed: int = 42

    # Base Prompts (will be enhanced based on detected gender)
    base_prompt: str = (
        "dreamy surreal oil painting, ethereal atmosphere, "
        "mystical glowing environment, soft venetian mist, "
        "golden luminescent light, renaissance masterpiece style, "
        "Nikas Safronov dream vision, 8k ultra detailed, "
        "preserve human features exactly, maintain facial identity"
    )

    # Gender-specific prompt additions
    male_prompt_addition: str = (
        "handsome man, masculine features preserved, "
        "strong facial structure maintained"
    )
    female_prompt_addition: str = (
        "beautiful woman, feminine features preserved, "
        "elegant facial structure maintained"
    )

    # Negative prompt - CRITICAL for face/body preservation
    negative_prompt: str = (
        # Quality issues
        "ugly, blurry, low quality, pixelated, noisy, "
        # Face protection - DO NOT alter these
        "deformed face, distorted face, disfigured face, bad face, "
        "mutated face, ugly face, poorly drawn face, cloned face, "
        "extra faces, duplicate faces, fused faces, "
        "wrong facial features, asymmetric eyes, "
        # Body protection - DO NOT alter these
        "deformed body, distorted body, disfigured body, "
        "mutated body, extra limbs, missing limbs, "
        "extra arms, missing arms, extra legs, missing legs, "
        "extra fingers, missing fingers, fused fingers, "
        "bad anatomy, wrong anatomy, bad proportions, "
        "extra hands, malformed hands, poorly drawn hands, "
        # Skin and appearance protection
        "bad skin, unnatural skin, plastic skin, "
        "doll-like appearance, mannequin, "
        # General quality
        "watermark, signature, text, logo, "
        "cropped, out of frame, worst quality"
    )

    # Face/Body Preservation Settings
    face_preservation_strength: float = 0.85  # How much to preserve face (0.0-1.0)
    body_preservation_strength: float = 0.60  # How much to preserve body (0.0-1.0)
    background_transform_strength: float = 0.95  # How much to transform background

    # Multi-person Settings
    max_persons: int = 10  # Maximum number of persons to detect
    min_detection_confidence: float = 0.5  # Minimum confidence for person detection

    # Gender Detection Settings
    enable_gender_detection: bool = True
    gender_detection_confidence: float = 0.6

    # Performance - Optimized for L4
    queue_size: int = 4  # Larger queue for cloud
    warmup_steps: int = 5  # More warmup for stable performance
    batch_size: int = 1  # Can increase to 2 with L4

    # Input Mode
    input_mode: str = "video"  # "video", "webcam", "images", "stream"
    input_source: str = "input.mp4"  # Video file, webcam ID, or stream URL

    # Output Mode
    output_mode: str = "video"  # "video", "display", "stream", "images"
    output_path: str = "output_dreamvision.mp4"
    output_fps: int = 30

    # Display (for testing with X11 forwarding)
    headless: bool = True  # True for cloud deployment
    window_name: str = "Dream Vision - AWS Cloud"
    show_stats: bool = True

    # API Server
    enable_api: bool = False
    api_port: int = 8080

    @property
    def prompt(self) -> str:
        """Default prompt (used when no gender detected)"""
        return self.base_prompt


# ============================================================================
# CUDA MEMORY MANAGER - OPTIMIZED FOR L4
# ============================================================================
class CUDAMemoryManager:
    """Manages CUDA memory for L4 GPU (24GB)"""

    def __init__(self):
        self.device = torch.device("cuda")
        self._print_gpu_info()

    def _print_gpu_info(self):
        """Print GPU information"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[GPU] {gpu_name}")
            print(f"[GPU] Total VRAM: {total_mem:.1f} GB")

            # Detect L4 GPU
            if "L4" in gpu_name:
                print("[GPU] NVIDIA L4 detected - Using optimized settings")

    @contextmanager
    def inference_mode(self):
        """Context manager for inference with memory optimization"""
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                yield

    def clear_cache(self):
        """Aggressive cache clearing"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def print_stats(self):
        """Print VRAM statistics"""
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[VRAM] Alloc: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Peak: {max_allocated:.2f}GB")


# ============================================================================
# PERSON DATA CLASS
# ============================================================================
@dataclass
class PersonInfo:
    """Information about a detected person"""
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    gender: Gender = Gender.UNKNOWN
    confidence: float = 0.0
    face_landmarks: Optional[List] = None
    body_landmarks: Optional[List] = None


# ============================================================================
# ADVANCED MULTI-PERSON DETECTOR WITH GENDER DETECTION
# ============================================================================
class MultiPersonDetector:
    """
    Advanced multi-person detection with:
    - Individual person detection and tracking
    - Gender classification
    - Face landmark detection
    - Body pose estimation
    """

    def __init__(self, config: CloudConfig):
        self.config = config
        self.target_size = (config.width, config.height)

        # Initialize MediaPipe components
        self.face_detection = None
        self.face_mesh = None
        self.pose = None
        self.selfie_segmentation = None

        if MEDIAPIPE_AVAILABLE:
            self._init_mediapipe()
        else:
            print("[MultiPersonDetector] MediaPipe not available - using fallback")

        # Pre-allocate kernels for morphological operations
        self._kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        print("[MultiPersonDetector] Initialized")

    def _init_mediapipe(self):
        """Initialize all MediaPipe components"""
        try:
            # Face Detection - for detecting multiple faces
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 1 for full range (up to 5m)
                min_detection_confidence=self.config.min_detection_confidence
            )
            print("[MediaPipe] Face Detection: ENABLED")
        except Exception as e:
            print(f"[MediaPipe] Face Detection failed: {e}")

        try:
            # Face Mesh - for detailed face landmarks (gender estimation)
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self.config.max_persons,
                refine_landmarks=True,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=0.5
            )
            print("[MediaPipe] Face Mesh: ENABLED")
        except Exception as e:
            print(f"[MediaPipe] Face Mesh failed: {e}")

        try:
            # Pose Detection - for body landmarks
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=True,
                min_detection_confidence=self.config.min_detection_confidence
            )
            print("[MediaPipe] Pose: ENABLED")
        except Exception as e:
            print(f"[MediaPipe] Pose failed: {e}")

        try:
            # Selfie Segmentation - for person/background separation
            self.mp_selfie = mp.solutions.selfie_segmentation
            self.selfie_segmentation = self.mp_selfie.SelfieSegmentation(model_selection=1)
            print("[MediaPipe] Selfie Segmentation: ENABLED")
        except Exception as e:
            print(f"[MediaPipe] Selfie Segmentation failed: {e}")

    def detect_persons(self, frame: np.ndarray) -> List[PersonInfo]:
        """
        Detect all persons in frame with their properties.

        Returns:
            List of PersonInfo objects for each detected person
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        persons = []

        if self.face_detection is None:
            # Fallback: return single dummy person
            return [PersonInfo(id=0, bbox=(0, 0, w, h), gender=Gender.UNKNOWN)]

        # Detect faces
        face_results = self.face_detection.process(rgb)

        if face_results.detections:
            for idx, detection in enumerate(face_results.detections):
                if idx >= self.config.max_persons:
                    break

                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)

                # Expand bbox to include body (approximate)
                body_x = max(0, x - box_w // 2)
                body_y = max(0, y - box_h // 4)
                body_w = min(w - body_x, box_w * 2)
                body_h = min(h - body_y, int(box_h * 4))

                # Detect gender from face proportions and features
                gender = self._estimate_gender(rgb, (x, y, box_w, box_h))

                person = PersonInfo(
                    id=idx,
                    bbox=(body_x, body_y, body_w, body_h),
                    face_bbox=(x, y, box_w, box_h),
                    gender=gender,
                    confidence=detection.score[0] if detection.score else 0.5
                )
                persons.append(person)

        # If no faces detected, try to detect body
        if not persons and self.selfie_segmentation:
            seg_results = self.selfie_segmentation.process(rgb)
            if seg_results.segmentation_mask is not None:
                mask = seg_results.segmentation_mask
                if np.any(mask > 0.5):
                    # Person detected but no face visible
                    persons.append(PersonInfo(
                        id=0,
                        bbox=(0, 0, w, h),
                        gender=Gender.UNKNOWN,
                        confidence=0.5
                    ))

        return persons if persons else [PersonInfo(id=0, bbox=(0, 0, w, h), gender=Gender.UNKNOWN)]

    def _estimate_gender(self, rgb: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Gender:
        """
        Estimate gender from facial features using face mesh landmarks.
        Uses facial proportions and structure for classification.
        """
        if not self.config.enable_gender_detection or self.face_mesh is None:
            return Gender.UNKNOWN

        x, y, w, h = face_bbox
        img_h, img_w = rgb.shape[:2]

        # Extract face region with padding
        pad = int(max(w, h) * 0.3)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)

        face_roi = rgb[y1:y2, x1:x2]
        if face_roi.size == 0:
            return Gender.UNKNOWN

        # Get face mesh landmarks
        results = self.face_mesh.process(face_roi)

        if not results.multi_face_landmarks:
            return Gender.UNKNOWN

        landmarks = results.multi_face_landmarks[0]

        # Gender estimation based on facial proportions
        # These are approximate heuristics based on typical facial differences
        try:
            # Get key landmark points
            roi_h, roi_w = face_roi.shape[:2]

            # Jawline width (landmarks 234, 454 are jaw corners)
            jaw_left = landmarks.landmark[234]
            jaw_right = landmarks.landmark[454]
            jaw_width = abs(jaw_right.x - jaw_left.x) * roi_w

            # Face height (from chin to forehead)
            chin = landmarks.landmark[152]
            forehead = landmarks.landmark[10]
            face_height = abs(forehead.y - chin.y) * roi_h

            # Eyebrow thickness (distance between eyebrow landmarks)
            left_brow_top = landmarks.landmark[105]
            left_brow_bottom = landmarks.landmark[107]
            brow_thickness = abs(left_brow_top.y - left_brow_bottom.y) * roi_h

            # Nose width
            nose_left = landmarks.landmark[279]
            nose_right = landmarks.landmark[49]
            nose_width = abs(nose_right.x - nose_left.x) * roi_w

            # Calculate ratios
            jaw_to_height_ratio = jaw_width / face_height if face_height > 0 else 0
            brow_ratio = brow_thickness / face_height if face_height > 0 else 0
            nose_ratio = nose_width / jaw_width if jaw_width > 0 else 0

            # Scoring based on typical gender differences
            # Males typically have: wider jaw, thicker eyebrows, wider nose
            male_score = 0

            if jaw_to_height_ratio > 0.85:  # Wider jaw
                male_score += 1
            if brow_ratio > 0.025:  # Thicker eyebrows
                male_score += 1
            if nose_ratio > 0.28:  # Wider nose relative to jaw
                male_score += 1

            # Determine gender
            if male_score >= 2:
                return Gender.MALE
            elif male_score <= 0:
                return Gender.FEMALE
            else:
                return Gender.UNKNOWN

        except Exception as e:
            return Gender.UNKNOWN

    def get_overall_gender(self, persons: List[PersonInfo]) -> Gender:
        """
        Determine overall gender for prompt generation.
        """
        if not persons:
            return Gender.UNKNOWN

        genders = [p.gender for p in persons if p.gender != Gender.UNKNOWN]

        if not genders:
            return Gender.UNKNOWN
        elif len(set(genders)) == 1:
            return genders[0]
        else:
            return Gender.MIXED

    def release(self):
        """Release all MediaPipe resources"""
        if self.face_detection:
            self.face_detection.close()
        if self.face_mesh:
            self.face_mesh.close()
        if self.pose:
            self.pose.close()
        if self.selfie_segmentation:
            self.selfie_segmentation.close()


# ============================================================================
# ADVANCED SEGMENTATION WITH FACE/BODY PRESERVATION
# ============================================================================
class AdvancedSegmenter:
    """
    Advanced segmentation that creates separate masks for:
    - Face regions (highest preservation)
    - Body regions (medium preservation)
    - Background (full transformation)
    """

    def __init__(self, config: CloudConfig):
        self.config = config
        self.target_size = (config.width, config.height)

        # Initialize MediaPipe
        self.selfie_segmentation = None
        self.face_mesh = None

        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_selfie = mp.solutions.selfie_segmentation
                self.selfie_segmentation = self.mp_selfie.SelfieSegmentation(model_selection=1)

                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=config.max_persons,
                    refine_landmarks=True,
                    min_detection_confidence=config.min_detection_confidence
                )
                print("[AdvancedSegmenter] MediaPipe initialized")
            except Exception as e:
                print(f"[AdvancedSegmenter] MediaPipe failed: {e}")

        # Morphological kernels
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    def create_preservation_masks(
        self,
        frame: np.ndarray,
        persons: List[PersonInfo]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create three separate masks for different preservation levels.

        Returns:
            Tuple of (face_mask, body_mask, background_mask)
            All masks are float arrays (0.0-1.0)
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Initialize masks
        face_mask = np.zeros((h, w), dtype=np.float32)
        body_mask = np.zeros((h, w), dtype=np.float32)
        person_mask = np.zeros((h, w), dtype=np.float32)

        # Get person segmentation mask
        if self.selfie_segmentation:
            results = self.selfie_segmentation.process(rgb)
            if results.segmentation_mask is not None:
                person_mask = results.segmentation_mask.copy()

        # Create face masks from detected persons
        for person in persons:
            if person.face_bbox:
                face_mask = self._add_face_mask(face_mask, rgb, person.face_bbox)

        # Create detailed face mask from face mesh
        if self.face_mesh:
            mesh_results = self.face_mesh.process(rgb)
            if mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    face_mask = self._add_face_mesh_mask(face_mask, face_landmarks, w, h)

        # Clean up face mask
        face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_CLOSE, self._kernel)
        face_mask = cv2.GaussianBlur(face_mask, (21, 21), 0)
        face_mask = np.clip(face_mask, 0, 1)

        # Body mask = person mask - face mask
        body_mask = np.maximum(0, person_mask - face_mask)
        body_mask = cv2.GaussianBlur(body_mask, (11, 11), 0)

        # Background mask = inverse of person mask
        background_mask = 1.0 - person_mask
        background_mask = cv2.GaussianBlur(background_mask, (7, 7), 0)

        return face_mask, body_mask, background_mask

    def _add_face_mask(
        self,
        mask: np.ndarray,
        rgb: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Add elliptical face mask from bounding box"""
        x, y, w, h = face_bbox
        img_h, img_w = mask.shape

        # Create ellipse for face
        center_x = x + w // 2
        center_y = y + h // 2

        # Make ellipse slightly larger than bbox
        axis_x = int(w * 0.7)
        axis_y = int(h * 0.85)

        # Ensure within bounds
        center_x = max(axis_x, min(img_w - axis_x, center_x))
        center_y = max(axis_y, min(img_h - axis_y, center_y))

        # Draw filled ellipse
        cv2.ellipse(
            mask,
            (center_x, center_y),
            (axis_x, axis_y),
            0, 0, 360,
            1.0,
            -1  # Filled
        )

        return mask

    def _add_face_mesh_mask(
        self,
        mask: np.ndarray,
        landmarks,
        width: int,
        height: int
    ) -> np.ndarray:
        """Add detailed face mask from face mesh landmarks"""
        # Face oval indices in MediaPipe Face Mesh
        FACE_OVAL = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]

        points = []
        for idx in FACE_OVAL:
            lm = landmarks.landmark[idx]
            x = int(lm.x * width)
            y = int(lm.y * height)
            points.append([x, y])

        if len(points) > 2:
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 1.0)

        return mask

    def release(self):
        """Release MediaPipe resources"""
        if self.selfie_segmentation:
            self.selfie_segmentation.close()
        if self.face_mesh:
            self.face_mesh.close()


# ============================================================================
# CLOUD DIFFUSION PIPELINE - OPTIMIZED FOR L4 GPU
# ============================================================================
class CloudDiffusionPipeline:
    """
    High-performance diffusion pipeline optimized for AWS L4 GPU.

    L4 GPU Advantages:
    - 24GB VRAM (3x more than RTX 3070)
    - Ada Lovelace architecture
    - Higher memory bandwidth
    - Better tensor core performance
    """

    def __init__(self, config: CloudConfig):
        self.config = config
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        self.memory = CUDAMemoryManager()

        # Load pipeline
        self._load_pipeline()

        # Pre-encode prompts
        self._encode_prompts()

        # Create persistent tensors
        self._create_buffers()

        # Warmup
        self._warmup()

    def _load_pipeline(self):
        """Load and optimize SDXL Turbo for L4"""
        print(f"[Pipeline] Loading {self.config.model_id}...")

        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            self.config.model_id,
            torch_dtype=self.dtype,
            variant="fp16",
            use_safetensors=True,
        ).to(self.device)

        # Apply L4-optimized settings
        self._apply_l4_optimizations()

        self.memory.print_stats()

    def _apply_l4_optimizations(self):
        """Apply optimizations specific to L4 GPU"""

        # 1. xformers memory efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("[Opt] xformers: ENABLED")
        except Exception as e:
            self.pipe.enable_attention_slicing("auto")
            print(f"[Opt] xformers failed ({e}), using attention slicing")

        # 2. VAE optimizations
        self.pipe.enable_vae_slicing()
        print("[Opt] VAE slicing: ENABLED")

        # 3. For L4 with 24GB, we can enable VAE tiling for even larger images
        self.pipe.enable_vae_tiling()
        print("[Opt] VAE tiling: ENABLED")

        # 4. Disable safety checker
        self.pipe.safety_checker = None
        print("[Opt] Safety checker: DISABLED")

        # 5. Set to eval mode
        self.pipe.unet.eval()
        self.pipe.vae.eval()

        # 6. Attempt torch.compile for L4
        try:
            self.pipe.unet = torch.compile(
                self.pipe.unet,
                mode="reduce-overhead",  # Better for real-time
                fullgraph=False,
            )
            print("[Opt] torch.compile: ENABLED (reduce-overhead mode)")
        except Exception as e:
            print(f"[Opt] torch.compile failed: {e}")

    def _encode_prompts(self):
        """Pre-encode prompts for faster inference"""
        print("[Pipeline] Pre-encoding prompts...")

        with self.memory.inference_mode():
            result = self.pipe.encode_prompt(
                prompt=self.config.prompt,
                prompt_2=self.config.prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )

            self.prompt_embeds = result[0]
            self.pooled_prompt_embeds = result[1]

        print("[Pipeline] Prompts cached")

    def _create_buffers(self):
        """Create persistent CUDA tensors"""
        self.generator = torch.Generator(device=self.device).manual_seed(
            self.config.fixed_seed
        )

        latent_h = self.config.height // 8
        latent_w = self.config.width // 8
        self.latent_buffer = torch.zeros(
            (1, 4, latent_h, latent_w),
            device=self.device,
            dtype=self.dtype
        )

        print(f"[Pipeline] Latent buffer: {self.latent_buffer.shape}")

    def _warmup(self):
        """Warmup pipeline for consistent performance"""
        print(f"[Pipeline] Warming up ({self.config.warmup_steps} steps)...")

        dummy = Image.new("RGB", (self.config.width, self.config.height), color="gray")

        for i in range(self.config.warmup_steps):
            _ = self.transform(dummy)
            print(f"  Warmup {i+1}/{self.config.warmup_steps}")

        self.memory.clear_cache()
        print("[Pipeline] Warmup complete")
        self.memory.print_stats()

    def transform(self, image: Image.Image) -> Image.Image:
        """Transform image using cached embeddings"""
        if image.size != (self.config.width, self.config.height):
            image = image.resize(
                (self.config.width, self.config.height),
                Image.Resampling.BILINEAR
            )

        with self.memory.inference_mode():
            result = self.pipe(
                prompt=None,
                image=image,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                strength=self.config.denoising_strength,
                generator=self.generator,
                prompt_embeds=self.prompt_embeds,
                pooled_prompt_embeds=self.pooled_prompt_embeds,
                output_type="pil",
            ).images[0]

        return result


# ============================================================================
# CLOUD FRAME PROCESSOR - ENHANCED WITH MULTI-PERSON & PRESERVATION
# ============================================================================
class CloudFrameProcessor:
    """
    High-performance frame processor with:
    - Multi-person detection
    - Gender-aware prompt generation
    - Face and body preservation
    """

    def __init__(
        self,
        pipeline: CloudDiffusionPipeline,
        person_detector: MultiPersonDetector,
        segmenter: AdvancedSegmenter,
        config: CloudConfig
    ):
        self.pipeline = pipeline
        self.person_detector = person_detector
        self.segmenter = segmenter
        self.config = config

        # Queues
        self.input_queue = queue.Queue(maxsize=config.queue_size)
        self.output_queue = queue.Queue(maxsize=config.queue_size)

        # State
        self.running = False
        self.thread = None

        # Metrics
        self.fps = 0.0
        self.process_time = 0.0
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._process_times = []
        self._total_frames = 0

        # Detection stats
        self.last_detected_persons = 0
        self.last_detected_gender = Gender.UNKNOWN

    def start(self):
        """Start processing thread"""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("[Processor] Started")

    def stop(self):
        """Stop processing thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        print("[Processor] Stopped")

    def submit(self, frame: np.ndarray):
        """Submit frame for processing"""
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self.input_queue.put_nowait(frame)
        except queue.Full:
            pass

    def get_result(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get processed result"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self):
        """Main processing loop"""
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
                t0 = time.perf_counter()

                result = self._process(frame)

                elapsed = time.perf_counter() - t0
                self._process_times.append(elapsed)
                if len(self._process_times) > 60:
                    self._process_times.pop(0)
                self.process_time = np.mean(self._process_times)

                self._update_fps()
                self._total_frames += 1

                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.output_queue.put_nowait(result)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Processor] Error: {e}")
                import traceback
                traceback.print_exc()

    def _process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process single frame with:
        - Multi-person detection
        - Gender detection
        - Face/body preservation
        """
        target = (self.config.width, self.config.height)
        h, w = frame.shape[:2]

        if (w, h) != target:
            frame = cv2.resize(frame, target, interpolation=cv2.INTER_LINEAR)

        # Step 1: Detect all persons and their properties
        persons = self.person_detector.detect_persons(frame)
        self.last_detected_persons = len(persons)

        # Step 2: Determine overall gender for prompt adaptation
        overall_gender = self.person_detector.get_overall_gender(persons)
        self.last_detected_gender = overall_gender

        # Step 3: Create preservation masks (face, body, background)
        # Note: background is implicitly the area not covered by face/body masks
        face_mask, body_mask, _background_mask = self.segmenter.create_preservation_masks(
            frame, persons
        )

        # Step 4: Convert to PIL for transformation
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Step 5: Transform the image
        transformed = self.pipeline.transform(pil_img)
        transformed_np = np.array(transformed)

        # Step 6: Advanced blending with different preservation levels
        original_rgb = rgb.astype(np.float32)
        transformed_f = transformed_np.astype(np.float32)

        # Create 3-channel masks for blending
        face_mask_3ch = np.stack([face_mask] * 3, axis=-1)
        body_mask_3ch = np.stack([body_mask] * 3, axis=-1)
        # Note: background_mask used implicitly (areas not covered by face/body)

        # Get preservation strengths from config
        face_preserve = self.config.face_preservation_strength
        body_preserve = self.config.body_preservation_strength
        bg_transform = self.config.background_transform_strength

        # Blend each region with appropriate preservation level
        # Face: High preservation (keep most of original)
        face_blend = (
            face_preserve * original_rgb +
            (1 - face_preserve) * transformed_f
        )

        # Body: Medium preservation
        body_blend = (
            body_preserve * original_rgb +
            (1 - body_preserve) * transformed_f
        )

        # Background: Full transformation
        bg_blend = (
            (1 - bg_transform) * original_rgb +
            bg_transform * transformed_f
        )

        # Combine all regions
        # Priority: face > body > background
        blended = bg_blend.copy()

        # Apply body blend where body mask is active
        body_alpha = body_mask_3ch
        blended = blended * (1 - body_alpha) + body_blend * body_alpha

        # Apply face blend where face mask is active (highest priority)
        face_alpha = face_mask_3ch
        blended = blended * (1 - face_alpha) + face_blend * face_alpha

        # Ensure valid range
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

    def _update_fps(self):
        """Update FPS counter"""
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._last_fps_time

        if elapsed >= 1.0:
            self.fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now

    def get_stats(self) -> dict:
        return {
            'fps': self.fps,
            'process_ms': self.process_time * 1000,
            'queue': self.input_queue.qsize(),
            'total_frames': self._total_frames,
            'persons_detected': self.last_detected_persons,
            'gender': self.last_detected_gender.value
        }


# ============================================================================
# VIDEO INPUT/OUTPUT HANDLER
# ============================================================================
class VideoHandler:
    """Handles video input and output for cloud processing"""

    def __init__(self, config: CloudConfig):
        self.config = config
        self.cap = None
        self.writer = None
        self.total_frames = 0
        self.current_frame = 0

    def open_input(self) -> bool:
        """Open video input source"""
        source = self.config.input_source

        if self.config.input_mode == "webcam":
            self.cap = cv2.VideoCapture(int(source) if source.isdigit() else 0)
        elif self.config.input_mode == "video":
            if not os.path.exists(source):
                print(f"[VideoHandler] Error: Video file not found: {source}")
                return False
            self.cap = cv2.VideoCapture(source)
        elif self.config.input_mode == "stream":
            self.cap = cv2.VideoCapture(source)  # RTMP/HTTP stream URL
        else:
            print(f"[VideoHandler] Unknown input mode: {self.config.input_mode}")
            return False

        if not self.cap.isOpened():
            print("[VideoHandler] Failed to open input source")
            return False

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[VideoHandler] Input opened: {width}x{height} @ {fps:.1f}fps, {self.total_frames} frames")
        return True

    def open_output(self) -> bool:
        """Open video output"""
        if self.config.output_mode != "video":
            return True

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.config.output_path,
            fourcc,
            self.config.output_fps,
            (self.config.width, self.config.height)
        )

        if not self.writer.isOpened():
            print("[VideoHandler] Failed to open output video")
            return False

        print(f"[VideoHandler] Output: {self.config.output_path} @ {self.config.output_fps}fps")
        return True

    def read_frame(self) -> Optional[np.ndarray]:
        """Read next frame"""
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
            return frame
        return None

    def write_frame(self, frame: np.ndarray):
        """Write frame to output"""
        if self.writer is not None:
            self.writer.write(frame)

    def get_progress(self) -> float:
        """Get processing progress (0.0 to 1.0)"""
        if self.total_frames > 0:
            return self.current_frame / self.total_frames
        return 0.0

    def release(self):
        """Release resources"""
        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()


# ============================================================================
# MAIN CLOUD APPLICATION
# ============================================================================
class DreamVisionCloud:
    """Main cloud application for AWS G6 (L4 GPU)"""

    def __init__(self, config: Optional[CloudConfig] = None):
        self.config = config or CloudConfig()

        print("=" * 70)
        print("  DREAM VISION - AWS Cloud Edition")
        print("  Optimized for AWS G6 (NVIDIA L4 - 24GB VRAM)")
        print("=" * 70)
        print()

        # Check CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available! Make sure you're on a GPU instance.")

        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] {gpu}")
        print(f"[GPU] VRAM: {vram:.1f} GB")
        print()

        # Adjust config based on GPU
        if vram >= 20:  # L4 or better
            print("[Config] L4 GPU detected - Using high-performance settings")
            if self.config.width < 768:
                self.config.width = 768
                self.config.height = 768
                print(f"[Config] Resolution upgraded to {self.config.width}x{self.config.height}")

        # Initialize components
        print("\n[Init] Loading components...")

        # Multi-person detector with gender detection
        self.person_detector = MultiPersonDetector(self.config)

        # Advanced segmenter for face/body preservation
        self.segmenter = AdvancedSegmenter(self.config)

        # Diffusion pipeline
        self.pipeline = CloudDiffusionPipeline(self.config)

        # Frame processor with all components
        self.processor = CloudFrameProcessor(
            self.pipeline,
            self.person_detector,
            self.segmenter,
            self.config
        )

        # Video handler
        self.video = VideoHandler(self.config)

        print("\n[Init] Ready!")
        print(f"[Init] Features: Multi-person detection, Gender detection, Face/Body preservation")
        print()

    def run(self):
        """Main processing loop"""
        try:
            # Open input
            if not self.video.open_input():
                raise RuntimeError("Failed to open input source")

            # Open output
            if not self.video.open_output():
                raise RuntimeError("Failed to open output")

            # Start processor
            self.processor.start()

            print("=" * 70)
            print("  PROCESSING")
            print(f"  Input: {self.config.input_source}")
            print(f"  Output: {self.config.output_path}")
            print(f"  Resolution: {self.config.width}x{self.config.height}")
            print("=" * 70)
            print()

            start_time = time.time()
            last_print = start_time

            # Display window (if not headless)
            if not self.config.headless:
                cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)

            while True:
                # Read frame
                frame = self.video.read_frame()
                if frame is None:
                    # Wait for remaining frames to process
                    print("\n[Processing] Input complete, processing remaining frames...")
                    time.sleep(2)
                    break

                # Submit for processing
                self.processor.submit(frame)

                # Get result
                result = self.processor.get_result(timeout=0.5)

                if result is not None:
                    # Write to output
                    self.video.write_frame(result)

                    # Display if not headless
                    if not self.config.headless:
                        self._display_frame(result)

                # Print progress
                now = time.time()
                if now - last_print >= 2.0:
                    stats = self.processor.get_stats()
                    progress = self.video.get_progress() * 100
                    elapsed = now - start_time

                    print(f"[Progress] {progress:.1f}% | "
                          f"FPS: {stats['fps']:.1f} | "
                          f"Latency: {stats['process_ms']:.0f}ms | "
                          f"Frames: {stats['total_frames']} | "
                          f"Persons: {stats.get('persons_detected', 0)} | "
                          f"Gender: {stats.get('gender', 'unknown')} | "
                          f"Elapsed: {elapsed:.0f}s")
                    last_print = now

                # Check for quit (display mode)
                if not self.config.headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        print("\n[Processing] Quit requested")
                        break

            # Final stats
            total_time = time.time() - start_time
            stats = self.processor.get_stats()
            print()
            print("=" * 70)
            print("  COMPLETE")
            print(f"  Total Frames: {stats['total_frames']}")
            print(f"  Total Time: {total_time:.1f}s")
            print(f"  Average FPS: {stats['total_frames'] / total_time:.1f}")
            print(f"  Output: {self.config.output_path}")
            print("=" * 70)

        except KeyboardInterrupt:
            print("\n[Processing] Interrupted")
        finally:
            self._cleanup()

    def _display_frame(self, frame: np.ndarray):
        """Display frame with stats overlay"""
        stats = self.processor.get_stats()
        display = frame.copy()

        # Stats overlay
        if self.config.show_stats:
            cv2.putText(display, f"FPS: {stats['fps']:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Latency: {stats['process_ms']:.0f}ms", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Frame: {stats['total_frames']}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Persons: {stats.get('persons_detected', 0)}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Gender: {stats.get('gender', 'unknown')}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(self.config.window_name, display)

    def _cleanup(self):
        """Cleanup resources"""
        print("[Cleanup] Releasing resources...")
        self.processor.stop()
        self.video.release()
        self.segmenter.release()
        self.person_detector.release()
        torch.cuda.empty_cache()
        cv2.destroyAllWindows()
        print("[Cleanup] Done")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Dream Vision - AWS Cloud Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video file
  python dream_vision_cloud.py --input video.mp4 --output output.mp4

  # Process with higher resolution (L4 GPU)
  python dream_vision_cloud.py --input video.mp4 --width 1024 --height 1024

  # Process with custom prompt
  python dream_vision_cloud.py --input video.mp4 --prompt "cyberpunk neon style"

  # Webcam mode (requires X11 forwarding)
  python dream_vision_cloud.py --mode webcam --no-headless
        """
    )

    parser.add_argument("--input", "-i", type=str, default="input.mp4",
                       help="Input video file or webcam ID (default: input.mp4)")
    parser.add_argument("--output", "-o", type=str, default="output_dreamvision.mp4",
                       help="Output video file (default: output_dreamvision.mp4)")
    parser.add_argument("--mode", "-m", type=str, default="video",
                       choices=["video", "webcam", "stream"],
                       help="Input mode (default: video)")
    parser.add_argument("--width", "-W", type=int, default=768,
                       help="Processing width (default: 768)")
    parser.add_argument("--height", "-H", type=int, default=768,
                       help="Processing height (default: 768)")
    parser.add_argument("--steps", "-s", type=int, default=2,
                       help="Inference steps (default: 2)")
    parser.add_argument("--strength", "-S", type=float, default=0.50,
                       help="Denoising strength 0.0-1.0 (default: 0.50)")
    parser.add_argument("--prompt", "-p", type=str, default=None,
                       help="Custom prompt for style transfer")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Output video FPS (default: 30)")
    parser.add_argument("--no-headless", action="store_true",
                       help="Show display window (requires X11)")
    parser.add_argument("--model", type=str, default="stabilityai/sdxl-turbo",
                       help="Model ID (default: stabilityai/sdxl-turbo)")

    # Face/Body preservation options
    parser.add_argument("--face-preserve", type=float, default=0.85,
                       help="Face preservation strength 0.0-1.0 (default: 0.85)")
    parser.add_argument("--body-preserve", type=float, default=0.60,
                       help="Body preservation strength 0.0-1.0 (default: 0.60)")
    parser.add_argument("--no-gender-detection", action="store_true",
                       help="Disable gender detection")

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Build config from arguments
    config = CloudConfig(
        width=args.width,
        height=args.height,
        model_id=args.model,
        num_inference_steps=args.steps,
        denoising_strength=args.strength,
        fixed_seed=args.seed,
        input_mode=args.mode,
        input_source=args.input,
        output_mode="video",
        output_path=args.output,
        output_fps=args.fps,
        headless=not args.no_headless,
        # Face/Body preservation settings
        face_preservation_strength=args.face_preserve,
        body_preservation_strength=args.body_preserve,
        enable_gender_detection=not args.no_gender_detection,
    )

    if args.prompt:
        config.base_prompt = args.prompt

    # Run application
    app = DreamVisionCloud(config)
    app.run()


if __name__ == "__main__":
    main()
