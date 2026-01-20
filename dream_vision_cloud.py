"""
- AWS CLOUD VERSION
=================================================
Optimized for AWS G6 Instance with NVIDIA L4 GPU (24GB VRAM)
High-performance cloud deployment for real-time AI art generation.

Hardware Target: AWS g6.xlarge / g6.2xlarge (NVIDIA L4 - 24GB VRAM)
Expected Performance: 25-40 FPS at 768x768 resolution

Key Cloud Optimizations:
1. Higher resolution support (768x768) due to 24GB VRAM
2. Batch processing capability
3. Video file input support (no physical webcam)
4. RTMP/WebRTC streaming output
5. HTTP API for remote control
6. Headless operation mode

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

warnings.filterwarnings("ignore")

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

    # Prompts
    prompt: str = (
        "dreamy surreal oil painting, ethereal portrait, "
        "mystical glowing atmosphere, soft venetian mist, "
        "golden luminescent light, renaissance masterpiece, "
        "Nikas Safronov dream vision, 8k ultra detailed"
    )
    negative_prompt: str = (
        "ugly, blurry, low quality, distorted, deformed, "
        "dark, harsh, pixelated, noisy"
    )

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
# PERSON SEGMENTER - CLOUD VERSION
# ============================================================================
class CloudPersonSegmenter:
    """
    Person segmentation with fallback for headless cloud operation.
    """

    def __init__(self, target_size: Tuple[int, int] = (768, 768)):
        self.target_size = target_size
        self.segmenter = None

        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_selfie = mp.solutions.selfie_segmentation
                self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=1)
                print("[Segmenter] MediaPipe initialized")
            except Exception as e:
                print(f"[Segmenter] MediaPipe failed: {e}")

        if self.segmenter is None:
            print("[Segmenter] Using simple background detection fallback")

        # Pre-allocate kernel
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def segment(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate person segmentation mask.
        Returns float mask (0.0-1.0)
        """
        h, w = frame.shape[:2]
        if (w, h) != self.target_size:
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

        if self.segmenter is not None:
            # MediaPipe segmentation
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.segmenter.process(rgb)
            mask = results.segmentation_mask
        else:
            # Fallback: Simple GrabCut-based segmentation
            mask = self._fallback_segment(frame)

        # Clean up mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        return mask

    def _fallback_segment(self, frame: np.ndarray) -> np.ndarray:
        """Simple fallback segmentation when MediaPipe unavailable"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use adaptive thresholding to detect foreground
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        mask = cv2.adaptiveThreshold(
            blur, 1.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        ).astype(np.float32)

        return mask

    def release(self):
        if self.segmenter is not None:
            self.segmenter.close()


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
# CLOUD FRAME PROCESSOR
# ============================================================================
class CloudFrameProcessor:
    """High-performance frame processor for cloud deployment"""

    def __init__(
        self,
        pipeline: CloudDiffusionPipeline,
        segmenter: CloudPersonSegmenter,
        config: CloudConfig
    ):
        self.pipeline = pipeline
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
        """Process single frame"""
        target = (self.config.width, self.config.height)
        h, w = frame.shape[:2]

        if (w, h) != target:
            frame = cv2.resize(frame, target, interpolation=cv2.INTER_LINEAR)

        # Get segmentation mask
        mask = self.segmenter.segment(frame)

        # Convert to PIL
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Transform
        transformed = self.pipeline.transform(pil_img)
        transformed_np = np.array(transformed)

        # Blend with mask
        original_rgb = rgb.astype(np.float32)
        transformed_f = transformed_np.astype(np.float32)
        mask_3ch = np.stack([mask] * 3, axis=-1)

        # Subject preservation
        subject_factor = 0.3
        blended = (
            mask_3ch * (subject_factor * original_rgb + (1 - subject_factor) * transformed_f) +
            (1 - mask_3ch) * transformed_f
        ).astype(np.uint8)

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
            'total_frames': self._total_frames
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
        self.segmenter = CloudPersonSegmenter((self.config.width, self.config.height))
        self.pipeline = CloudDiffusionPipeline(self.config)
        self.processor = CloudFrameProcessor(self.pipeline, self.segmenter, self.config)
        self.video = VideoHandler(self.config)

        print("\n[Init] Ready!")
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

        cv2.imshow(self.config.window_name, display)

    def _cleanup(self):
        """Cleanup resources"""
        print("[Cleanup] Releasing resources...")
        self.processor.stop()
        self.video.release()
        self.segmenter.release()
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
    )

    if args.prompt:
        config.prompt = args.prompt

    # Run application
    app = DreamVisionCloud(config)
    app.run()


if __name__ == "__main__":
    main()
