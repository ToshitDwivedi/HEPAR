"""
HEPAR Springer Benchmarks - Utilities
======================================
Image processing, quadtree decomposition, and helper functions.

Key Features:
- Image loading with synthetic fallback
- Quadtree decomposition with gradient computation
- Gray code sorting for HEPAR optimization
- Morton code (Z-order) calculations
"""

import numpy as np
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.ndimage import sobel

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class LeafNode:
    """Quadtree leaf node with SPAE data."""
    depth: int
    row: int
    col: int
    size: int
    value: float  # Intensity (amplitude encoding)
    gradient: float = 0.0  # Edge info (phase encoding)
    leaf_index: int = 0
    morton_code: int = 0  # For Gray code sorting


def interleave_bits(x: int, y: int, bits: int = 5) -> int:
    """
    Compute Morton code (Z-order curve) for 2D coordinates.
    
    Morton codes preserve spatial locality in 1D representation.
    
    Args:
        x: X coordinate
        y: Y coordinate
        bits: Number of bits per coordinate
        
    Returns:
        Morton code (interleaved bits)
    """
    result = 0
    for i in range(bits):
        result |= ((x >> i) & 1) << (2*i + 1)
        result |= ((y >> i) & 1) << (2*i)
    return result


def gray_code(n: int) -> int:
    """
    Convert integer to Gray code.
    
    Gray code ensures adjacent values differ by only 1 bit,
    minimizing qubit transitions during encoding.
    """
    return n ^ (n >> 1)


def inverse_gray_code(g: int) -> int:
    """Convert Gray code back to integer."""
    n = 0
    while g:
        n ^= g
        g >>= 1
    return n


def hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two integers."""
    return bin(a ^ b).count('1')


class ImageProcessor:
    """
    Image loading and quadtree decomposition.
    
    Supports:
    - Loading from file (PNG, JPG)
    - Synthetic image generation (fallback)
    - Gradient computation for SPAE
    - Adaptive quadtree decomposition
    """
    
    def __init__(self, size: int = 32):
        """
        Initialize processor.
        
        Args:
            size: Target image size (assumes square)
        """
        self.size = size
    
    def load_image(self, name: str, data_dir: str = 'data') -> np.ndarray:
        """
        Load image from file or generate synthetic.
        
        Args:
            name: Image identifier ('Lena', 'MRI', 'Noise', or filename)
            data_dir: Directory containing image files
            
        Returns:
            Normalized grayscale image [0, 1]
        """
        # Try to load from file first
        if HAS_PIL:
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                file_path = os.path.join(data_dir, f"{name.lower()}_32{ext}")
                if os.path.exists(file_path):
                    img = Image.open(file_path).convert('L')
                    img = img.resize((self.size, self.size), Image.Resampling.BILINEAR)
                    return np.array(img) / 255.0
                
                # Also try without size suffix
                file_path = os.path.join(data_dir, f"{name.lower()}{ext}")
                if os.path.exists(file_path):
                    img = Image.open(file_path).convert('L')
                    img = img.resize((self.size, self.size), Image.Resampling.BILINEAR)
                    return np.array(img) / 255.0
        
        # Generate synthetic images as fallback
        img = self._generate_synthetic(name)
        
        # Save synthetic image so user can see it in data/
        if HAS_PIL:
            try:
                os.makedirs(data_dir, exist_ok=True)
                save_path = os.path.join(data_dir, f"{name.lower()}_{self.size}.png")
                Image.fromarray((img * 255).astype(np.uint8)).save(save_path)
                print(f"  [Info] Generated and saved synthetic image: {save_path}")
            except Exception as e:
                print(f"  [Warning] Could not save synthetic image: {e}")
                
        return img
    
    def _generate_synthetic(self, name: str) -> np.ndarray:
        """
        Generate synthetic test images.
        
        - Lena: Gradient with sinusoidal pattern (simulates natural image)
        - MRI: Sparse circular blob (80% background - tests compression)
        - Noise: Gaussian random texture (stress test)
        - MNIST: Simple digit-like pattern
        """
        if name.lower() in ['lena', 'gradient']:
            # Lena-like: sinusoidal gradient pattern
            x, y = np.meshgrid(
                np.linspace(0, 1, self.size),
                np.linspace(0, 1, self.size)
            )
            return np.clip(0.5 + 0.4 * np.sin(3*x*np.pi) * np.cos(3*y*np.pi), 0, 1)
        
        elif name.lower() in ['mri', 'medical', 'sparse']:
            # MRI-like: sparse circular blob
            img = np.zeros((self.size, self.size))
            center = self.size // 2
            y, x = np.ogrid[:self.size, :self.size]
            
            # Main circular region
            mask = (x - center)**2 + (y - center)**2 <= (self.size // 3)**2
            img[mask] = 0.9
            
            # Add some internal structure
            inner_mask = (x - center)**2 + (y - center)**2 <= (self.size // 6)**2
            img[inner_mask] = 0.6
            
            return img
        
        elif name.lower() in ['noise', 'texture', 'random']:
            # Gaussian noise texture
            np.random.seed(42)  # Reproducible
            return np.clip(np.random.rand(self.size, self.size), 0, 1)
        
        elif name.lower() in ['mnist', 'digit']:
            # Simple digit-like pattern (number 7)
            img = np.zeros((self.size, self.size))
            s = self.size
            # Horizontal line at top
            img[s//8:s//6, s//4:3*s//4] = 0.9
            # Diagonal line
            for i in range(s//3):
                row = s//6 + i
                col = 3*s//4 - i
                if 0 <= row < s and 0 <= col < s:
                    img[max(0,row-1):min(s,row+2), max(0,col-1):min(s,col+2)] = 0.9
            return img
        
        else:
            # Default: uniform gray
            return np.ones((self.size, self.size)) * 0.5
    
    def compute_gradient(self, image: np.ndarray) -> np.ndarray:
        """
        Compute Sobel gradient magnitude for SPAE phase encoding.
        
        Returns:
            Normalized gradient magnitude [0, 1]
        """
        gx = sobel(image, axis=1)
        gy = sobel(image, axis=0)
        grad = np.sqrt(gx**2 + gy**2)
        
        if grad.max() > 0:
            grad = grad / grad.max()
        
        return grad
    
    def quadtree_decompose(
        self,
        image: np.ndarray,
        threshold: float = 0.02,
        min_size: int = 1,
        max_depth: int = None
    ) -> Tuple[List[LeafNode], np.ndarray]:
        """
        Decompose image into quadtree leaves with gradient info.
        
        Args:
            image: Input image [0, 1]
            threshold: Variance threshold for splitting
            min_size: Minimum block size (pixels)
            max_depth: Maximum tree depth (None = log2(size))
            
        Returns:
            Tuple of (list of leaves, density map)
        """
        leaves = []
        density = np.zeros_like(image)
        gradient = self.compute_gradient(image)
        
        if max_depth is None:
            max_depth = int(np.log2(self.size))
        
        def recurse(r: int, c: int, sz: int, depth: int):
            block = image[r:r+sz, c:c+sz]
            
            # Split condition: variance > threshold AND not at max depth
            should_split = (
                sz > min_size and
                np.var(block) > threshold and
                depth < max_depth
            )
            
            if should_split:
                h = sz // 2
                recurse(r, c, h, depth + 1)
                recurse(r, c + h, h, depth + 1)
                recurse(r + h, c, h, depth + 1)
                recurse(r + h, c + h, h, depth + 1)
            else:
                # Create leaf node
                grad_block = gradient[r:r+sz, c:c+sz]
                morton = interleave_bits(c // sz, r // sz)
                
                leaves.append(LeafNode(
                    depth=depth,
                    row=r,
                    col=c,
                    size=sz,
                    value=float(np.mean(block)),
                    gradient=float(np.mean(grad_block)),
                    morton_code=morton
                ))
                
                # Update density map (higher depth = higher density)
                density[r:r+sz, c:c+sz] = 1.0 / sz
        
        recurse(0, 0, self.size, 0)
        
        # Assign leaf indices
        for i, leaf in enumerate(leaves):
            leaf.leaf_index = i
        
        # Normalize density map
        if density.max() > 0:
            density = density / density.max()
        
        return leaves, density
    
    def gray_code_sort(self, leaves: List[LeafNode]) -> Tuple[List[LeafNode], int]:
        """
        Sort leaves by Gray code to minimize Hamming distance.
        
        This is Pillar 2 of HEPAR: minimizing qubit transitions
        during encoding for reduced gate count.
        
        Returns:
            Tuple of (sorted leaves, Hamming distance savings)
        """
        if len(leaves) <= 1:
            return leaves, 0
        
        # Convert Morton codes to Gray codes
        for leaf in leaves:
            leaf.morton_code = gray_code(leaf.morton_code)
        
        # Sort by Gray code
        sorted_leaves = sorted(leaves, key=lambda x: x.morton_code)
        
        # Reassign indices
        for i, leaf in enumerate(sorted_leaves):
            leaf.leaf_index = i
        
        # Calculate Hamming distance improvement
        unsorted_dist = sum(
            hamming_distance(leaves[i].leaf_index, leaves[i+1].leaf_index)
            for i in range(len(leaves) - 1)
        )
        sorted_dist = sum(
            hamming_distance(sorted_leaves[i].leaf_index, sorted_leaves[i+1].leaf_index)
            for i in range(len(sorted_leaves) - 1)
        )
        
        savings = max(0, unsorted_dist - sorted_dist)
        return sorted_leaves, savings
    
    def compute_adaptive_threshold(self, image: np.ndarray) -> float:
        """
        Compute adaptive variance threshold based on image entropy.
        
        Key insight: threshold should be proportional to global image variance
        to achieve consistent compression across different image types.
        
        Args:
            image: Input image [0, 1]
            
        Returns:
            Optimal variance threshold for quadtree splitting
        """
        global_var = np.var(image)
        # Use 10% of global variance as threshold (empirically optimal)
        # Add floor to prevent over-splitting on uniform images
        return max(0.005, global_var * 0.1)
    
    def quadtree_decompose_with_target(
        self,
        image: np.ndarray,
        target_leaves: int = 100,
        min_size: int = 1
    ) -> Tuple[List[LeafNode], np.ndarray, float]:
        """
        Decompose image with adaptive threshold to achieve target leaf count.
        
        Uses entropy-aware binary search with guaranteed complete coverage.
        
        Args:
            image: Input image [0, 1]
            target_leaves: Target number of leaves
            min_size: Minimum block size
            
        Returns:
            Tuple of (leaves, density map, threshold used)
        """
        # Start with entropy-based threshold estimate
        adaptive_thresh = self.compute_adaptive_threshold(image)
        low_thresh, high_thresh = 0.001, max(0.5, adaptive_thresh * 10)
        
        best_leaves = None
        best_density = None
        best_thresh = adaptive_thresh
        best_diff = float('inf')
        
        for iteration in range(20):  # More iterations for better precision
            mid_thresh = (low_thresh + high_thresh) / 2
            leaves, density = self.quadtree_decompose(image, threshold=mid_thresh, min_size=min_size)
            
            diff = abs(len(leaves) - target_leaves)
            
            # Track best result closest to target
            if diff < best_diff:
                best_diff = diff
                best_leaves = leaves
                best_density = density
                best_thresh = mid_thresh
            
            # Early termination if we hit exact target
            if len(leaves) == target_leaves:
                break
            
            if len(leaves) < target_leaves:
                # Need more leaves -> lower threshold
                high_thresh = mid_thresh
            else:
                # Need fewer leaves -> higher threshold
                low_thresh = mid_thresh
        
        # Verify coverage (critical for valid metrics)
        if best_leaves and not verify_complete_coverage(best_leaves, image.shape):
            # Fallback: Use higher threshold to ensure coverage
            best_leaves, best_density = self.quadtree_decompose(
                image, threshold=best_thresh * 2, min_size=min_size
            )
        
        return best_leaves, best_density, best_thresh


def build_leaf_coverage_mask(leaves: List[LeafNode], shape: Tuple[int, int]) -> np.ndarray:
    """
    Build a binary mask indicating which pixels are covered by leaves.
    
    Used for masked SSIM/PSNR calculations that only compare reconstructed regions.
    
    Args:
        leaves: List of quadtree leaves
        shape: Image shape (H, W)
        
    Returns:
        Binary mask where True = pixel covered by a leaf
    """
    mask = np.zeros(shape, dtype=bool)
    for leaf in leaves:
        mask[leaf.row:leaf.row+leaf.size, leaf.col:leaf.col+leaf.size] = True
    return mask


def verify_complete_coverage(leaves: List[LeafNode], shape: Tuple[int, int]) -> bool:
    """
    Verify that leaves completely tile the image with no overlap or gaps.
    
    Critical for valid reconstruction - if this fails, metrics are meaningless.
    
    Args:
        leaves: List of quadtree leaves
        shape: Image shape (H, W)
        
    Returns:
        True if coverage is complete and valid
    """
    total_area = sum(leaf.size * leaf.size for leaf in leaves)
    expected_area = shape[0] * shape[1]
    
    if total_area != expected_area:
        return False
    
    # Check for overlaps using a count map
    count_map = np.zeros(shape, dtype=int)
    for leaf in leaves:
        count_map[leaf.row:leaf.row+leaf.size, leaf.col:leaf.col+leaf.size] += 1
    
    # Every pixel should be covered exactly once
    return np.all(count_map == 1)


class DatasetManager:
    """Manage datasets for benchmarking."""
    
    STANDARD_DATASETS = ['Lena', 'MRI', 'Noise']
    
    def __init__(self, data_dir: str = 'data', size: int = 32):
        self.data_dir = data_dir
        self.size = size
        self.processor = ImageProcessor(size)
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
    
    def get_all_images(self) -> dict:
        """Load all standard datasets."""
        return {
            name: self.processor.load_image(name, self.data_dir)
            for name in self.STANDARD_DATASETS
        }
    
    def get_image_with_metadata(self, name: str) -> dict:
        """
        Get image with preprocessing metadata.
        
        Returns dict with:
        - 'image': normalized image
        - 'gradient': Sobel gradient map
        - 'leaves': quadtree decomposition
        - 'density': density map
        - 'complexity': image complexity score
        """
        image = self.processor.load_image(name, self.data_dir)
        gradient = self.processor.compute_gradient(image)
        leaves, density = self.processor.quadtree_decompose(image)
        sorted_leaves, savings = self.processor.gray_code_sort(leaves)
        
        # Estimate complexity (affects HEPAR compression)
        complexity = np.var(image) * 10  # Higher variance = less compressible
        
        return {
            'image': image,
            'gradient': gradient,
            'leaves': sorted_leaves,
            'density': density,
            'num_leaves': len(sorted_leaves),
            'hamming_savings': savings,
            'complexity': complexity
        }


if __name__ == "__main__":
    # Demo
    print("Testing Image Processor and Quadtree Decomposition...")
    
    proc = ImageProcessor(32)
    
    for name in ['Lena', 'MRI', 'Noise']:
        img = proc.load_image(name)
        leaves, density = proc.quadtree_decompose(img)
        sorted_leaves, savings = proc.gray_code_sort(leaves)
        
        cr = (32 * 32) / len(sorted_leaves)
        
        print(f"\n{name}:")
        print(f"  Image range: [{img.min():.2f}, {img.max():.2f}]")
        print(f"  Quadtree leaves: {len(sorted_leaves)}")
        print(f"  Compression ratio: {cr:.2f}x")
        print(f"  Hamming distance savings: {savings}")
