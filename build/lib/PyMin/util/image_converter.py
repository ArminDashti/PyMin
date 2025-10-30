import base64
import io
import mimetypes
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Union, Tuple


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


def base64_to_image(base64_string, output_path=None):
    image_data = base64.b64decode(base64_string)
    if output_path:
        with open(output_path, "wb") as image_file:
            image_file.write(image_data)
    return Image.open(io.BytesIO(image_data))


def image_to_tensor(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    tensor = torch.from_numpy(image_array)
    return tensor


def tensor_to_image(tensor, output_path=None):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3 and tensor.shape[0] in [1, 3, 4]:
        tensor = tensor.permute(1, 2, 0)
    numpy_array = tensor.numpy()
    if numpy_array.dtype != np.uint8:
        numpy_array = (numpy_array * 255).astype(np.uint8)
    image = Image.fromarray(numpy_array)
    if output_path:
        image.save(output_path)
    return image


def image_to_numpy(image_path):
    image = Image.open(image_path)
    numpy_array = np.array(image)
    return numpy_array


def numpy_to_image(numpy_array, output_path):
    if numpy_array.dtype != np.uint8:
        numpy_array = (numpy_array * 255).astype(np.uint8)
    image = Image.fromarray(numpy_array)
    if output_path:
        image.save(output_path)
    return image


def validate_image_path(image_path: Union[str, Path], supported_formats: set = None) -> Tuple[bool, str]:
    """Validate if the image path exists and is in a supported format."""
    if supported_formats is None:
        supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
    
    try:
        path = Path(image_path)
        if not path.exists():
            return False, f"Image file does not exist: {image_path}"
        
        if not path.is_file():
            return False, f"Path is not a file: {image_path}"
        
        suffix = path.suffix.lower()
        if suffix not in supported_formats:
            return False, f"Unsupported image format: {suffix}. Supported formats: {', '.join(supported_formats)}"
        
        return True, "Valid image file"
    except Exception as e:
        return False, f"Error validating image path: {str(e)}"


def encode_image_to_base64(image_path: Union[str, Path], supported_formats: set = None) -> Tuple[bool, str, str]:
    """
    Encode an image file to base64 format.
    
    Args:
        image_path: Path to the image file
        supported_formats: Set of supported image file extensions (optional)
    
    Returns:
        tuple: (success: bool, data: str, mime_type: str)
    """
    if supported_formats is None:
        supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
    
    try:
        path = Path(image_path)
        
        # Validate image path
        is_valid, error_msg = validate_image_path(path, supported_formats)
        if not is_valid:
            return False, "", error_msg
        
        # Read and encode the image
        with open(path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type or not mime_type.startswith('image/'):
            mime_type = 'image/jpeg'  # Default fallback
        
        return True, image_data, mime_type
        
    except Exception as e:
        return False, "", f"Error encoding image: {str(e)}"
