"""TrOCR inference helper (uses HuggingFace Transformers)."""
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch


class TrOCRRunner:
    def __init__(self, model_name='microsoft/trocr-small-handwritten', device=None, use_fast: bool = False):
        """Initialize TrOCR runner.

        Args:
            model_name: HF model id
            device: 'cuda' or 'cpu' or None to auto-select
            use_fast: whether to use fast tokenizer; default False to avoid slow->fast conversion errors
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = TrOCRProcessor.from_pretrained(model_name, use_fast=use_fast)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        except ImportError as ie:
            # Commonly happens when `protobuf` is not installed in the environment
            raise ImportError(
                "A dependency is missing when loading TrOCR. This commonly means the `protobuf` package is not installed.\n"
                "Fix: activate your environment and run `pip install protobuf` (or `pip install -r requirements.txt`)\n"
                "Then restart the Python process and try again. Original error: {}".format(ie)
            ) from ie
        except Exception as e:
            # Catch other tokenizer/model conversion issues and provide guidance
            raise RuntimeError(
                "Failed to load TrOCR processor/model. Try installing/upgrading `transformers` and tokenizer deps,\n"
                "and call TrOCRRunner(..., use_fast=False) to avoid fast tokenizer conversion.\n"
                f"Original error: {e}"
            ) from e

    def recognize_image(self, pil_image: Image.Image):
        pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        pred_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return pred_text
