"""Multimodal content handler for image processing and temp file management."""

from __future__ import annotations

import base64
import os
import time
import warnings
from pathlib import Path
from contextlib import contextmanager


class MultimodalContentHandler:
    """Handles conversion of multimodal content (images) to temp files."""

    def __init__(self, workspace_dir: Path):
        """Initialize the multimodal content handler.

        Args:
            workspace_dir: Directory for storing temporary image files
        """
        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(exist_ok=True)

    def process_content_blocks(self, message: str | list[dict]) -> tuple[str, list[str]]:
        """Process message content blocks, converting images to temp files.

        Args:
            message: Either a string message or list of content blocks

        Returns:
            Tuple of (final_prompt, temp_file_paths)
        """
        if isinstance(message, str):
            return message, []

        # Handle list of content blocks (multimodal)
        prompt_parts = []
        temp_files = []

        for block in message:
            if block.get("type") == "text":
                prompt_parts.append(block["text"])
            elif block.get("type") == "image":
                source = block.get("source", {})

                # Handle URL-based images
                if source.get("type") == "url":
                    image_url = source.get("url", "")
                    prompt_parts.append(
                        f"Please analyze the image at this URL: {image_url}"
                    )
                    # No temp file for URLs
                else:
                    # Handle base64 images - save to temp file
                    temp_file = self._save_image_to_file(block)
                    if temp_file:  # Only add if we got a file path
                        temp_files.append(temp_file)
                        prompt_parts.append(
                            f"Please read and analyze the image file at this exact path: {Path(temp_file).absolute()}"
                        )

        final_prompt = "\n\n".join(prompt_parts)
        return final_prompt, temp_files

    def _save_image_to_file(self, image_block: dict) -> str | None:
        """Save an image block to a temporary file or return None for URLs.

        Args:
            image_block: Image content block with source data

        Returns:
            Path to the temporary file, or None if image is a URL
        """
        source = image_block.get("source", {})
        source_type = source.get("type")

        # Handle URL-based images - return None to indicate no temp file needed
        if source_type == "url":
            return None

        # Handle base64-encoded images
        if source_type == "base64":
            # Decode base64 image
            image_data = base64.b64decode(source["data"])
            media_type = source.get("media_type", "image/png")
            ext = media_type.split("/")[-1]

            # Create temp file with descriptive name
            timestamp = int(time.time() * 1000)
            temp_filename = f"vision_input_{timestamp}.{ext}"
            temp_filepath = self.workspace_dir / temp_filename

            # Write image data with explicit flush and sync
            with open(temp_filepath, 'wb') as f:
                f.write(image_data)
                f.flush()
                os.fsync(f.fileno())

            # Verify file was written correctly
            if not temp_filepath.exists():
                raise FileNotFoundError(f"Failed to create temp image file: {temp_filepath}")

            file_size = temp_filepath.stat().st_size
            if file_size != len(image_data):
                raise IOError(f"Temp file size mismatch: {file_size} != {len(image_data)}")

            return str(temp_filepath)

        raise ValueError(f"Unsupported image source type: {source_type}")

    @staticmethod
    def cleanup_files(file_paths: list[str]) -> None:
        """Clean up temporary files.

        Args:
            file_paths: List of file paths to delete
        """
        for temp_file in file_paths:
            try:
                Path(temp_file).unlink()
            except FileNotFoundError:
                pass  # File already deleted
            except Exception as e:
                # Log error but don't raise to avoid breaking cleanup flow
                warnings.warn(f"Failed to cleanup temp file {temp_file}: {e}")

    @contextmanager
    def managed_content(self, message: str | list[dict]):
        """Context manager for automatic cleanup of temporary files.

        Args:
            message: Message content to process

        Yields:
            Tuple of (processed_prompt, temp_file_paths)
        """
        prompt, temp_files = self.process_content_blocks(message)
        try:
            yield prompt, temp_files
        finally:
            self.cleanup_files(temp_files)
