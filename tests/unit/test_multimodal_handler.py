"""Unit tests for MultimodalContentHandler - testing image handling and temp file management."""

from __future__ import annotations

import base64
import pytest
from pathlib import Path

from clowclow.multimodal_handler import MultimodalContentHandler


class TestBase64ImageProcessing:
    """Test base64 image saving and processing."""

    def test_save_base64_image(self, test_workspace):
        """Test saving base64 image to temp file."""
        handler = MultimodalContentHandler(test_workspace)

        # Create fake PNG header
        fake_png = b"\x89PNG\r\n\x1a\n" + b"fake image data"
        image_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(fake_png).decode()
            }
        }

        filepath = handler._save_image_to_file(image_block)

        assert filepath is not None
        assert Path(filepath).exists()
        assert Path(filepath).suffix == ".png"

        # Verify content
        with open(filepath, "rb") as f:
            saved_data = f.read()
        assert saved_data == fake_png

    def test_save_jpeg_image(self, test_workspace):
        """Test saving JPEG image with correct extension."""
        handler = MultimodalContentHandler(test_workspace)

        fake_jpeg = b"\xff\xd8\xff" + b"fake jpeg data"
        image_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(fake_jpeg).decode()
            }
        }

        filepath = handler._save_image_to_file(image_block)

        assert filepath is not None
        assert Path(filepath).suffix == ".jpeg"

    def test_url_image_returns_none(self, test_workspace):
        """Test that URL images return None (no temp file needed)."""
        handler = MultimodalContentHandler(test_workspace)

        image_block = {
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.com/image.png"
            }
        }

        filepath = handler._save_image_to_file(image_block)
        assert filepath is None


class TestContentProcessing:
    """Test content block processing."""

    def test_process_text_blocks(self, test_workspace):
        """Test processing text-only content."""
        handler = MultimodalContentHandler(test_workspace)

        content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"}
        ]

        prompt, temp_files = handler.process_content_blocks(content)

        assert "Hello" in prompt
        assert "World" in prompt
        assert len(temp_files) == 0

    def test_process_base64_image_blocks(self, test_workspace):
        """Test processing base64 image blocks."""
        handler = MultimodalContentHandler(test_workspace)

        fake_png = b"\x89PNG\r\n\x1a\n" + b"fake"
        content = [
            {"type": "text", "text": "Analyze this"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(fake_png).decode()
                }
            }
        ]

        prompt, temp_files = handler.process_content_blocks(content)

        assert "Analyze this" in prompt
        assert len(temp_files) == 1
        assert Path(temp_files[0]).exists()
        assert temp_files[0] in prompt

    def test_process_url_image_blocks(self, test_workspace):
        """Test processing URL image blocks."""
        handler = MultimodalContentHandler(test_workspace)

        content = [
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.com/image.png"
                }
            }
        ]

        prompt, temp_files = handler.process_content_blocks(content)

        assert "https://example.com/image.png" in prompt
        assert len(temp_files) == 0  # No temp files for URLs


class TestContextManager:
    """Test context manager behavior."""

    def test_managed_content_with_text(self, test_workspace):
        """Test managed_content with text-only content."""
        handler = MultimodalContentHandler(test_workspace)

        with handler.managed_content("Hello world") as (prompt, temp_files):
            assert prompt == "Hello world"
            assert len(temp_files) == 0

    def test_managed_content_with_images(self, test_workspace):
        """Test managed_content with images creates and cleans up temp files."""
        handler = MultimodalContentHandler(test_workspace)

        fake_png = b"\x89PNG\r\n\x1a\n" + b"fake"
        content = [
            {"type": "text", "text": "Test"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(fake_png).decode()
                }
            }
        ]

        temp_file_path = None
        with handler.managed_content(content) as (prompt, temp_files):
            assert len(temp_files) == 1
            temp_file_path = temp_files[0]
            assert Path(temp_file_path).exists()

        # File should be cleaned up after context exit
        assert not Path(temp_file_path).exists()

    def test_managed_content_cleanup_on_error(self, test_workspace):
        """Test that temp files are cleaned up even if error occurs."""
        handler = MultimodalContentHandler(test_workspace)

        fake_png = b"\x89PNG\r\n\x1a\n" + b"fake"
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(fake_png).decode()
                }
            }
        ]

        temp_file_path = None
        try:
            with handler.managed_content(content) as (prompt, temp_files):
                temp_file_path = temp_files[0]
                raise ValueError("Test error")
        except ValueError:
            pass

        # File should still be cleaned up
        assert not Path(temp_file_path).exists()

    def test_managed_content_with_list_content(self, test_workspace):
        """Test managed_content handles list content correctly."""
        handler = MultimodalContentHandler(test_workspace)

        content = [
            {"type": "text", "text": "Multiple"},
            {"type": "text", "text": "Text blocks"}
        ]

        with handler.managed_content(content) as (prompt, temp_files):
            assert "Multiple" in prompt
            assert "Text blocks" in prompt
            assert len(temp_files) == 0


class TestErrorHandling:
    """Test error handling in multimodal handler."""

    def test_unsupported_image_source_type(self, test_workspace):
        """Test that unsupported image source types raise ValueError."""
        handler = MultimodalContentHandler(test_workspace)

        image_block = {
            "type": "image",
            "source": {
                "type": "unsupported",
                "data": "something"
            }
        }

        with pytest.raises(ValueError, match="Unsupported image source type"):
            handler._save_image_to_file(image_block)

    def test_invalid_base64_data(self, test_workspace):
        """Test handling of invalid base64 data."""
        handler = MultimodalContentHandler(test_workspace)

        image_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "invalid_base64!!!"
            }
        }

        # Should raise an error during base64 decode
        with pytest.raises(Exception):
            handler._save_image_to_file(image_block)
