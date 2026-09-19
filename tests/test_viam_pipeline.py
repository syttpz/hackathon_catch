import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from vision.viam_pipeline import get_box_midpoints


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_midpoint_from_viam_detection(self):
        detection = SimpleNamespace(class_name="red", confidence=0.9,
                                    x_min=10, y_min=20, x_max=31, y_max=45)
        service = SimpleNamespace(get_detections_from_camera=AsyncMock(return_value=[detection]))
        with patch("vision.viam_pipeline.VisionClient.from_robot", return_value=service):
            result = await get_box_midpoints(object())
        self.assertEqual(result[0]["midpoint_px"], [20.5, 32.5])
        service.get_detections_from_camera.assert_awaited_once_with("cam", timeout=10)
