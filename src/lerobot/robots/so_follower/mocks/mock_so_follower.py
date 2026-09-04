#!/usr/bin/env python
"""
Mock implementation of SOFollowerClient for lerobot.
This mock simulates the behavior of the real SO follower client without requiring actual hardware connection.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Any

import cv2
import numpy as np

from lerobot.robots.robot import Robot
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("mock_so_follower")
@dataclass
class MockSOFollowerClientConfig(RobotConfig):
    """Configuration for the mock SO follower client."""

    # Mock configuration parameters
    mock_cam_keys: tuple = ("front", "wrist")
    mock_cam_height: int = 720
    mock_cam_width: int = 1280
    mock_staleness_warning_s: float = 5.0
    mock_state_keys: tuple = (
        "shoulder_pan.pos",
        "shoulder_lift.pos", 
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos"
    )


class MockSOFollowerClient(Robot):
    """Mock implementation of SOFollowerClient for lerobot."""
    
    config_class = MockSOFollowerClientConfig
    name = "mock_so_follower"

    def __init__(self, config: MockSOFollowerClientConfig):
        super().__init__(config)
        self.config = config
        self.id = config.id
        
        # Mock state and camera configuration
        self.mock_cam_keys = config.mock_cam_keys
        self.mock_cam_height = config.mock_cam_height
        self.mock_cam_width = config.mock_cam_width
        self.mock_staleness_warning_s = config.mock_staleness_warning_s
        self.mock_state_keys = config.mock_state_keys
        
        # Mock connection state
        self._is_connected = False
        self._is_calibrated = True
        
        # Mock data storage
        self.last_frames = {}
        self.last_state = {}
        
        # Mock timing and staleness tracking
        self._last_receive_time = None
        self._staleness_warned = False

    @cached_property
    def _state_ft(self) -> dict[str, type]:
        """Mock state features."""
        return dict.fromkeys(self.mock_state_keys, float)

    @cached_property
    def _state_order(self) -> tuple:
        """Mock state order."""
        return tuple(self._state_ft.keys())

    @cached_property
    def _cameras_ft(self) -> dict:
        """Mock camera features."""
        return dict.fromkeys(self.mock_cam_keys, np.ndarray)
        
    def calibrate(self) -> None:
        """Mock calibration method."""
        # No actual calibration needed for mock
        pass
        
    def configure(self) -> None:
        """Mock configuration method."""
        # No actual configuration needed for mock
        pass

    def _check_staleness(self) -> None:
        """Mock staleness checking."""
        if self._last_receive_time is None:
            return
        elapsed = time.monotonic() - self._last_receive_time
        if elapsed > self.mock_staleness_warning_s:
            if not self._staleness_warned:
                logging.warning(
                    f"No new observation from Mock Host for {elapsed:.1f}s "
                    f"(threshold={self.mock_staleness_warning_s}s). "
                    "Mock Host may have crashed or disconnected."
                )
                self._staleness_warned = True
        else:
            self._staleness_warned = False

    def _poll_and_get_latest_message(self):
        """Mock polling method."""
        # Return mock data for testing purposes
        return None  # Mock returns None to use cached data

    def _decode_image(self, jpeg: bytes):
        """Mock image decoding."""
        # Return mock image or None
        return None

    @cached_property
    def action_features(self) -> dict:
        """Mock action features - same as state order for compatibility."""
        return dict.fromkeys(self._state_order, float)

    @cached_property
    def observation_features(self) -> dict:
        """Mock observation features."""
        return {**self._state_ft, **self._cameras_ft}

    @property
    def is_connected(self) -> bool:
        """Mock connection status."""
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """Mock calibration status."""
        return self._is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        """Mock connection method."""
        # Simulate connection process
        logging.info("Mock SOFollowerClient connecting...")
        self._is_connected = True
        self._is_calibrated = calibrate
        logging.info("Mock SOFollowerClient connected successfully.")

    def disconnect(self) -> None:
        """Mock disconnection method."""
        logging.info("Mock SOFollowerClient disconnecting...")
        self._is_connected = False
        logging.info("Mock SOFollowerClient disconnected.")

    def get_observation(self) -> dict:
        """Mock observation retrieval."""
        # Generate mock data
        if not self._is_connected:
            raise RuntimeError("Cannot get observation when not connected")
            
        # Simulate receiving data
        self._last_receive_time = time.monotonic()
        
        # Mock state data (random values for testing)
        mock_state = {}
        for key in self._state_order:
            if key == "gripper.pos":
                mock_state[key] = np.random.uniform(0.0, 1.0)  # Gripper position between 0 and 1
            else:
                mock_state[key] = np.random.uniform(-3.14, 3.14)  # Joint positions in radians
        
        # Mock frame data
        mock_frames = {}
        for cam_name in self.mock_cam_keys:
            # Create a mock RGB image (720x1280)
            mock_frames[cam_name] = np.random.randint(0, 255, (self.mock_cam_height, self.mock_cam_width, 3), dtype=np.uint8)
            
        # Combine state and frames
        obs = dict(mock_state)
        obs.update(mock_frames)
        
        return obs

    def send_action(self, action: dict) -> dict:
        """Mock action sending."""
        logging.info("Mock SOFollowerClient received action (but does not send to hardware)")
        return {}