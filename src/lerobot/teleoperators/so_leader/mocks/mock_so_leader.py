#!/usr/bin/env python
"""
Mock implementation of SOLeaderClient for lerobot.
This mock simulates the behavior of the real SO leader client without requiring actual hardware connection.
"""

from dataclasses import dataclass
from typing import Dict, Any

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator


@TeleoperatorConfig.register_subclass("mock_so_leader")
@dataclass
class MockSOLeaderClientConfig(TeleoperatorConfig):
    """
    Configuration for the mock SO leader client.
    """
    # Mock configuration parameters
    action_keys: tuple = (
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    )
    mock_cam_keys: tuple = ("front", "wrist")
    mock_cam_height: int = 720
    mock_cam_width: int = 1280
    mock_state_keys: tuple = (
        "shoulder_pan.pos",
        "shoulder_lift.pos", 
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos"
    )
    mock_staleness_warning_s: float = 5.0


class MockSOLeaderClient(Teleoperator):
    """Mock implementation of SOLeaderClient for lerobot."""
    
    config_class = MockSOLeaderClientConfig
    name = "mock_so_leader"

    def __init__(self, config: MockSOLeaderClientConfig):
        super().__init__(config)
        self.config = config
        self.id = config.id
        self.action_keys = config.action_keys
        self.mock_staleness_warning_s = config.mock_staleness_warning_s

        # Mock connection state and data
        self._robot_ref = None
        self._last_action = dict.fromkeys(self.action_keys, 0.0)
        self._last_action_time = None
        self._staleness_warned = False

    def link_robot(self, robot) -> None:
        """
        Mock method to link with a robot instance.
        In real implementation, this would establish connection to actual hardware.
        """
        self._robot_ref = robot
        # For mock, we don't need to actually connect to hardware

    @property
    def action_features(self) -> dict:
        """Mock action features."""
        return dict.fromkeys(self.action_keys, float)

    @property
    def feedback_features(self) -> dict:
        """Mock feedback features - no feedback in mock."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Mock connection status."""
        # In mock, we assume it's connected if robot reference exists
        return self._robot_ref is not None

    def connect(self, calibrate: bool = True) -> None:
        """Mock connection method."""
        if self._robot_ref is None:
            raise ConnectionError(
                "MockSOLeaderClient requires link_robot() to be called with a "
                "connected robot instance before use."
            )
        # In mock, we don't actually connect to hardware
        print("Mock SOLeaderClient connected (simulated)")

    @property
    def is_calibrated(self) -> bool:
        """Mock calibration status."""
        return True

    def calibrate(self) -> None:
        """Mock calibration method."""
        # No actual calibration needed for mock
        pass

    def configure(self) -> None:
        """Mock configuration method."""
        # No actual configuration needed for mock
        pass

    def get_action(self) -> dict:
        """
        Mock action retrieval.
        Returns a random action for testing purposes.
        """
        if self._robot_ref is None:
            return dict(self._last_action)
            
        # For mock, generate random action values
        import numpy as np
        
        # Generate mock action data (random values for demonstration)
        mock_action = {}
        for key in self.action_keys:
            if key == "gripper.pos":
                mock_action[key] = np.random.uniform(0.0, 1.0)  # Gripper position between 0 and 1
            else:
                mock_action[key] = np.random.uniform(-3.14, 3.14)  # Joint positions in radians
                
        self._last_action = mock_action
        self._last_action_time = None  # In mock, we don't track timing
        
        return dict(self._last_action)

    def send_feedback(self, feedback: dict) -> None:
        """Mock feedback sending - no actual hardware interaction."""
        # No feedback in mock implementation
        pass

    def disconnect(self) -> None:
        """Mock disconnection method."""
        self._robot_ref = None
        print("Mock SOLeaderClient disconnected (simulated)")