#!/usr/bin/env python
"""
Test for MockSOFollowerClient.
"""

import pytest
import numpy as np
from lerobot.robots.so_follower.mocks.mock_so_follower import MockSOFollowerClient, MockSOFollowerClientConfig


def test_mock_so_follower_init():
    """Test initialization of MockSOFollowerClient."""
    # Create a mock config instance
    config = MockSOFollowerClientConfig()
    
    # Test that we can create the client without errors
    client = MockSOFollowerClient(config=config)
    
    # Test that it's properly initialized
    assert hasattr(client, 'config')
    assert hasattr(client, '_is_connected')
    assert hasattr(client, '_is_calibrated')


def test_mock_so_follower_properties():
    """Test properties of MockSOFollowerClient."""
    config = MockSOFollowerClientConfig()
    client = MockSOFollowerClient(config=config)
    
    # Test is_connected property
    assert hasattr(client, 'is_connected')
    assert isinstance(client.is_connected, bool)
    
    # Test is_calibrated property
    assert hasattr(client, 'is_calibrated')
    assert isinstance(client.is_calibrated, bool)


def test_mock_so_follower_methods():
    """Test methods of MockSOFollowerClient."""
    config = MockSOFollowerClientConfig()
    client = MockSOFollowerClient(config=config)
    
    # Test connect method (should not raise an error)
    try:
        client.connect()
        assert True  # If no exception raised, it's good
    except Exception as e:
        pytest.fail(f"connect() raised an exception: {e}")
        
    # Test disconnect method (should not raise an error)
    try:
        client.disconnect()
        assert True  # If no exception raised, it's good
    except Exception as e:
        pytest.fail(f"disconnect() raised an exception: {e}")

    # Test get_observation method (should not raise an error when connected)
    try:
        client.connect()
        obs = client.get_observation()
        assert isinstance(obs, dict)
        assert len(obs) > 0
    except Exception as e:
        pytest.fail(f"get_observation() raised an exception: {e}")
        
    # Test send_action method (should not raise an error)
    try:
        action = {}
        result = client.send_action(action)
        assert isinstance(result, dict)
    except Exception as e:
        pytest.fail(f"send_action() raised an exception: {e}")


if __name__ == "__main__":
    test_mock_so_follower_init()
    test_mock_so_follower_properties()
    test_mock_so_follower_methods()
    print("All tests passed!")