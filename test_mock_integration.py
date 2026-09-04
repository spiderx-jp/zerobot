#!/usr/bin/env python
"""
Simple test to verify that the mock implementations are correctly integrated
and can be imported and instantiated without errors.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

def test_mock_imports():
    """Test that mock modules can be imported."""
    print("Testing mock module imports...")
    
    try:
        from lerobot.robots.so_follower.mocks.mock_so_follower import MockSOFollowerClient
        print("✓ MockSOFollowerClient imported successfully")
    except Exception as e:
        print(f"✗ Failed to import MockSOFollowerClient: {e}")
        return False
        
    try:
        from lerobot.teleoperators.so_leader.mocks.mock_so_leader import MockSOLeaderClient
        print("✓ MockSOLeaderClient imported successfully")
    except Exception as e:
        print(f"✗ Failed to import MockSOLeaderClient: {e}")
        return False
        
    return True

def test_mock_instantiation():
    """Test that mock classes can be instantiated."""
    print("\nTesting mock class instantiation...")
    
    try:
        from lerobot.robots.so_follower.mocks.mock_so_follower import MockSOFollowerClient, MockSOFollowerClientConfig
        from lerobot.teleoperators.so_leader.mocks.mock_so_leader import MockSOLeaderClient, MockSOLeaderClientConfig
        
        # Test SO follower mock instantiation
        follower_config = MockSOFollowerClientConfig()
        follower_mock = MockSOFollowerClient(follower_config)
        print("✓ MockSOFollowerClient instantiated successfully")
        
        # Test SO leader mock instantiation
        leader_config = MockSOLeaderClientConfig()
        leader_mock = MockSOLeaderClient(leader_config)
        print("✓ MockSOLeaderClient instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to instantiate mock classes: {e}")
        return False

def test_mock_basic_methods():
    """Test basic methods of mock classes."""
    print("\nTesting basic mock methods...")
    
    try:
        from lerobot.robots.so_follower.mocks.mock_so_follower import MockSOFollowerClient, MockSOFollowerClientConfig
        from lerobot.teleoperators.so_leader.mocks.mock_so_leader import MockSOLeaderClient, MockSOLeaderClientConfig
        
        # Test SO follower mock methods
        follower_config = MockSOFollowerClientConfig()
        follower_mock = MockSOFollowerClient(follower_config)
        
        # Test properties
        print(f"  - is_connected: {follower_mock.is_connected}")
        print(f"  - is_calibrated: {follower_mock.is_calibrated}")
        print(f"  - action_features: {follower_mock.action_features}")
        print(f"  - observation_features: {follower_mock.observation_features}")
        
        # Test SO leader mock methods
        leader_config = MockSOLeaderClientConfig()
        leader_mock = MockSOLeaderClient(leader_config)
        
        # Test properties
        print(f"  - is_connected: {leader_mock.is_connected}")
        print(f"  - is_calibrated: {leader_mock.is_calibrated}")
        print(f"  - action_features: {leader_mock.action_features}")
        print(f"  - feedback_features: {leader_mock.feedback_features}")
        
        print("✓ Basic mock methods work correctly")
        return True
        
    except Exception as e:
        print(f"✗ Failed to test basic mock methods: {e}")
        return False

def main():
    """Run all tests."""
    print("Running mock integration tests...\n")
    
    success = True
    success &= test_mock_imports()
    success &= test_mock_instantiation()
    success &= test_mock_basic_methods()
    
    print("\n" + "="*50)
    if success:
        print("✓ All mock integration tests PASSED")
        print("The mock implementations are correctly integrated and functional.")
    else:
        print("✗ Some mock integration tests FAILED")
        print("There may be issues with the mock implementations.")
    
    return success

if __name__ == "__main__":
    main()