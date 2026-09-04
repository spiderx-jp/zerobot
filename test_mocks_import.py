#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test script to verify that mock files can be imported correctly."""

import sys
import traceback

def test_imports():
    """Test that all mock modules can be imported."""
    mock_modules = [
        "tests.mocks.mock_robot",
        "tests.mocks.mock_feetech",
        "tests.mocks.mock_dynamixel",
        "tests.mocks.mock_teleop",
        "tests.mocks.mock_serial_patch",
        "tests.mocks.mock_motors_bus",
    ]
    
    print("Testing imports of mock modules...")
    success_count = 0
    
    for module in mock_modules:
        try:
            __import__(module)
            print(f"✓ {module} imported successfully")
            success_count += 1
        except Exception as e:
            print(f"✗ Failed to import {module}: {e}")
            traceback.print_exc()
    
    print(f"\nImport test completed: {success_count}/{len(mock_modules)} modules imported successfully")
    return success_count == len(mock_modules)

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)