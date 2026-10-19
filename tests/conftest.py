from unittest.mock import patch

import pytest

import smolagents.persistent_memory
from smolagents.agents import MultiStepAgent
from smolagents.monitoring import LogLevel
from smolagents.persistent_memory import PersistentMemory


# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.agents", "tests.fixtures.tools"]

original_multi_step_agent_init = MultiStepAgent.__init__


@pytest.fixture(autouse=True)
def patch_multi_step_agent_with_suppressed_logging():
    with patch.object(MultiStepAgent, "__init__", autospec=True) as mock_init:

        def init_with_suppressed_logging(self, *args, verbosity_level=LogLevel.OFF, **kwargs):
            original_multi_step_agent_init(self, *args, verbosity_level=verbosity_level, **kwargs)

        mock_init.side_effect = init_with_suppressed_logging
        yield


@pytest.fixture(autouse=True)
def isolate_persistent_memory(tmp_path):
    smolagents.persistent_memory._default_store = PersistentMemory(str(tmp_path / "memory.json"))
    yield
    smolagents.persistent_memory._default_store = None
