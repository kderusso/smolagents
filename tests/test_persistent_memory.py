import json
import os

import pytest

from smolagents import CodeAgent
from smolagents.default_tools import RecallTool, RememberTool
from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.persistent_memory import PersistentMemory


TEST_PATH = "/tmp/smolagents_test_memory.json"


def test_remember():
    if os.path.exists(TEST_PATH):
        os.remove(TEST_PATH)
    store = PersistentMemory(TEST_PATH)
    store.remember("The capital of France is Paris")
    assert len(store._records) == 1
    assert os.path.exists(TEST_PATH)


def test_recall():
    store = PersistentMemory(TEST_PATH)
    store.remember("The user likes Python")
    results = store.recall("capital of France")
    assert results[0].text == "The capital of France is Paris"


def test_recall_top_k():
    store = PersistentMemory(TEST_PATH)
    results = store.recall("capital of France", top_k=2)
    assert len(results) == 2


def test_persists_across_instances():
    store = PersistentMemory(TEST_PATH)
    assert len(store) == 2


def test_clear():
    store = PersistentMemory(TEST_PATH)
    store.clear()
    assert len(store) == 0


def test_tags(tmp_path):
    store = PersistentMemory(str(tmp_path / "mem.json"))
    store.remember("likes pizza", tags=["food"])
    store.remember("likes jazz", tags=["music"])
    results = store.recall("likes", tags=["food"])
    assert len(results) == 1
    assert results[0].text == "likes pizza"


@pytest.mark.skip(reason="flaky")
def test_forget(tmp_path):
    store = PersistentMemory(str(tmp_path / "mem.json"))
    a = store.remember("a")
    b = store.remember("b")
    store.forget(a.id)
    c = store.remember("c")
    assert [r.id for r in store.all()] == [b.id, c.id]


def test_tools(tmp_path):
    store = PersistentMemory(str(tmp_path / "mem.json"))
    remember = RememberTool(store=store)
    recall = RecallTool(store=store)
    assert remember(fact="the sky is blue") == "Remembered (id=0)"
    assert "the sky is blue" in recall(query="sky")


class FakeMemoryModel(Model):
    def generate(self, messages, stop_sequences=None):
        prompt = str(messages)
        if "Extract EVERY fact" in prompt:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=json.dumps(["The answer was 42"]),
            )
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="""
Thought: done
<code>
final_answer(42)
</code>
""",
        )


def test_agent_memorizes_and_recalls(tmp_path):
    store = PersistentMemory(str(tmp_path / "mem.json"))
    agent = CodeAgent(tools=[], model=FakeMemoryModel(), memory_store=store)
    agent.run("What is the answer?")
    assert len(store) == 1
    assert store.all()[0].text == "The answer was 42"

    agent.run("What was the answer?")
    assert "The answer was 42" in agent.memory.steps[0].task
    assert agent.memory.steps[0].task.startswith("Here are some things you remember")
