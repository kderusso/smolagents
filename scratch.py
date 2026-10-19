from smolagents import PersistentMemory

m = PersistentMemory("/tmp/test.json")
m.remember("hello world")
print(m.recall("hello"))
