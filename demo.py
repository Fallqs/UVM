import uvm
from uvm.ISA import *


# uvm creates LM using config dictionary, which can either be given
# directly, read from json or read from yml.
m1 = uvm.LM(config="demo_config.yml", name='m1')

# you can directly call a uvm.LM even it doesn't have a name.
response = uvm.LM(config="demo_config.yml")("hello, let's start chat!")
print(response)

# in fact, uvm does not directly use Python str, it automatically transfers
# str to uvm.MemStr whenever recieves a Python str. This is for unified
# memory management from tokens to kv cache.
# fortunately, uvm.MemStr behaves almost the same as Python str.
s = uvm.MemStr("hello world")
print(s)      # >>> hello world
print(s + s)  # >>> hello worldhello world
print(type(s + 'hello world'), s + 'hello world')
              # >>> MemStr hello worldhello world

# uvm will not memorize any anonymous uvm.LM. to use LM later in
# a workflow, you have to give it a name. you can use uvm.register
# & uvm.delete to manually control LM registration.
m1 = uvm.LM(config="demo_config.yml", name='m1')  # m1 is automatically registered/updated
m2 = uvm.LM(config="demo_config.yml", name=None)  # m2 is anonymous
print(CALL("m1", "hello, let's start chat!"))  # returns text response

try:
    CALL("m2", "hello, let's start chat!")  # Throw NotFound Error
except KeyError as e:
    print(f"Expected error: {e}")

uvm.register('m2', m2)
print(CALL("m2", "hello, let's start chat!"))  # returns text response
uvm.delete('m1')

try:
    CALL("m1", "hello, let's start chat!")  # Throw NotFound Error
except KeyError as e:
    print(f"Expected error: {e}")

# Though, you can still call an anonymous/deleted LM directly.
print(m1("hello, let's start chat!"))  # returns text response

# uvm.ISA provides Procedure-Oriented instructions to use mutliple LMs.
# Among which, the CALL instruction directly invokes a LM.
uvm.LM(config="demo_config.yml", name='m1')
uvm.LM(config="demo_config.yml", name='m2')
uvm.LM(config="demo_config.yml", name='m3')

# Demonstrate AGENT creation with harness
print("\n--- Creating Agents ---")
AGENT('judger', 'm1', 'you are a judger')
AGENT('planner', 'm1', 'you are a planner')

print("Agent 'judger' created:", uvm.get_context().get_agent('judger'))
print("Agent 'planner' created:", uvm.get_context().get_agent('planner'))

# Demonstrate USL execution with EXEC
print("\n--- Executing USL ---")
usl_code = '''
step = "initial"
step = step + " modified"
RETURN(step)
'''
ret = EXEC(usl_code)
print(f"USL result: {ret}")

# Demonstrate USL with INPUT and CALL
print("\n--- USL with INPUT (commented out, requires user input) ---")
# usl_input = '''
# name = INPUT()
# greeting = "Hello " + name
# RETURN(greeting)
# '''
# result = EXEC(usl_input)
# print(f"Input result: {result}")
print("(Skipped - requires manual input)")

# Demonstrate context stats
print("\n--- Context Stats ---")
ctx = uvm.get_context()
print(f"Context stats: {ctx.stats()}")

# USL defines agentic instructions and support reflection, enabling AI-generated dynamic workflow.
# USL supports some Object-Oriented features.
# AGENT(...)  # <==> uvm.AGENT(name, lm, prompt)
# CALL(name, args)  # <==> uvm.LM(name=name).__call__(args)
# INPUT()  # <~~> Python's input()
# RETURN()  # <~~> Python's function return
# 
# uvm.AGENT & uvm.HARNESS have OO style design:
# class AGENT:
#     name: str
#     lm: LM
#     memory: dict[str, MemStr]
#     harness: HARNESS = DefaultHarness
#     def call(self, args):
#         inputs = self.harness.on_update(self, args)
#         ret = self.lm(*inputs)
#         return self.harness.on_return(self, ret)
# 
# class HARNESS:
#     # HARNESS is a memory control module, which updates memory on the fly.
#     on_update: Callable = lambda y, x: x
#     on_return: Callable = lambda y, x: x
