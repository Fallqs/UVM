"""UVM - The UVM Bytecode Virtual Machine.

Stack-based bytecode interpreter with yield/resume support for
memory management and checkpointing.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from contextlib import contextmanager

from .bytecode import Op, Instruction, VMState
from ..core.memstr import MemStr, to_memstr
from ..core.memory_engine import CallEvent


@dataclass
class Frame:
    """Call frame for nested execution."""
    return_ip: int
    locals: Dict[str, Any] = field(default_factory=dict)


class UVM:
    """UVM Bytecode Virtual Machine.
    
    A stack-based VM that executes USL bytecode with support for:
    - Yield/resume at external calls (for memory management)
    - Checkpoint/restore (for persistence)
    - Memory tracking integration
    
    Attributes:
        bytecode: List of instructions to execute
        context: UVMContext for registry access
        ip: Instruction pointer
        stack: Operand stack
        frames: Call frames (for future function support)
        globals: Global variable namespace
        state: Current VM execution state
    """
    
    def __init__(self, bytecode: List[Instruction], context: "UVMContext", 
                 initial_vars: Optional[Dict[str, Any]] = None):
        """Initialize UVM.
        
        Args:
            bytecode: Compiled bytecode instructions
            context: UVMContext for LM/Agent registry access
            initial_vars: Initial variable values
        """
        self.bytecode = bytecode
        self.context = context
        self.memory_engine = context.memory_engine if context else None
        
        self.ip = 0
        self.stack: List[Any] = []
        self.frames: List[Frame] = []
        self.globals: Dict[str, Any] = initial_vars.copy() if initial_vars else {}
        
        self.state = VMState.RUNNING
        self.yield_payload = None
        self.return_value = None
        
        # Track loop positions for BREAK
        self.loop_stack: List[int] = []
    
    def snapshot(self) -> dict:
        """Create serializable snapshot of VM state.
        
        Returns:
            Dict containing all state needed to resume execution.
        """
        return {
            "ip": self.ip,
            "stack": [(str(item) if isinstance(item, MemStr) else item) 
                      for item in self.stack],
            "globals": {k: (str(v) if isinstance(v, MemStr) else v) 
                       for k, v in self.globals.items()},
            "frames": [(f.return_ip, dict(f.locals)) for f in self.frames],
            "loop_stack": self.loop_stack.copy(),
        }
    
    @classmethod
    def from_snapshot(cls, snapshot: dict, bytecode: List[Instruction], 
                      context: "UVMContext") -> "UVM":
        """Restore UVM from snapshot.
        
        Args:
            snapshot: State dict from snapshot()
            bytecode: Bytecode to execute
            context: UVMContext
        
        Returns:
            Restored UVM instance.
        """
        vm = cls(bytecode, context)
        vm.ip = snapshot["ip"]
        vm.stack = [to_memstr(item, vm.memory_engine) for item in snapshot["stack"]]
        vm.globals = {k: to_memstr(v, vm.memory_engine) for k, v in snapshot["globals"].items()}
        vm.frames = [Frame(ip, locals_dict) for ip, locals_dict in snapshot["frames"]]
        vm.loop_stack = snapshot["loop_stack"]
        return vm
    
    def step(self) -> tuple:
        """Execute until next yield or completion.
        
        Returns:
            Tuple of (VMState, payload)
            - RUNNING: payload is None (continue executing)
            - YIELDED: payload is (callable, args) for external call
            - YIELD_CHECKPOINT: payload is snapshot dict
            - HALTED: payload is return value
            - ERROR: payload is exception
        """
        while self.ip < len(self.bytecode) and self.state == VMState.RUNNING:
            # Check memory engine for checkpoint request
            if self.memory_engine and self.memory_engine.checkpoint_request():
                self.state = VMState.YIELD_CHECKPOINT
                return VMState.YIELD_CHECKPOINT, self.snapshot()
            
            inst = self.bytecode[self.ip]
            self.ip += 1
            
            try:
                self._execute_instruction(inst)
            except Exception as e:
                self.state = VMState.ERROR
                return VMState.ERROR, e
        
        if self.state == VMState.HALTED:
            return VMState.HALTED, self.return_value
        
        return self.state, self.yield_payload
    
    def run_to_completion(self, max_steps: int = 100000) -> Any:
        """Run VM until completion (blocking).
        
        Args:
            max_steps: Maximum instructions to execute
        
        Returns:
            Final return value.
        """
        steps = 0
        while self.state not in (VMState.HALTED, VMState.ERROR) and steps < max_steps:
            state, payload = self.step()
            
            if state == VMState.YIELDED:
                # Execute the yielded call
                callable_info, args = payload
                result = self._execute_yielded_call(callable_info, args)
                self._resume_with(result)
            
            elif state == VMState.YIELD_CHECKPOINT:
                # Checkpoint requested, could serialize here
                # For now just continue
                self.state = VMState.RUNNING
            
            steps += 1
        
        if self.state == VMState.ERROR:
            raise payload
        
        return self.return_value
    
    def _execute_instruction(self, inst: Instruction):
        """Execute a single instruction."""
        op = inst.op
        operand = inst.operand
        
        # Stack operations
        if op == Op.LOAD_CONST:
            self.stack.append(operand)
        
        elif op == Op.LOAD_VAR:
            if operand not in self.globals:
                raise NameError(f"Variable '{operand}' not defined")
            val = self.globals[operand]
            # Retain when loading to stack
            if isinstance(val, MemStr):
                val._retain()
            self.stack.append(val)
        
        elif op == Op.STORE_VAR:
            if self.stack:
                val = self.stack.pop()
                # Release old value if exists
                if operand in self.globals:
                    old = self.globals[operand]
                    if isinstance(old, MemStr):
                        old._release()
                # Retain new value
                if isinstance(val, MemStr):
                    val._retain()
                self.globals[operand] = val
        
        elif op == Op.POP:
            if self.stack:
                val = self.stack.pop()
                if isinstance(val, MemStr):
                    val._release()
        
        elif op == Op.DUP:
            if self.stack:
                val = self.stack[-1]
                if isinstance(val, MemStr):
                    val._retain()
                self.stack.append(val)
        
        # Arithmetic
        elif op == Op.ADD:
            right = self.stack.pop()
            left = self.stack.pop()
            result = left + right
            if isinstance(result, str):
                result = to_memstr(result, self.memory_engine, "add")
            self.stack.append(result)
        
        elif op == Op.EQ:
            right = self.stack.pop()
            left = self.stack.pop()
            self.stack.append(left == right)
        
        elif op == Op.LT:
            right = self.stack.pop()
            left = self.stack.pop()
            self.stack.append(left < right)
        
        elif op == Op.GT:
            right = self.stack.pop()
            left = self.stack.pop()
            self.stack.append(left > right)
        
        # Control flow
        elif op == Op.JUMP:
            self.ip = operand
        
        elif op == Op.JUMP_IF_FALSE:
            cond = self.stack.pop()
            if not cond:
                self.ip = operand
        
        # UVM-specific
        elif op == Op.CALL:
            # Immediate call (non-yielding)
            num_args = operand
            args = [self.stack.pop() for _ in range(num_args)]
            args.reverse()
            callee = self.stack.pop()
            result = self._do_call(callee, args)
            self.stack.append(result)
        
        elif op == Op.YIELD_CALL:
            # Yield before calling
            num_args = operand
            args = [self.stack.pop() for _ in range(num_args)]
            args.reverse()
            callee = self.stack.pop()
            
            # Notify memory engine
            if self.memory_engine:
                event = CallEvent(
                    caller=None,
                    callee=str(callee),
                    args_summary=tuple(type(a).__name__ for a in args),
                    token_estimate=sum(len(str(a)) for a in args) // 4
                )
                self._call_handle = self.memory_engine.on_call_enter(event)
                self._call_event = event
            else:
                self._call_handle = None
                self._call_event = None
            
            self.state = VMState.YIELDED
            self.yield_payload = (callee, args)
        
        elif op == Op.INPUT:
            result = input()
            self.stack.append(to_memstr(result, self.memory_engine, "input"))
        
        elif op == Op.RETURN:
            if self.stack:
                self.return_value = self.stack.pop()
            else:
                self.return_value = None
            self.state = VMState.HALTED
        
        # Loop operations
        elif op == Op.DO_WHILE_START:
            self.loop_stack.append(self.ip - 1)  # Position after DO_WHILE_START
        
        elif op == Op.DO_WHILE_END:
            # Condition is on stack
            cond = self.stack.pop()
            if cond:
                # Jump back to loop start
                self.ip = self.loop_stack[-1]
            else:
                # Exit loop
                self.loop_stack.pop()
        
        elif op == Op.BREAK:
            # Jump past the DO_WHILE_END
            if self.loop_stack:
                # Find the DO_WHILE_END by scanning ahead
                depth = 1
                while self.ip < len(self.bytecode) and depth > 0:
                    if self.bytecode[self.ip].op == Op.DO_WHILE_START:
                        depth += 1
                    elif self.bytecode[self.ip].op == Op.DO_WHILE_END:
                        depth -= 1
                        if depth == 0:
                            self.ip += 1  # Skip the DO_WHILE_END
                            break
                    self.ip += 1
                self.loop_stack.pop()
        
        # Agent operations
        elif op == Op.AGENT_CREATE:
            prompt = self.stack.pop()
            lm = self.stack.pop()
            name = self.stack.pop()
            self._create_agent(name, lm, prompt)
        
        elif op == Op.NOP:
            pass
        
        else:
            raise ValueError(f"Unknown opcode: {op}")
    
    def _do_call(self, callee, args):
        """Execute a call immediately."""
        # Resolve callee
        if isinstance(callee, str):
            # Look up in context
            try:
                obj = self.context.get(callee)
            except KeyError:
                raise KeyError(f"'{callee}' not found in registry")
        else:
            obj = callee
        
        # Convert args to MemStr
        args = [to_memstr(a, self.memory_engine, "call_arg") for a in args]
        
        # Call
        if hasattr(obj, 'call'):
            result = obj.call(*args)
        elif callable(obj):
            result = obj(*args)
        else:
            raise TypeError(f"Cannot call {type(obj)}")
        
        # Ensure MemStr result
        if not isinstance(result, MemStr):
            result = to_memstr(result, self.memory_engine, "call_result")
        
        return result
    
    def _execute_yielded_call(self, callee, args):
        """Execute a call that was yielded."""
        result = self._do_call(callee, args)
        
        # Notify memory engine of completion
        if self.memory_engine and self._call_event:
            self.memory_engine.on_call_exit(self._call_event, self._call_handle, result)
        
        return result
    
    def _resume_with(self, value):
        """Resume execution with a value from a yielded call."""
        self.stack.append(value)
        self.state = VMState.RUNNING
        self.yield_payload = None
    
    def _create_agent(self, name, lm, prompt):
        """Create and register an agent."""
        from ..core.agent import AGENT
        
        # Get LM object if string
        if isinstance(lm, str):
            lm_obj = self.context.get_lm(lm)
        else:
            lm_obj = lm
        
        # Create agent
        agent = AGENT(
            name=str(name),
            lm=lm_obj,
            prompt=str(prompt),
            context=self.context,
            memory_engine=self.memory_engine
        )
        
        # Store reference in globals
        self.globals[str(name)] = agent
    
    def __repr__(self):
        return f"UVM(ip={self.ip}, stack_depth={len(self.stack)}, state={self.state.name})"
