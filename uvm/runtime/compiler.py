"""USL Compiler - Compiles AST to UVM Bytecode.

Transforms the high-level AST into low-level stack-based instructions.
"""

from typing import List

from .bytecode import Op, Instruction
from .parser import (
    ASTNode, Assign, Call, BinaryOp, Variable, Literal,
    DoWhile, If, Break, Return, Input, Exec, AgentCreate, Block
)


class Compiler:
    """Compile AST to bytecode."""
    
    def __init__(self):
        self.code: List[Instruction] = []
        self.constants: List = []
        self.variables: set = set()
        self.break_stack: List[List[int]] = []  # Pending BREAK patches per loop level
    
    def compile(self, node: ASTNode) -> List[Instruction]:
        """Compile an AST node to bytecode."""
        self.code = []
        self.constants = []
        self.variables = set()
        self.break_stack = []
        
        self._compile_node(node)
        
        if self.break_stack:
            raise SyntaxError("BREAK without enclosing loop")
        
        # Add implicit return if not present
        if not self.code or self.code[-1].op != Op.RETURN:
            self._emit(Op.LOAD_CONST, None)
            self._emit(Op.RETURN)
        
        return self.code
    
    def _emit(self, op: Op, operand=None):
        """Emit an instruction."""
        self.code.append(Instruction(op, operand))
    
    def _add_const(self, value) -> int:
        """Add a constant and return its index."""
        # Try to find existing
        try:
            return self.constants.index(value)
        except ValueError:
            idx = len(self.constants)
            self.constants.append(value)
            return idx
    
    def _compile_node(self, node: ASTNode):
        """Dispatch to appropriate compile method."""
        # TODO: convert this into a dictionary version for better extensibility.
        if isinstance(node, Block):
            self._compile_block(node)
        elif isinstance(node, Assign):
            self._compile_assign(node)
        elif isinstance(node, Call):
            self._compile_call(node)
        elif isinstance(node, BinaryOp):
            self._compile_binary_op(node)
        elif isinstance(node, Variable):
            self._compile_variable(node)
        elif isinstance(node, Literal):
            self._compile_literal(node)
        elif isinstance(node, DoWhile):
            self._compile_do_while(node)
        elif isinstance(node, If):
            self._compile_if(node)
        elif isinstance(node, Break):
            self._compile_break(node)
        elif isinstance(node, Return):
            self._compile_return(node)
        elif isinstance(node, Input):
            self._compile_input(node)
        elif isinstance(node, Exec):
            self._compile_exec(node)
        elif isinstance(node, AgentCreate):
            self._compile_agent_create(node)
        else:
            raise ValueError(f"Unknown AST node type: {type(node)}")
    
    def _compile_block(self, node: Block):
        """Compile a block of statements."""
        for stmt in node.statements:
            self._compile_node(stmt)
    
    def _compile_assign(self, node: Assign):
        """Compile assignment: name = value"""
        # Compile value (leaves result on stack)
        self._compile_node(node.value)
        
        # Store to variable
        self._emit(Op.STORE_VAR, node.name)
        self.variables.add(node.name)
    
    def _compile_call(self, node: Call):
        """Compile function call."""
        # Compile callee
        if isinstance(node.callee, Literal):
            self._emit(Op.LOAD_CONST, node.callee.value)
        elif isinstance(node.callee, Variable):
            self._emit(Op.LOAD_VAR, node.callee.name)
        else:
            self._compile_node(node.callee)
        
        # Compile arguments (each leaves value on stack)
        for arg in node.args:
            self._compile_node(arg)
        
        # Emit call with arg count
        self._emit(Op.YIELD_CALL, len(node.args))
    
    def _compile_binary_op(self, node: BinaryOp):
        """Compile binary operation."""
        # Compile operands
        self._compile_node(node.left)
        self._compile_node(node.right)
        
        # Emit operator
        op_map = {
            '+': Op.ADD,
            '==': Op.EQ,
            '<': Op.LT,
            '>': Op.GT,
        }
        if node.op not in op_map:
            raise ValueError(f"Unknown operator: {node.op}")
        self._emit(op_map[node.op])
    
    def _compile_variable(self, node: Variable):
        """Compile variable reference."""
        self._emit(Op.LOAD_VAR, node.name)
    
    def _compile_literal(self, node: Literal):
        """Compile literal value.
        
        For simple values, inline them directly.
        For complex values, use the constants table.
        """
        # Inline simple values directly
        self._emit(Op.LOAD_CONST, node.value)
    
    def _compile_do_while(self, node: DoWhile):
        """Compile DO-WHILE loop to plain jumps with backpatching.
        
        Structure:
            start:
            [body]
            [condition]
            JUMP_IF_TRUE start
            end:
        """
        start_pos = len(self.code)
        self.break_stack.append([])  # Collect BREAKs for this loop
        
        # Compile body
        for stmt in node.body:
            self._compile_node(stmt)
        
        breaks = self.break_stack.pop()
        
        # Compile condition
        self._compile_node(node.condition)
        
        # Loop back if true, else fall through
        self._emit(Op.JUMP_IF_TRUE, start_pos)
        end_pos = len(self.code)
        
        # Patch BREAKs to jump past the loop
        for break_idx in breaks:
            self.code[break_idx] = Instruction(Op.JUMP, end_pos)
    
    def _compile_if(self, node: If):
        """Compile IF statement.
        
        Structure:
            [condition]
            JUMP_IF_FALSE else_pos
            [then body]
            JUMP end_pos
            else_pos:
            [else body]
            end_pos:
        """
        # Condition
        self._compile_node(node.condition)
        
        # Jump to else if false
        jump_else_idx = len(self.code)
        self._emit(Op.JUMP_IF_FALSE, None)  # Will patch
        
        # Then body
        for stmt in node.then_body:
            self._compile_node(stmt)
        
        # Jump to end
        jump_end_idx = len(self.code)
        self._emit(Op.JUMP, None)  # Will patch
        
        # Else position
        else_pos = len(self.code)
        self.code[jump_else_idx] = Instruction(Op.JUMP_IF_FALSE, else_pos)
        
        # Else body
        if node.else_body:
            for stmt in node.else_body:
                self._compile_node(stmt)
        
        # End position
        end_pos = len(self.code)
        self.code[jump_end_idx] = Instruction(Op.JUMP, end_pos)
    
    def _compile_break(self, node: Break):
        """Compile BREAK statement using backpatching."""
        if not self.break_stack:
            raise SyntaxError("BREAK outside of loop")
        self._emit(Op.JUMP, None)  # Placeholder
        self.break_stack[-1].append(len(self.code) - 1)
    
    def _compile_return(self, node: Return):
        """Compile RETURN statement."""
        if node.value:
            self._compile_node(node.value)
        else:
            self._emit(Op.LOAD_CONST, None)
        self._emit(Op.RETURN)
    
    def _compile_input(self, node: Input):
        """Compile INPUT() call."""
        self._emit(Op.INPUT)
    
    def _compile_exec(self, node: Exec):
        """Compile EXEC(plan) or EXEC(plan, initial_vars).
        
        Structure:
            [plan]
            [initial_vars or None]
            EXEC
        """
        # Compile plan expression
        self._compile_node(node.plan)
        
        # Compile initial_vars if present, else push None
        if node.initial_vars:
            self._compile_node(node.initial_vars)
        else:
            self._emit(Op.LOAD_CONST, None)
        
        self._emit(Op.EXEC)
    
    def _compile_agent_create(self, node: AgentCreate):
        """Compile AGENT creation.
        
        Structure:
            LOAD_CONST name
            LOAD_CONST lm
            LOAD_CONST prompt
            AGENT_CREATE
        """
        self._emit(Op.LOAD_CONST, node.name)
        self._emit(Op.LOAD_CONST, node.lm)
        self._emit(Op.LOAD_CONST, node.prompt)
        self._emit(Op.AGENT_CREATE)


def compile_ast(node: ASTNode) -> List[Instruction]:
    """Compile an AST node to bytecode."""
    return Compiler().compile(node)
