"""Boundary USL testcases for UVM."""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uvm.runtime.parser import parse
from uvm.runtime.compiler import compile_ast
from uvm.runtime.uvm import UVM
from uvm.runtime.bytecode import Op, Instruction
from uvm.ISA import EXEC, AGENT, CALL

from tests.fixtures import (
    EchoLM, CounterLM, JudgeLM, FixedLM, create_test_context
)


class TestParserBoundary:
    """Boundary tests for the USL parser."""

    @staticmethod
    def _compile_usl(usl: str):
        ast = parse(usl)
        return compile_ast(ast)

    def test_empty_source(self):
        """Empty source should compile to implicit return."""
        bytecode = self._compile_usl("")
        assert len(bytecode) == 2  # LOAD_CONST None, RETURN
        assert bytecode[-1].op == Op.RETURN

    def test_whitespace_only(self):
        """Whitespace-only source should compile to implicit return."""
        bytecode = self._compile_usl("   \n\t  \n  ")
        assert len(bytecode) == 2
        assert bytecode[-1].op == Op.RETURN

    def test_comment_only(self):
        """Comment-only source should compile to implicit return."""
        bytecode = self._compile_usl("# this is a comment\n")
        assert len(bytecode) == 2
        assert bytecode[-1].op == Op.RETURN

    def test_unterminated_string(self):
        """Unterminated string should raise SyntaxError."""
        try:
            self._compile_usl('x = "hello')
            assert False, "Should have raised SyntaxError"
        except SyntaxError:
            pass

    def test_break_outside_loop(self):
        """BREAK outside loop should raise SyntaxError at compile time."""
        try:
            self._compile_usl("BREAK")
            assert False, "Should have raised SyntaxError"
        except SyntaxError:
            pass

    def test_keyword_as_identifier_fails(self):
        """Keywords should not be usable as identifiers."""
        try:
            self._compile_usl("AGENT = 1")
            assert False, "Should have raised SyntaxError"
        except SyntaxError:
            pass

    def test_partial_keyword_as_identifier(self):
        """Partial keyword matches should be valid identifiers."""
        bytecode = self._compile_usl("breakVar = 5")
        # Should compile successfully
        assert any(inst.op == Op.STORE_VAR and inst.operand == "breakVar" for inst in bytecode)

    def test_eq_vs_assign_tokenization(self):
        """== should be parsed as EQ operator, = as assignment."""
        bytecode = self._compile_usl("x = 1 == 2")
        ops = [inst.op for inst in bytecode]
        assert Op.STORE_VAR in ops  # x = ...
        assert Op.EQ in ops         # 1 == 2

    def test_lt_gt_operators(self):
        """< and > should be parsed correctly."""
        bytecode = self._compile_usl("x = 1 < 2\ny = 3 > 2")
        ops = [inst.op for inst in bytecode]
        assert Op.LT in ops
        assert Op.GT in ops

    def test_invalid_character(self):
        """Invalid characters should raise SyntaxError."""
        try:
            self._compile_usl("x @ 1")
            assert False, "Should have raised SyntaxError"
        except SyntaxError:
            pass

    def test_empty_string_literal(self):
        """Empty string literal should be valid."""
        bytecode = self._compile_usl('x = ""')
        assert any(inst.op == Op.LOAD_CONST and inst.operand == "" for inst in bytecode)

    def test_string_with_quotes(self):
        """String literals with escaped quotes should be handled."""
        bytecode = self._compile_usl(r'x = "hello\"world"')
        # The literal should contain the escaped quote
        consts = [inst.operand for inst in bytecode if inst.op == Op.LOAD_CONST]
        assert any('"' in str(c) for c in consts if isinstance(c, str))


class TestVMBoundary:
    """Boundary tests for the UVM execution engine."""

    @staticmethod
    def _run_usl(usl: str, ctx=None, initial_vars=None):
        if ctx is None:
            ctx = create_test_context()
        ast = parse(usl)
        bytecode = compile_ast(ast)
        vm = UVM(bytecode, ctx, initial_vars)
        return vm.run_to_completion()

    def test_empty_program(self):
        """Empty program should return None."""
        result = self._run_usl("")
        assert result is None

    def test_single_return(self):
        """Single RETURN statement should work."""
        result = self._run_usl('RETURN(42)')
        assert result == 42

    def test_return_no_value(self):
        """RETURN without value should return None."""
        result = self._run_usl('RETURN()')
        assert result is None

    def test_do_while_single_iteration(self):
        """Do-while with false condition should run body once."""
        result = self._run_usl('''
            i = 0
            DO(
                i = i + 1
            )WHILE(0 == 1)
            RETURN(i)
        ''')
        assert result == 1

    def test_do_while_many_iterations(self):
        """Do-while with many iterations should complete."""
        result = self._run_usl('''
            i = 0
            DO(
                i = i + 1
            )WHILE(i < 100)
            RETURN(i)
        ''')
        assert result == 100

    def test_nested_loops_three_levels(self):
        """Three-level nested loops should work correctly."""
        result = self._run_usl('''
            total = 0
            i = 0
            DO(
                j = 0
                DO(
                    k = 0
                    DO(
                        total = total + 1
                        k = k + 1
                    )WHILE(k < 2)
                    j = j + 1
                )WHILE(j < 3)
                i = i + 1
            )WHILE(i < 4)
            RETURN(total)
        ''')
        assert result == 4 * 3 * 2  # 24

    def test_break_inner_loop(self):
        """BREAK in inner loop should only break that loop."""
        result = self._run_usl('''
            total = 0
            i = 0
            DO(
                j = 0
                DO(
                    total = total + 1
                    BREAK
                    j = j + 1
                )WHILE(j < 10)
                i = i + 1
            )WHILE(i < 5)
            RETURN(total)
        ''')
        assert result == 5  # inner loop runs once per outer iteration

    def test_break_outer_loop(self):
        """BREAK in outer loop should break the outer loop."""
        result = self._run_usl('''
            i = 0
            DO(
                i = i + 1
                BREAK
            )WHILE(i < 100)
            RETURN(i)
        ''')
        assert result == 1

    def test_return_inside_loop(self):
        """RETURN inside loop should halt immediately."""
        result = self._run_usl('''
            i = 0
            DO(
                i = i + 1
                RETURN(i)
                i = i + 99
            )WHILE(i < 100)
            RETURN(i)
        ''')
        assert result == 1

    def test_if_true_branch(self):
        """IF with true condition should execute then branch."""
        result = self._run_usl('''
            x = 0
            IF(1 == 1, {
                x = 1
            })
            RETURN(x)
        ''')
        assert result == 1

    def test_if_false_branch(self):
        """IF with false condition should skip then branch."""
        result = self._run_usl('''
            x = 0
            IF(1 == 2, {
                x = 1
            })
            RETURN(x)
        ''')
        assert result == 0

    def test_if_else_branches(self):
        """IF with else should execute correct branch."""
        result = self._run_usl('''
            x = 0
            IF(1 == 2, {
                x = 1
            }, {
                x = 2
            })
            RETURN(x)
        ''')
        assert result == 2

    def test_variable_reassignment(self):
        """Variable should be reassignable many times."""
        result = self._run_usl('''
            x = 1
            x = 2
            x = 3
            x = x + 1
            RETURN(x)
        ''')
        assert result == 4

    def test_undefined_variable_error(self):
        """Reading undefined variable should raise NameError."""
        ctx = create_test_context()
        try:
            self._run_usl('RETURN(undefined_var)', ctx=ctx)
            assert False, "Should have raised NameError"
        except NameError:
            pass

    def test_string_concatenation(self):
        """String concatenation should work."""
        result = self._run_usl('''
            a = "hello"
            b = "world"
            c = a + b
            RETURN(c)
        ''')
        assert str(result) == "helloworld"

    def test_empty_string_concat(self):
        """Concatenating empty string should work."""
        result = self._run_usl('''
            a = ""
            b = "x"
            c = a + b
            RETURN(c)
        ''')
        assert str(result) == "x"

    def test_number_zero(self):
        """Zero should be handled correctly."""
        result = self._run_usl('RETURN(0)')
        assert result == 0

    def test_nested_exec(self):
        """Nested EXEC should execute inner code."""
        ctx = create_test_context()
        inner = "x = 42\nRETURN(x)"
        outer = f'''result = EXEC("{inner}")
RETURN(result)
'''
        result = self._run_usl(outer, ctx=ctx)
        assert result == 42

    def test_agent_create_via_usl(self):
        """AGENT creation and calling via USL should work."""
        ctx = create_test_context()
        ctx.register_lm("backend", EchoLM(prefix="AGENT"))
        usl = '''
            AGENT("test_agent", "backend", "test prompt")
            result = CALL("test_agent", "hello")
            RETURN(result)
        '''
        result = self._run_usl(usl, ctx=ctx)
        assert "AGENT" in str(result)
        assert "hello" in str(result)

    def test_call_lm_by_name(self):
        """CALL should invoke registered LM."""
        ctx = create_test_context()
        ctx.register_lm("echo", EchoLM(prefix="TEST"))
        result = self._run_usl('''
            result = CALL("echo", "hello")
            RETURN(result)
        ''', ctx=ctx)
        assert "[TEST] hello" in str(result)

    def test_call_agent(self):
        """CALL should invoke registered AGENT."""
        ctx = create_test_context()
        ctx.register_lm("backend", EchoLM())
        from uvm.core.agent import AGENT
        AGENT("helper", "backend", "you are a helper", context=ctx)
        result = self._run_usl('''
            result = CALL("helper", "task")
            RETURN(result)
        ''', ctx=ctx)
        assert "task" in str(result)

    def test_counter_lm_multiple_calls(self):
        """Multiple CALLs should increment counter."""
        ctx = create_test_context()
        counter = CounterLM()
        ctx.register_lm("counter", counter)
        result = self._run_usl('''
            a = CALL("counter", "first")
            b = CALL("counter", "second")
            c = CALL("counter", "third")
            RETURN(c)
        ''', ctx=ctx)
        assert "COUNT:3" in str(result)
        assert counter.count == 3

    def test_judge_lm_loop(self):
        """Judge LM should return expected verdicts."""
        ctx = create_test_context()
        ctx.register_lm("judge", JudgeLM(yes_keyword="done", no_keyword="step"))
        result = self._run_usl('''
            status = "step"
            verdict = CALL("judge", status)
            RETURN(verdict)
        ''', ctx=ctx)
        assert str(result) == "NO"

    def test_initial_vars(self):
        """VM should accept initial variables."""
        ctx = create_test_context()
        result = self._run_usl('RETURN(x + y)', ctx=ctx, initial_vars={"x": 10, "y": 20})
        assert result == 30

    def test_snapshot_restore(self):
        """Snapshot and restore should preserve execution state."""
        ctx = create_test_context()
        bytecode = compile_ast(parse('''
            i = 0
            DO(
                i = i + 1
            )WHILE(i < 5)
            RETURN(i)
        '''))
        vm = UVM(bytecode, ctx)
        # Run one step (but since there are no yields, this completes)
        # So let's manually set ip mid-program for the test
        vm.ip = 2  # Skip initial i = 0
        vm.globals["i"] = 2  # Pretend we've looped twice
        snap = vm.snapshot()
        
        vm2 = UVM.from_snapshot(snap, bytecode, ctx)
        result = vm2.run_to_completion()
        assert result == 5

    def test_boolean_comparison(self):
        """Boolean-like comparisons should work in conditions."""
        result = self._run_usl('''
            x = 0
            DO(
                x = x + 1
            )WHILE(1 == 0)
            RETURN(x)
        ''')
        assert result == 1  # Body runs once, condition false

    def test_complex_expression(self):
        """Complex nested expression should evaluate correctly."""
        result = self._run_usl('''
            a = 1
            b = 2
            c = 3
            result = a + b == c
            RETURN(result)
        ''')
        assert result == True

    def test_loop_counter_with_break(self):
        """Break after specific count should work."""
        result = self._run_usl('''
            count = 0
            DO(
                count = count + 1
                IF(count == 5, {
                    BREAK
                })
            )WHILE(count < 100)
            RETURN(count)
        ''')
        assert result == 5

    def test_sequential_loops_no_leak(self):
        """Sequential loops should not interfere with each other."""
        result = self._run_usl('''
            i = 0
            DO(
                i = i + 1
            )WHILE(i < 3)
            j = 0
            DO(
                j = j + 1
            )WHILE(j < 4)
            RETURN(i + j)
        ''')
        assert result == 7


if __name__ == "__main__":
    import traceback

    classes = [TestParserBoundary, TestVMBoundary]
    passed = 0
    failed = 0

    for cls in classes:
        instance = cls()
        for name in dir(instance):
            if name.startswith("test_"):
                try:
                    getattr(instance, name)()
                    print(f"  PASS  {cls.__name__}.{name}")
                    passed += 1
                except Exception as e:
                    print(f"  FAIL  {cls.__name__}.{name}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
