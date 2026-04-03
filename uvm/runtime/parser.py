"""USL Parser - Parses UVM Specific Language into AST.

USL is a lightweight DSL for agentic workflows.
Example:
    agent = CALL("judger", "decide...")
    step = CALL(agent, problem, step)
    DO(
        x = CALL("a", "b")
        y = x + "suffix"
    )WHILE(CALL("check", y) == "NO")
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Union
from enum import IntEnum, auto


class ASTNode:
    """Base class for AST nodes."""
    pass


@dataclass
class Assign(ASTNode):
    """Variable assignment: name = value"""
    name: str
    value: "ASTNode"


@dataclass
class Call(ASTNode):
    """Function call: CALL(name, *args)"""
    callee: "ASTNode"  # Can be string literal or variable
    args: List["ASTNode"]


@dataclass
class BinaryOp(ASTNode):
    """Binary operation: left op right"""
    left: "ASTNode"
    op: str  # '+', '==', '<', '>'
    right: "ASTNode"


@dataclass
class Variable(ASTNode):
    """Variable reference"""
    name: str


@dataclass
class Literal(ASTNode):
    """String or number literal"""
    value: Any


@dataclass
class DoWhile(ASTNode):
    """DO(body)WHILE(cond)"""
    body: List["ASTNode"]
    condition: "ASTNode"


@dataclass
class If(ASTNode):
    """IF(cond, then_body, else_body)"""
    condition: "ASTNode"
    then_body: List["ASTNode"]
    else_body: Optional[List["ASTNode"]] = None


@dataclass
class Break(ASTNode):
    """BREAK statement"""
    pass


@dataclass
class Return(ASTNode):
    """RETURN(value)"""
    value: Optional["ASTNode"] = None


@dataclass
class Input(ASTNode):
    """INPUT()"""
    pass


@dataclass
class Exec(ASTNode):
    """EXEC(plan) - Execute USL code"""
    plan: "ASTNode"  # Can be string literal or variable
    initial_vars: Optional["ASTNode"] = None  # Optional dict of initial variables


@dataclass
class AgentCreate(ASTNode):
    """AGENT(name, lm, prompt)"""
    name: str
    lm: str
    prompt: str


@dataclass
class Block(ASTNode):
    """Block of statements"""
    statements: List[ASTNode]


class Parser:
    """Parse USL source into AST."""
    
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.tokens = self._tokenize()
        self.current = 0
    
    def _tokenize(self) -> List[tuple]:
        """Simple tokenizer."""
        import re
        
        # Token patterns
        patterns = [
            ('NEWLINE', r'\n'),
            ('SKIP', r'[ \t]+'),
            ('COMMENT', r'#.*'),
            ('STRING', r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\''),
            ('NUMBER', r'\d+\.?\d*'),
            ('IDENT', r'[a-zA-Z_][a-zA-Z0-9_]*'),
            ('LPAREN', r'\('),
            ('RPAREN', r'\)'),
            ('LBRACE', r'\{'),
            ('RBRACE', r'\}'),
            ('ASSIGN', r'='),
            ('EQ', r'=='),
            ('PLUS', r'\+'),
            ('LT', r'<'),
            ('GT', r'>'),
            ('COMMA', r','),
            ('COLON', r':'),
        ]
        
        regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in patterns)
        
        tokens = []
        for match in re.finditer(regex, self.source):
            kind = match.lastgroup
            value = match.group()
            
            if kind == 'SKIP' or kind == 'COMMENT':
                continue
            elif kind == 'NEWLINE':
                continue
            elif kind == 'STRING':
                # Remove quotes
                value = value[1:-1]
                tokens.append((kind, value))
            elif kind == 'NUMBER':
                if '.' in value:
                    tokens.append((kind, float(value)))
                else:
                    tokens.append((kind, int(value)))
            else:
                tokens.append((kind, value))
        
        tokens.append(('EOF', None))
        return tokens
    
    def _peek(self, offset=0):
        """Peek at current token."""
        pos = self.current + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return ('EOF', None)
    
    def _advance(self):
        """Consume and return current token."""
        token = self._peek()
        self.current += 1
        return token
    
    def _expect(self, kind, value=None):
        """Expect and consume a specific token."""
        token = self._advance()
        if token[0] != kind:
            raise SyntaxError(f"Expected {kind}, got {token[0]} at position {self.current}")
        if value is not None and token[1] != value:
            raise SyntaxError(f"Expected '{value}', got '{token[1]}'")
        return token
    
    def _match(self, kind, value=None):
        """Check if current token matches (without consuming)."""
        token = self._peek()
        if token[0] != kind:
            return False
        if value is not None and token[1] != value:
            return False
        return True
    
    def _consume(self, kind, value=None):
        """Consume token if it matches."""
        if self._match(kind, value):
            return self._advance()
        return None
    
    def parse(self) -> Block:
        """Parse the full source."""
        statements = []
        while not self._match('EOF'):
            stmt = self._parse_statement()
            if stmt:
                statements.append(stmt)
        return Block(statements)
    
    def _parse_statement(self) -> Optional[ASTNode]:
        """Parse a single statement."""
        if self._match('EOF'):
            return None
        
        # Check for keywords
        if self._match('IDENT', 'AGENT'):
            return self._parse_agent_create()
        if self._match('IDENT', 'CALL'):
            return self._parse_call()
        if self._match('IDENT', 'DO'):
            return self._parse_do_while()
        if self._match('IDENT', 'IF'):
            return self._parse_if()
        if self._match('IDENT', 'BREAK'):
            self._advance()
            return Break()
        if self._match('IDENT', 'RETURN'):
            return self._parse_return()
        if self._match('IDENT', 'INPUT'):
            self._advance()
            self._expect('LPAREN')
            self._expect('RPAREN')
            return Input()
        if self._match('IDENT', 'EXEC'):
            return self._parse_exec()
        
        # Check for assignment: ident = expr
        if self._match('IDENT'):
            if self._peek(1)[0] == 'ASSIGN':
                return self._parse_assignment()
        
        # Otherwise, try to parse as expression
        return self._parse_expression()
    
    def _parse_agent_create(self) -> AgentCreate:
        """Parse AGENT(name, lm, prompt)."""
        self._expect('IDENT', 'AGENT')
        self._expect('LPAREN')
        name = self._expect('STRING')[1]
        self._expect('COMMA')
        lm = self._expect('STRING')[1]
        self._expect('COMMA')
        prompt = self._expect('STRING')[1]
        self._expect('RPAREN')
        return AgentCreate(name, lm, prompt)
    
    def _parse_call(self) -> Call:
        """Parse CALL(name, *args)."""
        self._expect('IDENT', 'CALL')
        self._expect('LPAREN')
        
        # Callee - can be string or variable
        if self._match('STRING'):
            callee = Literal(self._advance()[1])
        else:
            callee = self._parse_expression()
        
        # Arguments
        args = []
        while self._match('COMMA'):
            self._advance()
            args.append(self._parse_expression())
        
        self._expect('RPAREN')
        return Call(callee, args)
    
    def _parse_do_while(self) -> DoWhile:
        """Parse DO(body)WHILE(cond)."""
        self._expect('IDENT', 'DO')
        self._expect('LPAREN')
        
        # Parse body statements until )WHILE
        body = []
        while not (self._match('RPAREN') and self._peek(1)[1] == 'WHILE'):
            stmt = self._parse_statement()
            if stmt:
                body.append(stmt)
            if self._match('COMMA'):
                self._advance()  # Allow comma separation
        
        self._expect('RPAREN')
        self._expect('IDENT', 'WHILE')
        self._expect('LPAREN')
        condition = self._parse_expression()
        self._expect('RPAREN')
        
        return DoWhile(body, condition)
    
    def _parse_if(self) -> If:
        """Parse IF(cond, then, else?)."""
        self._expect('IDENT', 'IF')
        self._expect('LPAREN')
        condition = self._parse_expression()
        self._expect('COMMA')
        
        # Then body - can be single statement or block
        then_body = []
        if self._match('LBRACE'):
            self._advance()
            while not self._match('RBRACE'):
                stmt = self._parse_statement()
                if stmt:
                    then_body.append(stmt)
            self._expect('RBRACE')
        else:
            then_body.append(self._parse_statement())
        
        # Optional else
        else_body = None
        if self._match('COMMA'):
            self._advance()
            else_body = []
            if self._match('LBRACE'):
                self._advance()
                while not self._match('RBRACE'):
                    stmt = self._parse_statement()
                    if stmt:
                        else_body.append(stmt)
                self._expect('RBRACE')
            else:
                else_body.append(self._parse_statement())
        
        self._expect('RPAREN')
        return If(condition, then_body, else_body)
    
    def _parse_return(self) -> Return:
        """Parse RETURN(value?)."""
        self._expect('IDENT', 'RETURN')
        self._expect('LPAREN')
        
        value = None
        if not self._match('RPAREN'):
            value = self._parse_expression()
        
        self._expect('RPAREN')
        return Return(value)
    
    def _parse_exec(self) -> Exec:
        """Parse EXEC(plan) or EXEC(plan, initial_vars)."""
        self._expect('IDENT', 'EXEC')
        self._expect('LPAREN')
        
        # Plan - can be string literal or variable/expression
        plan = self._parse_expression()
        
        # Optional initial_vars
        initial_vars = None
        if self._match('COMMA'):
            self._advance()
            initial_vars = self._parse_expression()
        
        self._expect('RPAREN')
        return Exec(plan, initial_vars)
    
    def _parse_assignment(self) -> Assign:
        """Parse name = value."""
        name = self._expect('IDENT')[1]
        self._expect('ASSIGN')
        value = self._parse_expression()
        return Assign(name, value)
    
    def _parse_expression(self) -> ASTNode:
        """Parse expression with operators."""
        return self._parse_equality()
    
    def _parse_equality(self) -> ASTNode:
        """Parse equality: term (== term)*"""
        left = self._parse_additive()
        
        while self._match('EQ'):
            op = self._advance()[1]
            right = self._parse_additive()
            left = BinaryOp(left, op, right)
        
        return left
    
    def _parse_additive(self) -> ASTNode:
        """Parse addition: primary (+ primary)*"""
        left = self._parse_primary()
        
        while self._match('PLUS'):
            op = self._advance()[1]
            right = self._parse_primary()
            left = BinaryOp(left, op, right)
        
        return left
    
    def _parse_primary(self) -> ASTNode:
        """Parse primary expression."""
        if self._match('STRING'):
            return Literal(self._advance()[1])
        
        if self._match('NUMBER'):
            return Literal(self._advance()[1])
        
        if self._match('IDENT'):
            name = self._advance()[1]
            
            # Function calls
            if name == 'CALL' and self._match('LPAREN'):
                self.current -= 1  # Back up to parse as call
                return self._parse_call()
            
            if name == 'INPUT' and self._match('LPAREN'):
                self._advance()
                self._expect('RPAREN')
                return Input()
            
            if name == 'BREAK':
                return Break()
            
            if name == 'RETURN' and self._match('LPAREN'):
                self.current -= 1
                return self._parse_return()
            
            if name == 'EXEC' and self._match('LPAREN'):
                self.current -= 1
                return self._parse_exec()
            
            return Variable(name)
        
        if self._match('LPAREN'):
            self._advance()
            expr = self._parse_expression()
            self._expect('RPAREN')
            return expr
        
        raise SyntaxError(f"Unexpected token: {self._peek()}")


def parse(source: str) -> Block:
    """Parse USL source into AST."""
    return Parser(source).parse()
