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

import codecs
import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union
from enum import IntEnum, auto


# ---------------------------------------------------------------------------
# Trie singleton for keywords and operators
# ---------------------------------------------------------------------------

class _TrieNode:
    __slots__ = ('children', 'token_type')
    def __init__(self):
        self.children = {}
        self.token_type = None
    
    def qry(self, s: str, i: int, n: int):
        root, end, typ = self, i, None
        for j in range(i, n):
            if (u := s[j]) not in root.children:
                return end, typ
            root, end = root.children[u], end + 1
            if root.token_type is not None:
                typ = root.token_type
        return end, typ
    
    def register(self, s: str, k: str):
        root = self
        for u in s:
            if u not in root.children:
                root.children[u] = _TrieNode()
            root = root.children[u]
        root.token_type = k


def _create_trie():
    patterns = [
        # Keywords
        ('AGENT', 'AGENT'), ('CALL', 'CALL'), ('DO', 'DO'),
        ('WHILE', 'WHILE'), ('IF', 'IF'), ('BREAK', 'BREAK'),
        ('RETURN', 'RETURN'), ('INPUT', 'INPUT'), ('EXEC', 'EXEC'),
        # Operators
        ('==', 'EQ'), ('=', 'ASSIGN'), ('+', 'PLUS'),
        ('<', 'LT'), ('>', 'GT'),
        # Delimiters
        ('(', 'LPAREN'), (')', 'RPAREN'),
        ('{', 'LBRACE'), ('}', 'RBRACE'),
        (',', 'COMMA'), (':', 'COLON'),
    ]
    trie = _TrieNode()
    for text, kind in patterns:
        trie.register(text, kind)
    return trie


# Global singleton — language-level, not instance-level
_KEYWORD_TRIE = _create_trie()


# ---------------------------------------------------------------------------
# Matcher callables  (source, pos) -> (token_or_None, next_pos)
# ---------------------------------------------------------------------------

# -- regex patterns for String and Number matchers (full Python style) --
_STRING_RE = re.compile(r'''
    "(?:[^"\\]|\\.)*"      # double-quoted string
|   '(?:[^'\\]|\\.)*'       # single-quoted string
''', re.VERBOSE)

_NUMBER_RE = re.compile(r'''
    (?:(?:\d+(_\d+)*)?\.(?:\d+(_\d+)*)?(?:[eE][+-]?\d+)?   # .5  1.  1.5  1.5e3
    |   \d+(_\d+)*(?:[eE][+-]?\d+)?                         # 1  1e3  1_000
    |   0[bB](?:_?[01])+                                    # 0b101
    |   0[oO](?:_?[0-7])+                                   # 0o777
    |   0[xX](?:_?[0-9a-fA-F])+                             # 0xFF
    )
''', re.VERBOSE)

_IDENT_RE = re.compile(r'[a-zA-Z_]\w*')


def whitespace_matcher(s: str, i: int) -> Tuple[Optional[tuple], int]:
    """Skip whitespace. Returns (None, next_pos) if matched."""
    n = len(s)
    j = i
    while j < n and s[j].isspace():
        j += 1
    return (None, j) if j > i else (None, i)


def comment_matcher(s: str, i: int) -> Tuple[Optional[tuple], int]:
    """Skip # comment to end of line. Returns (None, next_pos) if matched."""
    if i < len(s) and s[i] == '#':
        j = i
        n = len(s)
        while j < n and s[j] not in '\r\n':
            j += 1
        return (None, j)
    return (None, i)


def string_matcher(s: str, i: int) -> Tuple[Optional[tuple], int]:
    """Match Python-style string literal with full escape support."""
    m = _STRING_RE.match(s, i)
    if not m:
        return (None, i)
    raw = m.group()
    quote = raw[0]
    if raw[-1] != quote:
        raise SyntaxError(f"Unterminated string starting at position {i}")
    # Strip quotes and decode Python escapes (\n, \t, \xhh, \uXXXX, etc.)
    value = codecs.decode(raw[1:-1], 'unicode_escape')
    return (('STRING', value), m.end())


def number_matcher(s: str, i: int) -> Tuple[Optional[tuple], int]:
    """Match Python-style numeric literal (decimal, hex, octal, binary,
    float, scientific notation, with underscores)."""
    m = _NUMBER_RE.match(s, i)
    if not m:
        return (None, i)
    text = m.group().replace('_', '')
    if any(c in text for c in '.eE'):
        value = float(text)
    else:
        value = int(text, 0)
    return (('NUMBER', value), m.end())


def trie_matcher(s: str, i: int) -> Tuple[Optional[tuple], int]:
    """Longest-match keyword/operator via global Trie."""
    end, typ = _KEYWORD_TRIE.qry(s, i, len(s))
    if typ is None:
        return (None, i)
    return ((typ, s[i:end]), end)


def identifier_matcher(s: str, i: int) -> Tuple[Optional[tuple], int]:
    """Match identifier [a-zA-Z_][a-zA-Z0-9_]*."""
    m = _IDENT_RE.match(s, i)
    if not m:
        return (None, i)
    return (('IDENT', m.group()), m.end())


# Default matcher pipeline — order matters for ties
default_matchers = [
    whitespace_matcher,
    comment_matcher,
    string_matcher,
    trie_matcher,
    number_matcher,
    identifier_matcher,
]


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

    def __init__(self, source: str, matchers: Optional[List[Callable]] = None):
        self.source = source
        self.pos = 0
        self.matchers = matchers if matchers is not None else default_matchers
        self.tokens = self._tokenize()
        self.current = 0
    
    def _tokenize(self) -> List[tuple]:
        """Matcher-based tokenizer with longest-match arbitration.
        
        Each matcher in self.matchers returns (token, next_pos).
        token = None means "consumed but don't emit" (whitespace, comments).
        The candidate that advances the farthest wins. Ties are resolved
        by matcher order (earlier in the list wins).
        """
        tokens = []
        s = self.source
        n = len(s)
        i = 0
        
        while i < n:
            best_pos = i
            best_token = None
            
            for matcher in self.matchers:
                token, next_pos = matcher(s, i)
                if next_pos > best_pos:
                    best_pos = next_pos
                    best_token = token
                elif next_pos == best_pos and best_pos > i and token is not None:
                    # Tie on length — prefer the earlier matcher by keeping current best
                    pass
            
            if best_pos == i:
                raise SyntaxError(f"Unexpected character '{s[i]}' at position {i}")
            
            if best_token is not None:
                tokens.append(best_token)
            i = best_pos
        
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
        if self._match('AGENT'):
            return self._parse_agent_create()
        if self._match('CALL'):
            return self._parse_call()
        if self._match('DO'):
            return self._parse_do_while()
        if self._match('IF'):
            return self._parse_if()
        if self._match('BREAK'):
            self._advance()
            return Break()
        if self._match('RETURN'):
            return self._parse_return()
        if self._match('INPUT'):
            self._advance()
            self._expect('LPAREN')
            self._expect('RPAREN')
            return Input()
        if self._match('EXEC'):
            return self._parse_exec()
        
        # Check for assignment: ident = expr
        if self._match('IDENT'):
            if self._peek(1)[0] == 'ASSIGN':
                return self._parse_assignment()
        
        # Otherwise, try to parse as expression
        return self._parse_expression()
    
    def _parse_agent_create(self) -> AgentCreate:
        """Parse AGENT(name, lm, prompt)."""
        self._expect('AGENT')
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
        self._expect('CALL')
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
        self._expect('DO')
        self._expect('LPAREN')
        
        # Parse body statements until )WHILE
        body = []
        while not (self._match('RPAREN') and self._peek(1)[0] == 'WHILE'):
            stmt = self._parse_statement()
            if stmt:
                body.append(stmt)
            if self._match('COMMA'):
                self._advance()  # Allow comma separation
        
        self._expect('RPAREN')
        self._expect('WHILE')
        self._expect('LPAREN')
        condition = self._parse_expression()
        self._expect('RPAREN')
        
        return DoWhile(body, condition)
    
    def _parse_if(self) -> If:
        """Parse IF(cond, then, else?)."""
        self._expect('IF')
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
        self._expect('RETURN')
        self._expect('LPAREN')
        
        value = None
        if not self._match('RPAREN'):
            value = self._parse_expression()
        
        self._expect('RPAREN')
        return Return(value)
    
    def _parse_exec(self) -> Exec:
        """Parse EXEC(plan) or EXEC(plan, initial_vars)."""
        self._expect('EXEC')
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
        """Parse equality/comparison: term (== | < | > term)*"""
        left = self._parse_additive()
        
        while self._match('EQ') or self._match('LT') or self._match('GT'):
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
        
        # Keyword expressions
        if self._match('CALL'):
            if self._peek(1)[0] == 'LPAREN':
                return self._parse_call()
            raise SyntaxError("Unexpected CALL without '('")
        
        if self._match('INPUT'):
            self._advance()
            self._expect('LPAREN')
            self._expect('RPAREN')
            return Input()
        
        if self._match('BREAK'):
            self._advance()
            return Break()
        
        if self._match('RETURN'):
            if self._peek(1)[0] == 'LPAREN':
                return self._parse_return()
            raise SyntaxError("Unexpected RETURN without '('")
        
        if self._match('EXEC'):
            if self._peek(1)[0] == 'LPAREN':
                return self._parse_exec()
            raise SyntaxError("Unexpected EXEC without '('")
        
        if self._match('IDENT'):
            name = self._advance()[1]
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
