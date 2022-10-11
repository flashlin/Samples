class Token(object):
    Name = 'name'
    String = 'string'
    Number = 'number'
    Operator = 'operator'
    Boolean = 'boolean'
    Undefined = 'undefined'
    Null = 'null'
    Regex = 'regex'
    EOF = '(end)'

    LITERALS = [String, Number, Boolean, Regex, Null, Undefined]

    def __init__(self, source, type, line=0, char=0):
        self.value = source
        self.type = type
        self.line = line
        self.char = char

    def __repr__(self):
        return '<%s: %s (line %d, char %d)>' % (self.type, self.value.replace('\n', '\\n'), self.line, self.char)

    def __str__(self):
        return self.value


class TokenContext:
    def __init__(self, stream):
        self.stream = stream
        self.length = len(stream)

    index = -1
    length = 0
    tokens = []

    def next(self):
        self.index += 1
        if self.index >= self.length:
            return None
        return self.stream[self.index]

    def peek(self):
        index = self.index + 1
        if index >= self.length:
            return None
        return self.stream[index]

    def move(self, pos):
        if self.index < 0:
            return
        if self.index >= self.length:
            return
        self.index += pos

    def is_end(self):
        return self.index >= self.length


def is_space_character(character):
    whitespace = [' ', '\n', '\t', '\r']
    return character in whitespace


def skip_spaces(context: TokenContext):
    character = context.next()
    while character is not None and is_space_character(character):
        character = context.next()
    context.move(-1)
    return context


def is_alpha(character):
    if character.isalpha():
        return True
    return character in ['_']


def is_number(character, first):
    return character.isnumeric() and first is not True


def operators_to_symbols(operators):
    dict = {}
    for word in operators:
        for ch in word:
            dict[ch] = 0
    return dict.keys()


def read_operators(context: TokenContext):
    operators = ['%=', '/=', '*=', '+=', '-=', '==', '!=', '++', '--',
                 '>=', '<=', '&=', '|=', '^=', '&&', '||',
                 '&', '|', '^', '~', '!', '{', '[', ']', '}', '(', ')',
                 '@', '*', '/', '%', '+', '-', '?', '<', '>',
                 ';', '=', ',', '.']
    symbols = operators_to_symbols(operators)
    character = context.next()
    token = ""
    while character is not None and (character in symbols):
        token += character
        character = context.next()
    context.move(-1)
    if token == "":
        return context
    if not (token in operators):
        for ch in token:
            context.tokens.append(ch)
        return context
    context.tokens.append(token)
    return context


def read_number(context: TokenContext):
    character = context.next()
    token = ""
    first = True
    while character is not None and (character.isnumeric() or (character == '.' and not first)):
        token += character
        character = context.next()
        first = False
    context.move(-1)

    if token == "":
        return context

    context.tokens.append(token)
    return context


def read_quote_string(context: TokenContext):
    quote = ["'", '"']
    character = context.next()

    token = ""
    prev_character = ""
    in_str = False
    while character is not None and ((character in quote and not in_str) or in_str):
        token += character
        prev_character = character
        if character in quote and in_str and prev_character != '\\':
            break
        character = context.next()
        in_str = True

    if token == "":
        return context

    if not (character in quote):
        raise Exception(f"read quote '{token}' {character} fail")

    context.tokens.append(token)
    return context


def is_identify(character, first):
    return character is not None and (is_alpha(character) or is_number(character, first))


def read_identify(context: TokenContext):
    character = context.next()
    token = ""
    first = True
    while is_identify(character, first):
        token += character
        character = context.next()
        first = False
    context.move(-1)

    if token == "":
        return context

    context.tokens.append(token)
    return context


def run_read_func(fn_list, context: TokenContext):
    start_index = context.index
    for fn in fn_list:
        rc = fn(context)
        if start_index != rc.index:
            return rc
    raise Exception(f"read token fail {context.index=} '{context.stream[context.index + 1:]}'")


def tokenize(stream):
    context = TokenContext(stream)
    fn_list = [skip_spaces, read_identify, read_operators, read_number, read_quote_string]
    while not context.is_end():
        run_read_func(fn_list, context)
    return context.tokens

