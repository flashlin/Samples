from utils.stream import StreamTokenIterator, read_identifier_token, reduce_token_list, Token


class TemplateText:
    def __init__(self, text: str):
        self.text = text
        self.dict = {}
        self.write_fn_list = []
        self._parse(text)

    def get_keys(self):
        return list(self.dict.keys())

    def set_value(self, key, value):
        self.dict[key] = value

    def to_string(self):
        text = ""
        for fn in self.write_fn_list:
            text += str(fn())
        return text

    def _parse(self, text):
        read_fn_list = [
            self.read_variable,
            TemplateText.read_text,
        ]
        write_fn_list = []
        stream_iterator = StreamTokenIterator(text)
        while not stream_iterator.is_done():
            write_fn = TemplateText.get_write_fn(stream_iterator, read_fn_list)
            if write_fn is None:
                raise Exception(f"try to parse template context fail at {stream_iterator.idx=} "
                                f"'{stream_iterator.peek_str(10)}'")
            write_fn_list.append(write_fn)
        self.write_fn_list = write_fn_list

    @staticmethod
    def get_write_fn(stream_iterator: StreamTokenIterator, fn_list: list):
        for parse_fn in fn_list:
            fn = parse_fn(stream_iterator)
            if fn is not None:
                return fn
        return None

    def read_variable(self, stream_iterator: StreamTokenIterator):
        text = stream_iterator.peek_str(2)
        if text.startswith('@@'):
            text = stream_iterator.next(2)

            def constant():
                return text

            return constant

        text = stream_iterator.peek_str(1)
        if text != '@':
            return None

        stream_iterator.next()
        token = read_identifier_token(stream_iterator)
        self.dict[token.text] = ''

        def write_var():
            return self.dict[token.text]

        return write_var

    @staticmethod
    def read_text(stream_iterator: StreamTokenIterator):
        buff = []
        while not stream_iterator.is_done():
            token = stream_iterator.peek()
            if token.text == "@":
                break
            stream_iterator.next()
            buff.append(token)
        token = reduce_token_list(Token.String, buff)

        def constant():
            return token.text

        return constant
