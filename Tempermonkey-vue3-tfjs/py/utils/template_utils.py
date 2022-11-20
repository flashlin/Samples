from utils.stream import StreamIterator, EmptyToken, read_identifier, reduce_token_list, Token


class Template:
    def __init__(self, text: str):
        self.text = text
        self.dict = {}

    def _parse(self):
        read_fn_list = [
            self.read_variable,
            Template.read_text,
        ]
        write_fn_list = []
        stream_iterator = StreamIterator(self.text)
        while not stream_iterator.is_done():
            write_fn = Template.try_parse_any(stream_iterator, read_fn_list)
            if write_fn is None:
                raise Exception(f"try to parse template context fail at {stream_iterator.idx=} "
                                f"'{stream_iterator.peek_str(10)}'")
            write_fn_list.append(write_fn)
        self.write_fn_list = write_fn_list

    @staticmethod
    def try_parse_any(stream_iterator: StreamIterator, fn_list: list):
        for fn in fn_list:
            token = fn(stream_iterator)
            if token != EmptyToken:
                return token
        return EmptyToken

    def read_variable(self, stream_iterator: StreamIterator):
        text = stream_iterator.peek_str(2)
        if text.startswith('@@'):
            text = stream_iterator.next(2)
            self.dict[text] = ''

            def constant():
                return text

            return constant

        text = stream_iterator.peek_str(1)
        if text != '@':
            return None

        stream_iterator.next()
        token = read_identifier(stream_iterator)

        def write_var():
            return dict[token.text]

        return write_var

    @staticmethod
    def read_text(stream_iterator: StreamIterator):
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
