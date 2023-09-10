from utils.stream import StreamTokenIterator


def test_next():
    stream = "abc123"
    stream_iter = StreamTokenIterator(stream)
    node = stream_iter.next()
    assert node.text == 'a'

def test_next_2_times():
    stream = "abc123"
    stream_iter = StreamTokenIterator(stream)
    stream_iter.next()
    node = stream_iter.next()
    assert node.text == 'b'

def test_next4():
    stream = "abc123"
    stream_iter = StreamTokenIterator(stream)
    node = stream_iter.next(4)
    assert node.text == '1'

def test_next1_prev1():
    stream = "abc123"
    stream_iter = StreamTokenIterator(stream)
    stream_iter.next(1)
    node = stream_iter.prev(1)
    assert node.text == 'a'

def test_prev():
    stream = "abc123"
    stream_iter = StreamTokenIterator(stream)
    node = stream_iter.prev()
    assert node.text is None

def test_next1():
    stream = "abc123"
    stream_iter = StreamTokenIterator(stream)
    node = stream_iter.next(1)
    assert node.text == 'a'

def test_next2():
    stream = "abc123"
    stream_iter = StreamTokenIterator(stream)
    node = stream_iter.next(2)
    assert node.text == 'b'

def test_next6():
    stream = "abc123"
    stream_iter = StreamTokenIterator(stream)
    node = stream_iter.next(6)
    assert node.text == '3'

def test_next7():
    stream = "abc123"
    stream_iter = StreamTokenIterator(stream)
    node = stream_iter.next(7)
    assert node.text is None


def test_next7_prev6():
    stream = "abc123"
    stream_iter = StreamTokenIterator(stream)
    stream_iter.next(7)
    node = stream_iter.prev(6)
    assert node.text == 'a'