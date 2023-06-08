import base64
import unittest


class TestBase64(unittest.TestCase):
    def test_encode(self):
        input_text = "abc"
        binary_data = input_text.encode("utf-8")
        base64_string = base64.b64encode(binary_data)
        decoded_data = base64.b64decode(base64_string)
        decoded_text = decoded_data.decode('utf-8')
        assert decoded_text == input_text


if __name__ == "__main__":
    unittest.main()
