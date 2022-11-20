from utils.template_utils import TemplateText


def test_keys():
    temp = TemplateText('from @tb1 in @table')
    keys = temp.get_keys()
    assert keys == ['tb1', 'table']


def test_string():
    temp = TemplateText('from @tb1 in @table')
    temp.set_value('tb1', 'tb1')
    temp.set_value('table', 'customer')
    assert temp.to_string() == 'from tb1 in customer'


if __name__ == "__main__":
    test_string()
