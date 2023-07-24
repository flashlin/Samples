import ast

data_str = '''[
    ("select id from cust",
    {
        'type': 'select',
        'columns': [(0, 1)],
        'froms': [3]
    }),
    ("select id , name from cust",
    {
        'type': 'select',
        'columns': [(0, 1), (0, 3)],
        'froms': [5]
    }),
    ("select id , name from ( select id, name from cust )",
    {
         'type': 'select',
         'columns': [(0, 1), (0, 3)],
         'froms': [{
            'type': 'select',
            'columns': [(0, 7), (0, 9)],
            'froms': [11]
         }]
    }),
]'''

# 將 data_str 反序列化為 Python 物件
data = ast.literal_eval(data_str)

print(data)
