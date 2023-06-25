import random
import string

def generate_id9() -> str:
    first_char = random.choice(string.ascii_uppercase)
    second_num = random.randint(1, 2)
    nums = ''.join(random.choice(string.digits) for _ in range(7))
    id9 = f"{first_char}{second_num}{nums}"
    return id9


def convert_id_to_numbers(id_str: str) -> [int]:
    first = {
        'A': 10,
        'B': 11,
        'C': 12,
        'D': 13,
        'E': 14,
        'F': 15,
        'G': 16,
        'H': 17,
        'I': 34,
        'J': 18,
        'K': 19,
        'L': 20,
        'M': 21,
        'N': 22,
        'O': 35,
        'P': 23,
        'Q': 24,
        'R': 25,
        'S': 26,
        'T': 27,
        'U': 28,
        'V': 29,
        'W': 32,
        'X': 30,
        'Y': 31,
        'Z': 33,
    }

    result = []
    for i in range(len(id_str)):
        if i == 0:
            result.append(first[id_str[0]])
            continue
        num = id_str[i]
        n = int(num)
        result.append(n)

    return result



def calculate_checksum(id_numbers: [int]) -> int:
    weights = [1, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    total = 0
    for i in range(len(id_numbers)):
        num = id_numbers[i]
        total += num * weights[i]
    checksum = 10 - (total % 10)
    if checksum == 10:
        return 0
    return checksum


def generate_random_id() -> str:
  id9 = generate_id9()
  id_numbers = convert_id_to_numbers(id9)
  check_code = calculate_checksum(id_numbers)
  return f"{id9}{check_code}"


def convert_to_train_id9(id9: str) -> [int]:
    first = ord(id9[0]) - ord('A') + 10
    numbers = [first]
    for n in range(1, len(id9)):
        num = ord(id9[n]) - ord('0')
        numbers.append(num)
    return numbers


def generate_random_id_for_train():
    id9 = generate_id9()
    train_id9 = convert_to_train_id9(id9)
    id9_numbers = convert_id_to_numbers(id9)
    check_code = calculate_checksum(id9_numbers)
    return (
        id9 + f'{check_code}',
        train_id9,
        check_code
    )

