import random

# palindromes
def gen_good_examples1(examples_num = 500):
    """
    gen palindromes good examples
    :param examples_num: num of examples
    :return: palindromes good examples
    """
    good_example_list = []
    for i in range(examples_num):
        cur_example = ''
        for i in range(random.randint(1, 50)):
            cur_example += str(random.randint(0, 1))
        cur_example + "".join(reversed(cur_example))
        good_example_list.append(cur_example)
    return good_example_list

def gen_bad_examples1(examples_num = 500):
    """
    gen palindromes bad examples
    :param examples_num: num of examples
    :return: palindromes bad examples
    """
    bad_example_list = []
    for i in range(examples_num):
        cur_example = ''
        for i in range(random.randint(1, 100)):
            cur_example += str(random.randint(0, 1))
        bad_example_list.append(cur_example)
    return bad_example_list

# middle value is 0
def gen_examples2(examples_num = 500):
    """
    gen middle value is zero example
    :param examples_num: num of examples
    :return: middle zero good and bad examples
    """
    good_example_list = []
    bad_example_list = []
    for i in range(examples_num):
        cur_example = ''
        num = random.randint(1, 100)
        for i in range(num):
            cur_example += str(random.randint(0, 1))
        if cur_example[num/2] == '0':
            good_example_list.append(cur_example)
        else:
            bad_example_list.append(cur_example)
    return good_example_list, bad_example_list

# first and last bits are similar
def gen_good_examples3(examples_num = 500):
    """
    gen same edge good examples
    :param examples_num: num of examples
    :return: same edge good examples
    """
    good_example_list = []
    for i in range(examples_num):
        cur_example = ''
        for i in range(random.randint(1, 100)):
            cur_example += str(random.randint(0, 1))
        cur_example += cur_example[0]
        good_example_list.append(cur_example)
    return good_example_list

def gen_bad_examples3(examples_num = 500):
    """
    gen same edge bad examples
    :param examples_num: num of examples
    :return: same edge bad examples
    """
    bad_example_list = []
    for i in range(examples_num):
        cur_example = ''
        for i in range(random.randint(1, 100)):
            cur_example += str(random.randint(0, 1))
        cur_example += '1' if cur_example[0]=='0' else '0'
        bad_example_list.append(cur_example)
    return bad_example_list


# if __name__ == "__main__":
#
#     examples_num = 10
#     good_example_list = []
#     bad_example_list = []
#
#     good_example_list = gen_good_examples3(examples_num)
#     bad_example_list = gen_bad_examples3(examples_num)
#
#     #good_example_list, bad_example_list = gen_examples2()
#     print good_example_list
#     print bad_example_list
