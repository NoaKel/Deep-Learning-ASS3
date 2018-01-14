import random

def gen_rand_int_seq(sequence_limit_length):
    """
    generates random int sequence uniformally
    :param sequence_limit_length: max length
    :return: sequence
    """
    cur_num_seq = ''
    cur_seq_length = random.randint(1,sequence_limit_length)
    for index in range(cur_seq_length):
        cur_num_seq += str(random.randint(1,9))
    return cur_num_seq

def gen_good_examples(examples_num = 500, sequence_limit_length = 10):
    """
    generates good examples
    :param examples_num: number of examples
    :param sequence_limit_length: max length
    :return: good examples
    """
    good_example_list = []
    for i in range(examples_num):
        cur_example =''
        cur_example += gen_rand_int_seq(sequence_limit_length)
        cur_example += 'a' * random.randint(1,sequence_limit_length)
        cur_example += gen_rand_int_seq(sequence_limit_length)
        cur_example += 'b' * random.randint(1,sequence_limit_length)
        cur_example += gen_rand_int_seq(sequence_limit_length)
        cur_example += 'c' * random.randint(1,sequence_limit_length)
        cur_example += gen_rand_int_seq(sequence_limit_length)
        cur_example += 'd' * random.randint(1,sequence_limit_length)
        cur_example += gen_rand_int_seq(sequence_limit_length)
        good_example_list.append(cur_example)
    return good_example_list

def gen_bad_examples(examples_num = 500, sequence_limit_length = 10):
    """
    generates bad examples
    :param examples_num: number of examples
    :param sequence_limit_length: max length
    :return: bad examples
    """
    bad_example_list = []
    for i in range(examples_num):
        cur_example =''
        cur_example += gen_rand_int_seq(sequence_limit_length)
        cur_example += 'a' * random.randint(1,sequence_limit_length)
        cur_example += gen_rand_int_seq(sequence_limit_length)
        cur_example += 'c' * random.randint(1,sequence_limit_length)
        cur_example += gen_rand_int_seq(sequence_limit_length)
        cur_example += 'b' * random.randint(1,sequence_limit_length)
        cur_example += gen_rand_int_seq(sequence_limit_length)
        cur_example += 'd' * random.randint(1,sequence_limit_length)
        cur_example += gen_rand_int_seq(sequence_limit_length)
        bad_example_list.append(cur_example)
    return bad_example_list

if __name__ == "__main__":

    examples_num = 500
    sequence_limit_length = 10
    good_example_list = []
    bad_example_list = []

    good_example_list = gen_good_examples(examples_num,sequence_limit_length)
    bad_example_list = gen_bad_examples(examples_num,sequence_limit_length)

    with open('pos_examples', 'w') as good_file:
        for word in good_example_list:
            good_file.write(word + '\n')
    with open('neg_examples', 'w') as bad_file:
        for word in bad_example_list:
            bad_file.write(word + '\n')
