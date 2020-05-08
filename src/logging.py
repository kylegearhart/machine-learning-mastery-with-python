def print_first_five_rows_of_data(data_rows):
    print('First five rows in dataset:')
    for row_index in range(5):
        print(data_rows[row_index])
    print('')


def print_tree_recursively(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % (depth * ' ', (node['property_index_to_split_on'] + 1), node['threshold_value']))
        print_tree_recursively(node['left_subtree'], depth + 1)
        print_tree_recursively(node['right_subtree'], depth + 1)
    else:
        print('%s[%s]' % (depth * ' ', node))
