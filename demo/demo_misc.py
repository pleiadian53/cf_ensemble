
"""
Demo of commonly used python idioms. 


Reference
---------
1. sorting dictionary: 
   https://www.pythoncentral.io/how-to-sort-python-dictionaries-by-key-or-value/

"""



def sort_dictionary(): 
    def repeats(string):
        # Lower the case in the string
        string = string.lower()   
 
        # Get a set of the unique letters
        uniques = set(string)
 
        # Count the max occurrences of each unique letter
        counts = [string.count(letter) for letter in uniques]
 
        return max(counts)

    # [ref] https://www.pythoncentral.io/how-to-sort-python-dictionaries-by-key-or-value/

    numbers = {'first': 1, 'second': 2, 'third': 3, 'Fourth': 4}
    sorted(numbers, key=numbers.__getitem__)   # pass a function that select item/value

    month = dict(one='January',
                 two='February',
                 three='March',
                 four='April',
                 five='May')
    numbermap = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}  # associate key with some numeric values
    sorted(month, key=numbermap.__getitem__)

    month_names = [month[i] for i in sorted(month, key=numbermap.__getitem__)]
    print month_names

    sorted(month.values(), key=repeats, reverse=True)

    return


if __name__ == "__main__": 
    test()