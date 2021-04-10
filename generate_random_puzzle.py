import random


def generate_random_puzzles(size, filename):
    size = int(size)
    numbers = list(range(1, (size * size) + 1))
    f = open(filename, "w+")
    for i in range(0, 20):
        random.shuffle(numbers)
        start = 0
        end = size
        for j in range(size):
            current = numbers[start:end]
            for k in range(size):
                f.write("{0},".format(current[k]))
            start = end
            end += size

        f.write("\n")
    f.close()