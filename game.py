import argparse
import copy
from functools import cmp_to_key
import math
import time
from typing import TextIO
from generate_random_puzzle import generate_random_puzzles


def start(size, filename, heuristic):
    if size is not None:
        generate_random_puzzles(size, filename)

    total_game_cost = 0
    total_time_taken = 0
    total_search_cost = 0
    total_solution_cost = 0
    total_no_solutions = 0

    puzzles: list = build_puzzles(filename)

    for puzzle in puzzles:
        print('Solving puzzle: ')
        print(*puzzle)

        attempt_at_solved_game_board, game_cost, time_taken, search_cost, solution_cost, no_solution = depth_first(
            puzzle)
        attempt_at_solved_game_board, game_cost, time_taken, search_cost, solution_cost, no_solution = iterative_deepening(
            puzzle)
    attempt_at_solved_game_board, game_cost, time_taken, search_cost, solution_cost, no_solution = a_star(puzzle,
                                                                                                          int(
                                                                                                              heuristic))

    total_game_cost += game_cost
    total_time_taken += time_taken
    total_search_cost += search_cost
    total_solution_cost += solution_cost
    if no_solution is True:
        total_no_solutions += 1

    print('Final puzzle:')
    print(*attempt_at_solved_game_board)

    print("\n")
    print("Total Time taken: " + str(round(total_time_taken, 4)) + " seconds")
    print("Average Time taken: " + str(round(total_time_taken / 20, 4)) + " seconds")
    # print("\n")
    print("Total Game Cost: " + str(total_game_cost))
    print("Average Game Cost: " + str(total_game_cost / 20))
    # print("\n")
    print("Total Search Cost: " + str(total_search_cost))
    print("Average Search Cost: " + str(total_search_cost / 20))
    # print("\n")
    print("Total Solution taken: " + str(total_solution_cost))
    print("Average Solution taken: " + str(total_solution_cost / 20))
    # print("\n")
    print("Total No Solutions: " + str(total_no_solutions))
    print("Average No Solutions: " + str(total_no_solutions / 20))


def build_puzzles(filename):
    puzzles_file: TextIO = open(filename, 'r')
    lines = puzzles_file.readlines()

    puzzles = []
    for line in lines:
        line: str = line.replace(",\n", "")

        size = int(math.sqrt(len(line.split(","))))
        puzzle: list = []
        start_index = 0
        end_index = size

        for i in range(size):
            nums = [int(x) for x in line.split(",")]  # convert from str to int
            puzzle.append(nums[start_index:end_index])
            start_index = end_index
            end_index = end_index + size
        puzzles.append(puzzle)
    return puzzles


def depth_first(unsolved_game_board):
    start_time = time.time()
    game_cost = [0]
    search_cost = 0
    solution_cost = [0]
    open_game_stack = []
    closed_game_stack = []
    game_board = [list(x) for x in unsolved_game_board]
    parent_dict = {}

    open_game_stack.append(copy.deepcopy(game_board))
    while open_game_stack:

        current_state = open_game_stack.pop(0)
        closed_game_stack.append(current_state)

        # Check if board is solved
        if goal_test(current_state) is True:
            print_solution_path(solution_cost, parent_dict, str(current_state))
            print_search_path(game_cost, closed_game_stack)
            search_cost = game_cost[0] * 2 - 2
            return current_state, game_cost[0], (time.time() - start_time), search_cost, solution_cost[0], False

        if is_time_more_than_sixty_seconds(start_time):
            return current_state, game_cost[0], (time.time() - start_time), search_cost, solution_cost[0], True

        new_list = generate_possible_moves(current_state, closed_game_stack)
        for i in new_list:
            parent_dict[str(i)] = str(current_state)

        if open_game_stack:
            for elm in open_game_stack:
                new_list.append(elm)
        open_game_stack = new_list


def iterative_deepening(unsolved_game_board):
    game_cost = [0]
    search_cost = 0
    solution_cost = [0]
    start_time = time.time()
    search_path = []
    limit = 0
    parent_dict = {}
    game_board = [list(x) for x in unsolved_game_board]

    while True:
        # current states become original board after each iteration
        current_state = copy.deepcopy(game_board)

        # reset after each iteration
        closed_game_stack = []
        # Check if board is solved
        if goal_test(current_state) is True:
            print_solution_path(solution_cost, parent_dict, str(current_state))
            print_search_path(game_cost, search_path)
            search_cost = game_cost[0] * 2 - 2
            return current_state, game_cost[0], (time.time() - start_time), search_cost, solution_cost[0], False
        if is_time_more_than_sixty_seconds(start_time):
            return current_state, game_cost[0], (time.time() - start_time), search_cost, solution_cost[0], True

        # add to closed game stack because not in list
        closed_game_stack.append(current_state)

        # if answer is not found, continue
        answer, modified_parent_dict, modified_closed_game_stack = dfs_with_limit(limit, current_state,
                                                                                  closed_game_stack, game_cost,
                                                                                  parent_dict)
        if answer:
            search_path.append(modified_closed_game_stack)
            print_solution_path(solution_cost, parent_dict, str(answer))
            print_search_path(game_cost, search_path)
            search_cost = game_cost[0] * 2 - 2
            return answer, game_cost[0], (time.time() - start_time), search_cost, solution_cost[0], False

        # increase limit
        limit += 1
        search_path.append(closed_game_stack)


def dfs_with_limit(limit, current_state, closed_game_stack, game_cost, parent_dict):
    # store nodes that arent the left most node
    # Key = limit
    # Value= list of game boards
    pending_game_stack = {}
    open_game_stack = []

    # going down
    # iterate limit amount of times
    for i in range(1, limit + 1):
        # there are i levels
        game_cost[0] = game_cost[0] + 1

        open_game_stack = generate_possible_moves(current_state, closed_game_stack)
        new_list = copy.deepcopy(open_game_stack)

        for j in open_game_stack:
            parent_dict[str(j)] = str(current_state)
        if open_game_stack:
            for elm in open_game_stack:
                new_list.append(elm)

        # empty open_game_stack means all potential moves have already been evaluated
        if not open_game_stack:
            return [], parent_dict, closed_game_stack

        # remove first element (left most node) and make it the new current_state
        current_state = open_game_stack.pop(0)
        # all the other elements get added to the dictionary
        pending_game_stack[limit - i] = copy.deepcopy(open_game_stack)

        # if new current state is not true, add it to closed
        if goal_test(current_state):
            closed_game_stack.append(current_state)
            return current_state, parent_dict, closed_game_stack

        # not sure if im allowed to do this
        # if current_state in closed_game_stack
        #     open_game_stack.pop(0)

        closed_game_stack.append(current_state)

    order = list(sorted(pending_game_stack.keys()))

    # coming back up
    # amount of children is the key and the limit
    for amount_of_children in order:
        for current_board in pending_game_stack[amount_of_children]:
            # if zero children, just check if its right
            if amount_of_children == 0:
                if goal_test(current_board):
                    closed_game_stack.append(current_board)
                    return current_board, parent_dict, closed_game_stack
                else:
                    closed_game_stack.append(current_board)
            else:
                # add the state to closed game and then recursive call
                closed_game_stack.append(current_board)
                answer, modified_parent_dict, modified_closed_game_stack = dfs_with_limit(amount_of_children,
                                                                                          current_board,
                                                                                          closed_game_stack, game_cost,
                                                                                          parent_dict)
                if answer:
                    return answer, parent_dict, closed_game_stack

    return [], parent_dict, closed_game_stack


def a_star(unsolved_game_board, heuristic):
    start_time: float = time.time()
    game_cost = [0]
    search_cost = 0
    solution_cost = [0]
    open_game_stack = []
    closed_game_stack = []
    current_state = []
    game_board = [list(x) for x in unsolved_game_board]
    parent_dict = {}

    open_game_stack.append(copy.deepcopy((0, game_board)))

    while open_game_stack:

        # node is composed of (cost_to_top, the move)
        node = open_game_stack.pop(0)
        price_to_top = node[0]
        current_state = node[1]

        closed_game_stack.append(current_state)

        # Check if board is solved
        if goal_test(current_state) is True:
            print_solution_path(solution_cost, parent_dict, str(current_state))
            print_search_path(game_cost, closed_game_stack)
            search_cost = game_cost[0] * 2 - 2
            return current_state, game_cost[0], (time.time() - start_time), search_cost, solution_cost[0], False

        # Check if time out
        if is_time_more_than_sixty_seconds(start_time):
            return current_state, game_cost[0], (time.time() - start_time), search_cost, solution_cost[0], True

        new_list = generate_possible_moves(current_state, closed_game_stack)
        pair_list = []

        for elm in new_list:
            parent_dict[str(elm)] = str(current_state)
            pair_list.append((price_to_top + 1, elm))

        if open_game_stack:
            for elm in open_game_stack:
                pair_list.append(elm)

        open_game_stack = pair_list
        open_game_stack = sort_by_f(open_game_stack, heuristic)

    closed_game_stack.reverse()
    return current_state, game_cost[0], (time.time() - start_time), search_cost, solution_cost[0], False


def sort_by_f(open_game_stack, heuristic_to_choose) -> list:
    ans = []
    if heuristic_to_choose == 1:
        ans = sorted(open_game_stack, key=cmp_to_key(compare_h1))
    if heuristic_to_choose == 2:
        ans = sorted(open_game_stack, key=cmp_to_key(compare_h2))
    return ans


def compare_h1(item1, item2) -> int:
    if f_from_pair(item1, 1) > f_from_pair(item2, 1):
        return -1
    elif f_from_pair(item1, 1) < f_from_pair(item2, 1):
        return 1
    else:
        return 0


def compare_h2(item1, item2) -> int:
    if f_from_pair(item1, 2) < f_from_pair(item2, 2):
        return -1
    elif f_from_pair(item1, 2) > f_from_pair(item2, 2):
        return 1
    else:
        return 0


def f_from_pair(pair, heuristic) -> int:
    if int(heuristic) == 1:
        return h1(pair[1]) + pair[0]
    if int(heuristic) == 2:
        return h2(pair[1]) + pair[0]


def h1(board):
    score = 0
    list_board: list = [item for sublist in board for item in sublist]
    for i in range(len(list_board)):
        if list_board[i] == i + 1:
            score += 1
    return score


def h2(board):
    score = 0
    list_board = [item for sublist in board for item in sublist]
    height = len(board)

    for i, item in enumerate(list_board):
        prev_row, prev_col = int(i / height) + 1, (i % height) + 1
        goal_row, goal_col = math.ceil(item / height), item % height
        if goal_col == 0:
            goal_col = height
        score += abs(prev_row - goal_row) + abs(prev_col - goal_col)

    return score


def print_solution_path(solution_cost, parent_dict, state):
    print('--- SOLUTION PATH ---')
    final_state = state
    correct_order = []
    num = 0
    while parent_dict.get(state) is not None:
        correct_order.append(parent_dict.get(state))
        state = parent_dict.get(state)
        num += 1
        if num > 100:
            print("infinite loop!!!!!")
            break
    correct_order.reverse()
    for state in correct_order:
        print(str(state))
        solution_cost[0] += 1
    print(final_state)
    print('--- END OF SOLUTION PATH ---')


def print_search_path(search_cost, search_path: []):
    print('--- SEARCH PATH ---')
    for node in search_path:
        print(node)
        search_cost[0] += 1
    print('--- END OF SEARCH PATH ---')


def generate_possible_moves(current_state, closed_game_stack):
    amount = len(current_state)
    open_game_stack = []
    for row in range(amount):
        for col in range(amount):

            # row increment
            temp_board_row = copy.deepcopy(current_state)
            temp = temp_board_row[row][col]
            if row + 1 < amount:
                temp_board_row[row][col] = temp_board_row[row + 1][col]
                temp_board_row[row + 1][col] = temp
                if check_for_dupes(closed_game_stack, temp_board_row) is False:
                    open_game_stack.append(temp_board_row)

            # column increment
            temp_board_col = copy.deepcopy(current_state)
            temp = temp_board_col[row][col]
            if col + 1 < amount:
                temp_board_col[row][col] = temp_board_col[row][col + 1]
                temp_board_col[row][col + 1] = temp
                if check_for_dupes(closed_game_stack, temp_board_col) is False:
                    open_game_stack.append(temp_board_col)

    return open_game_stack


def goal_test(board):
    prev = 0
    list_board = [item for sublist in board for item in sublist]
    for i in range(len(list_board)):
        if list_board[i] - 1 != prev:
            return False
        prev = list_board[i]
    return True


def check_for_dupes(visited_boards, board2):
    for i in range(len(visited_boards)):
        if visited_boards[i] == board2:
            return True
    return False


def is_time_more_than_sixty_seconds(start_time):
    if (time.time() - start_time) > 60:
        print('no solution')
        return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', action="store", default="puzzles.txt")
    parser.add_argument('--size', action="store", default=2)
    parser.add_argument('--heuristic', action="store", default=2)
    start(parser.parse_args().size, parser.parse_args().filename, parser.parse_args().heuristic)
