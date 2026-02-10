'''
MS - Artificial Intelligence and Machine Learning
Course: CSC510: Foundations of Artificial Intelligence
Module 4: Critical Thinking Assignment
Professor: Dr. Bingdong Li
Created by Mukul Mondal
February 7, 2026

Problem statement: 
Define a simple real-world search problem requiring a heuristic solution. 
You can base the problem on the 8-puzzle (or n-puzzle) problem, Towers of Hanoi, or even Traveling Salesman.

Write an interactive Python script (using either simpleAI's library or your resources) that utilizes 
either Best-First search, Greedy Best First search, Beam search, or A* search methods to calculate 
an appropriate output based on the proposed function. 

The search function does not have to be optimal nor efficient but must define an initial state, a goal state, 
reliably produce results by finding the sequence of actions leading to the goal state. 

Submission should be in an easily executable Python file alongside instructions for testing. 
Please include in your submission the type of search algorithm used along with at least a paragraph justifying your choice. 
'''

from os import system, name
import heapq
import math


# Clears the terminal
# This function clears the terminal screen based on the operating system.
# Returns: None
# This function is not required based on the requirements, but it improves user experience by providing a clean interface.
def clearScreen():
    if name == 'nt':  # For windows
        _ = system('cls')
    else:             # For mac and linux(here, os.name is 'posix')
        _ = system('clear')
    return

# ---------------------------------------------------------
#  Read user input points. delivery locations, and starting point.
# ---------------------------------------------------------
def read_user_input_fullPath():
    validInput: bool = False
    uInput: str = ''
    print("Enter coordinates of the delivery points in the format '(x1, y1); (x2, y2); ...' or 'q' to quit.")
    print("First coordinate should be the starting point, not a regular delivery point.")
    print("Example input: 1,2; 13,4; 5,16;  3 ,20;  0, 0")
    points = []
    while not validInput:        
        uInput = input("Enter coordinates of the points as: (x1, y1); (x2, y2);... : ")
        if uInput == 'q':
            return None
        try:            
            for pair in uInput.split(';'):
                x, y = map(int, pair.strip().split(','))
                points.append((x, y))
            validInput = True
        except ValueError:
            print("Invalid input. Please enter coordinates in the format '(x1, y1); (x2, y2); ...'.")
    return points


def read_user_input_point():
    validInput: bool = False
    uInput: str = ''
    while not validInput:
        uInput = input("Enter coordinates of the delivery point (x1, y1) separated by a comma (or 'q' to quit): ")
        if uInput == 'q':
            return None
        try:
            # x2, y2 = map(float, input("Enter coordinates of second point (x2, y2): ").split(','))
            x1, y1 = map(int, uInput.split(','))
            validInput = True
        except ValueError:
            print("Invalid input. Please enter coordinates in the format 'x, y'.")    
    return (x1,y1)


# ---------------------------------------------------------
#  Calculate Straight-line distance between 2 points
# ---------------------------------------------------------
def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# ---------------------------------------------------------
#  Best-first search Algorithm
# ---------------------------------------------------------
def best_first_search(start, deliveries, locations):
    """
    start: starting location name (e.g., "Start")
    deliveries: set of delivery location names
    locations: dict mapping name -> (x, y) coordinates
    """
    # Priority queue: (heuristic, state)
    frontier = []
    start_state = (start, frozenset(deliveries))
    heapq.heappush(frontier, (0, start_state))

    came_from = {start_state: None}

    while frontier:
        _, (current, remaining) = heapq.heappop(frontier)

        # Goal: no deliveries left
        if not remaining:
            return reconstruct_path(came_from, (current, remaining))

        # Expand neighbors
        for nxt in remaining:
            new_remaining = remaining - {nxt}
            new_state = (nxt, new_remaining)

            if new_state not in came_from:
                h = distance(locations[current], locations[nxt])
                heapq.heappush(frontier, (h, new_state))
                came_from[new_state] = (current, remaining)
    return None

# ---------------------------------------------------------
#  Reconstruct path. Algorithm.
# ---------------------------------------------------------
def reconstruct_path(came_from, state):
    path = []
    while state is not None:
        path.append(state[0])
        state = came_from[state]
    return list(reversed(path))


# ---------------------------------------------------------
#  Execution of the Algorithm
# ---------------------------------------------------------
if __name__ == "__main__":
    clearScreen()
    # Read user inputs, Coordinates for the delivery points and starting point.
    #
    # Assumption: The first point entered is the starting point, and the rest are the actual delivery points.
    # Delivery point names generated as "Delivery_Point_(1.2)", "Delivery_Point_(2,5)", etc., 
    #          while the starting point will be named "Start".
    delivery_points = read_user_input_fullPath()
    if delivery_points is None or len(set(delivery_points)) < 2:
        print("You did not enter valid or enough delivery points. Exiting program.")
        exit(0)
    #print("Delivery points:", delivery_points)

    # Prepare locations dictionary with delivery point names and coordinates
    locations = {}
    for i, point in enumerate(delivery_points):
        if i == 0:
            locations["Start"] = point
        else:
            if locations["Start"] != point:
                x,y = point
                locations[f"Delivery_Point_({x},{y})"] = point            
    
    deliveryPoints = [x for x in locations.keys() if x != "Start"] # all points except the 'start point'
    if deliveryPoints is None or len(deliveryPoints) < 2:
        print("We did not receive valid or enough delivery points. Exiting program.")
        exit(0)

    # Call best-first search Algorithm to find the optimal route
    route = best_first_search("Start", deliveryPoints, locations)

    # Print the best route with full details including coordinates
    print("Best-first route:", route)
    stPoint: str = ""
    fullRoute: str = ""
    for i in route:
        x,y = locations[i]
        if i == route[-1]:
            fullRoute += f"{i}"
        else:
            if i == "Start":
                stPoint = f" ==> {i}({x},{y})"
                fullRoute += f"{i}({x},{y}) ==> "
            else:
                fullRoute += f"{i} ==> "
    
    print("Full Route Details:\n",fullRoute, stPoint)
