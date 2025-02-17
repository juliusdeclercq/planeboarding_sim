import numpy as np
import numpy.random as rnd
import random
import time
from statistics import mean
import matplotlib.pyplot as plt

def BoardingStrat(type):
    list_order = []
    if type == "backtofront":
        zone1_seats = []
        zone2_seats = []
        zone3_seats = []
        zone4_seats = []
        zone5_seats = []
        zone6_seats = []
        for row in [1, 2, 3, 4, 5]:
            for col in range(6):
                zone1_seats.append((row, col + 1))
        for row in [6, 7, 8, 9, 10]:
            for col in range(6):
                zone2_seats.append((row, col + 1))
        for row in [11, 12, 13, 14, 15]:
            for col in range(6):
                zone3_seats.append((row, col + 1))
        for row in [16, 17, 18, 19, 20]:
            for col in range(6):
                zone4_seats.append((row, col + 1))
        for row in [21, 22, 23, 24, 25]:
            for col in range(6):
                zone5_seats.append((row, col + 1))
        for row in [26, 27, 28, 29, 30]:
            for col in range(6):
                zone6_seats.append((row, col + 1))
        while bool(zone6_seats) is True:
            choice = random.sample(zone6_seats, 1)[0]
            remove = zone6_seats.index(choice)
            list_order.append(choice)
            zone6_seats.pop(remove)
        while bool(zone5_seats) is True:
            choice = random.sample(zone5_seats, 1)[0]
            remove = zone5_seats.index(choice)
            list_order.append(choice)
            zone5_seats.pop(remove)
        while bool(zone4_seats) is True:
            choice = random.sample(zone4_seats, 1)[0]
            remove = zone4_seats.index(choice)
            list_order.append(choice)
            zone4_seats.pop(remove)
        while bool(zone3_seats) is True:
            choice = random.sample(zone3_seats, 1)[0]
            remove = zone3_seats.index(choice)
            list_order.append(choice)
            zone3_seats.pop(remove)
        while bool(zone2_seats) is True:
            choice = random.sample(zone2_seats, 1)[0]
            remove = zone2_seats.index(choice)
            list_order.append(choice)
            zone2_seats.pop(remove)
        while bool(zone1_seats) is True:
            choice = random.sample(zone1_seats, 1)[0]
            remove = zone1_seats.index(choice)
            list_order.append(choice)
            zone1_seats.pop(remove)
    elif type == "outsidein":
        window_seats = []
        middle_seats = []
        aisle_seats = []
        for row in range(30):
            for col in [1,6]:
                window_seats.append((row + 1, col))
        for row in range(30):
            for col in [2,5]:
                middle_seats.append((row + 1, col))
        for row in range(30):
            for col in [3,4]:
                aisle_seats.append((row + 1, col))
        while bool(window_seats) is True:
            choice = random.sample(window_seats, 1)[0]
            remove = window_seats.index(choice)
            list_order.append(choice)
            window_seats.pop(remove)
        while bool(middle_seats) is True:
            choice = random.sample(middle_seats, 1)[0]
            remove = middle_seats.index(choice)
            list_order.append(choice)
            middle_seats.pop(remove)
        while bool(aisle_seats) is True:
            choice = random.sample(aisle_seats, 1)[0]
            remove = aisle_seats.index(choice)
            list_order.append(choice)
            aisle_seats.pop(remove)
    elif type == "rotatingzone":
        zone1_seats = []
        zone2_seats = []
        zone3_seats = []
        zone4_seats = []
        zone5_seats = []
        zone6_seats = []
        for row in [1,2,3,4,5]:
            for col in range(6):
                zone1_seats.append((row, col+1))
        for row in [6,7,8,9,10]:
            for col in range(6):
                zone2_seats.append((row, col+1))
        for row in [11,12,13,14,15]:
            for col in range(6):
                zone3_seats.append((row, col+1))
        for row in [16,17,18,19,20]:
            for col in range(6):
                zone4_seats.append((row, col+1))
        for row in [21,22,23,24,25]:
            for col in range(6):
                zone5_seats.append((row, col+1))
        for row in [26,27,28,29,30]:
            for col in range(6):
                zone6_seats.append((row, col+1))
        while bool(zone6_seats) is True:
            choice = random.sample(zone6_seats, 1)[0]
            remove = zone6_seats.index(choice)
            list_order.append(choice)
            zone6_seats.pop(remove)
        while bool(zone1_seats) is True:
            choice = random.sample(zone1_seats, 1)[0]
            remove = zone1_seats.index(choice)
            list_order.append(choice)
            zone1_seats.pop(remove)
        while bool(zone5_seats) is True:
            choice = random.sample(zone5_seats, 1)[0]
            remove = zone5_seats.index(choice)
            list_order.append(choice)
            zone5_seats.pop(remove)
        while bool(zone2_seats) is True:
            choice = random.sample(zone2_seats, 1)[0]
            remove = zone2_seats.index(choice)
            list_order.append(choice)
            zone2_seats.pop(remove)
        while bool(zone4_seats) is True:
            choice = random.sample(zone4_seats, 1)[0]
            remove = zone4_seats.index(choice)
            list_order.append(choice)
            zone4_seats.pop(remove)
        while bool(zone3_seats) is True:
            choice = random.sample(zone3_seats, 1)[0]
            remove = zone3_seats.index(choice)
            list_order.append(choice)
            zone3_seats.pop(remove)
    elif type == "optimal":
        zone1_seats = []
        zone2_seats = []
        zone3_seats = []
        zone4_seats = []
        zone5_seats = []
        zone6_seats = []
        zone7_seats = []
        zone8_seats = []
        zone9_seats = []
        zone10_seats = []
        zone11_seats = []
        zone12_seats = []
        for row in range(30, 0, -2):
            zone1_seats.append((row, 1))
        for row in range(30, 0, -2):
            zone2_seats.append((row, 6))
        for row in range(29, 0, -2):
            zone3_seats.append((row, 1))
        for row in range(29, 0, -2):
            zone4_seats.append((row, 6))
        for row in range(30, 0, -2):
            zone5_seats.append((row, 2))
        for row in range(30, 0, -2):
            zone6_seats.append((row, 5))
        for row in range(29, 0, -2):
            zone7_seats.append((row, 2))
        for row in range(29, 0, -2):
            zone8_seats.append((row, 5))
        for row in range(30, 0, -2):
            zone9_seats.append((row, 3))
        for row in range(30, 0, -2):
            zone10_seats.append((row, 4))
        for row in range(29, 0, -2):
            zone11_seats.append((row, 3))
        for row in range(29, 0, -2):
            zone12_seats.append((row, 4))
        while bool(zone1_seats) is True:
            choice = random.sample(zone1_seats, 1)[0]
            remove = zone1_seats.index(choice)
            list_order.append(choice)
            zone1_seats.pop(remove)
        while bool(zone2_seats) is True:
            choice = random.sample(zone2_seats, 1)[0]
            remove = zone2_seats.index(choice)
            list_order.append(choice)
            zone2_seats.pop(remove)
        while bool(zone3_seats) is True:
            choice = random.sample(zone3_seats, 1)[0]
            remove = zone3_seats.index(choice)
            list_order.append(choice)
            zone3_seats.pop(remove)
        while bool(zone4_seats) is True:
            choice = random.sample(zone4_seats, 1)[0]
            remove = zone4_seats.index(choice)
            list_order.append(choice)
            zone4_seats.pop(remove)
        while bool(zone5_seats) is True:
            choice = random.sample(zone5_seats, 1)[0]
            remove = zone5_seats.index(choice)
            list_order.append(choice)
            zone5_seats.pop(remove)
        while bool(zone6_seats) is True:
            choice = random.sample(zone6_seats, 1)[0]
            remove = zone6_seats.index(choice)
            list_order.append(choice)
            zone6_seats.pop(remove)
        while bool(zone7_seats) is True:
            choice = random.sample(zone7_seats, 1)[0]
            remove = zone7_seats.index(choice)
            list_order.append(choice)
            zone7_seats.pop(remove)
        while bool(zone8_seats) is True:
            choice = random.sample(zone8_seats, 1)[0]
            remove = zone8_seats.index(choice)
            list_order.append(choice)
            zone8_seats.pop(remove)
        while bool(zone9_seats) is True:
            choice = random.sample(zone9_seats, 1)[0]
            remove = zone9_seats.index(choice)
            list_order.append(choice)
            zone9_seats.pop(remove)
        while bool(zone10_seats) is True:
            choice = random.sample(zone10_seats, 1)[0]
            remove = zone10_seats.index(choice)
            list_order.append(choice)
            zone10_seats.pop(remove)
        while bool(zone11_seats) is True:
            choice = random.sample(zone11_seats, 1)[0]
            remove = zone11_seats.index(choice)
            list_order.append(choice)
            zone11_seats.pop(remove)
        while bool(zone12_seats) is True:
            choice = random.sample(zone12_seats, 1)[0]
            remove = zone12_seats.index(choice)
            list_order.append(choice)
            zone12_seats.pop(remove)
    elif type == "pracoptimal":
        zone1_seats = []
        zone2_seats = []
        zone3_seats = []
        zone4_seats = []
        for row in range(30, 0, -2):
            for col in [1, 2, 3]:
                zone1_seats.append((row, col))
        for row in range(30, 0, -2):
            for col in [4, 5, 6]:
                zone2_seats.append((row, col))
        for row in range(29, 0, -2):
            for col in [1, 2, 3]:
                zone3_seats.append((row, col))
        for row in range(29, 0, -2):
            for col in [4, 5, 6]:
                zone4_seats.append((row, col))
        while bool(zone1_seats) is True:
            choice = random.sample(zone1_seats, 1)[0]
            remove = zone1_seats.index(choice)
            list_order.append(choice)
            zone1_seats.pop(remove)
        while bool(zone2_seats) is True:
            choice = random.sample(zone2_seats, 1)[0]
            remove = zone2_seats.index(choice)
            list_order.append(choice)
            zone2_seats.pop(remove)
        while bool(zone3_seats) is True:
            choice = random.sample(zone3_seats, 1)[0]
            remove = zone3_seats.index(choice)
            list_order.append(choice)
            zone3_seats.pop(remove)
        while bool(zone4_seats) is True:
            choice = random.sample(zone4_seats, 1)[0]
            remove = zone4_seats.index(choice)
            list_order.append(choice)
            zone4_seats.pop(remove)
    elif type == "revpyramid":
        zone1_seats = []
        zone2_seats = []
        zone3_seats = []
        zone4_seats = []
        zone5_seats = []
        zone6_seats = []
        # Zone 1
        for row in range(15, 30):
            for col in [1, 6]:
                zone1_seats.append((row + 1, col))
        # Zone 2
        for row in range(7, 15):
            for col in [1, 6]:
                zone2_seats.append((row + 1, col))
        for row in range(23, 30):
            for col in [2, 5]:
                zone2_seats.append((row + 1, col))
        # Zone 3
        for row in range(0, 7):
            for col in [1, 6]:
                zone3_seats.append((row + 1, col))
        for row in range(15, 23):
            for col in [2, 5]:
                zone3_seats.append((row + 1, col))
        # Zone 4
        for row in range(23, 30):
            for col in [3, 4]:
                zone4_seats.append((row + 1, col))
        for row in range(7, 15):
            for col in [2, 5]:
                zone4_seats.append((row + 1, col))
        # Zone 5
        for row in range(0, 7):
            for col in [2, 5]:
                zone5_seats.append((row + 1, col))
        for row in range(15, 23):
            for col in [3, 4]:
                zone5_seats.append((row + 1, col))
        # Zone 6
        for row in range(0, 15):
            for col in [3, 4]:
                zone6_seats.append((row + 1, col))
        while bool(zone1_seats) is True:
            choice = random.sample(zone1_seats, 1)[0]
            remove = zone1_seats.index(choice)
            list_order.append(choice)
            zone1_seats.pop(remove)
        while bool(zone2_seats) is True:
            choice = random.sample(zone2_seats, 1)[0]
            remove = zone2_seats.index(choice)
            list_order.append(choice)
            zone2_seats.pop(remove)
        while bool(zone3_seats) is True:
            choice = random.sample(zone3_seats, 1)[0]
            remove = zone3_seats.index(choice)
            list_order.append(choice)
            zone3_seats.pop(remove)
        while bool(zone4_seats) is True:
            choice = random.sample(zone4_seats, 1)[0]
            remove = zone4_seats.index(choice)
            list_order.append(choice)
            zone4_seats.pop(remove)
        while bool(zone5_seats) is True:
            choice = random.sample(zone5_seats, 1)[0]
            remove = zone5_seats.index(choice)
            list_order.append(choice)
            zone5_seats.pop(remove)
        while bool(zone6_seats) is True:
            choice = random.sample(zone6_seats, 1)[0]
            remove = zone6_seats.index(choice)
            list_order.append(choice)
            zone6_seats.pop(remove)
    else:  # random
        seats = []
        for row in range(30):
            for col in range(6):
                seats.append((row+1, col+1))
        while bool(seats) is True:
            choice = random.sample(seats, 1)[0]
            remove = seats.index(choice)
            list_order.append(choice)
            seats.pop(remove)
    return list_order



def Queue_former(strategy):
    arrival_queue = BoardingStrat(strategy)
    passengers = []
    ID = 0
    for i in enumerate(arrival_queue):
        ID += 1
        passengers.append(PassengerGenerator(ID, i[1][0], i[1][1]))
    return passengers



def SeatingManager(arriving_passenger, seating_matrix, current_time):
    seat = arriving_passenger["seat"]
    row = arriving_passenger["row"]
    if seat==3 or seat==4:
        # Arriving has aisle seat
        aisle_seating = random.uniform(2, 5)  # Aisle seating time (aisle can be taken regardless of people already sitting in the row
        arriving_passenger["timespent"]["seating_time"] += aisle_seating
        seating_time = aisle_seating
    elif seat==2 or seat==5:
        # Arriving has middle seat
        if seat==2:
            target_seat = 3
        else:
            target_seat = 4
        if bool(seating_matrix[row-1][target_seat-1]) is True:
            # Aisle seat is taken
            aisle_seating = random.uniform(2, 5)  # Standup time of the person in the aisle seat
            middle_seating = random.uniform(3, 6)  # middle seating time
            seating_matrix[row-1][target_seat-1]["timespent"]["standup"] += aisle_seating*2 + middle_seating
            arriving_passenger["timespent"]["seating_time"] += middle_seating+aisle_seating
            seating_time = aisle_seating*2 + middle_seating
        else:
            middle_seating = random.uniform(3, 6)  # middle seating time
            arriving_passenger["timespent"]["seating_time"] += middle_seating
            seating_time = middle_seating
    else:
        # Arriving has window seat
        if seat==1:
            target_middle = 2
            target_aisle = 3
        else:
            target_middle = 5
            target_aisle = 4
        if bool(seating_matrix[row - 1][target_aisle - 1]) is True:
            if bool(seating_matrix[row - 1][target_middle - 1]) is True:
                # Both aisle and middle are taken
                aisle_seating = random.uniform(2, 5)  # Standup time of the person in the aisle seat
                middle_seating = random.uniform(3, 6)  # Standup time of the person in the middle seat
                window_seating = random.uniform(4, 7)  # window seating time
                arriving_passenger["timespent"]["seating_time"] += window_seating + aisle_seating + middle_seating
                seating_matrix[row - 1][target_aisle - 1]["timespent"]["standup"] += aisle_seating*2 + middle_seating*2 + window_seating
                seating_matrix[row - 1][target_middle - 1]["timespent"]["standup"] += middle_seating * 2 + window_seating
                seating_time = aisle_seating*2 + middle_seating*2 + window_seating
            else:
                # Only aisle seat is taken
                aisle_seating = random.uniform(2, 5)  # Standup time of the person in the aisle seat
                window_seating = random.uniform(4, 7)  # window seating time
                seating_matrix[row - 1][target_aisle - 1]["timespent"]["standup"] += aisle_seating * 2 + window_seating
                arriving_passenger["timespent"]["seating_time"] += window_seating + aisle_seating
                seating_time = aisle_seating * 2 + window_seating
        else:
            if bool(seating_matrix[row - 1][target_middle - 1]) is True:
                # Only middle seat is taken
                middle_seating = random.uniform(3, 6)  # Standup time of the person in the middle seat
                window_seating = random.uniform(4, 7)  # window seating time
                seating_matrix[row - 1][target_middle - 1]["timespent"]["standup"] += middle_seating * 2 + window_seating
                arriving_passenger["timespent"]["seating_time"] += window_seating + middle_seating
                seating_time = middle_seating * 2 + window_seating
            else:
                window_seating = random.uniform(4, 7)  # window seating time
                arriving_passenger["timespent"]["seating_time"] += window_seating
                seating_time = window_seating
                
    # Setting time measures.
    aisle_occupancy_time = arriving_passenger["timespent"]["luggage_stash"] + seating_time
    arriving_passenger["timespent"]["time_seated"] = current_time[0] + arriving_passenger["timespent"]["luggage_stash"] + arriving_passenger["timespent"]["seating_time"]
    arriving_passenger["timespent"]["board_time_cum"] = arriving_passenger["timespent"]["time_seated"] - arriving_passenger["timespent"]["time_entrance"]
    
    # Seat the arriving passenger in the seating matrix
    seating_matrix[row - 1][seat - 1] = arriving_passenger
    
    # Return the seating matrix, the time it takes for the aisle to clear.
    res_seating = [seating_matrix, aisle_occupancy_time]
    return res_seating

    
    
def PassengerGenerator(ID, row, seat):
    # Simulated parameters
    has_luggage = rnd.choice((1, 0), p=[0.8, 0.2])  # logical if person has a luggage (1-yes, 0-no)
    if has_luggage==1:
        luggage_stash = random.uniform(6, 30)  # seconds spent with luggage stash
    else:
        luggage_stash = 0
    walkspeed = random.uniform(0.27, 0.44)  # meter per second

    # Calculated attributes and containing these attributes
    walking_time = (0.7112/walkspeed) * row + 4*(0.4572/walkspeed)
    dict_res = {
        "row": row,
        "seat": seat,
        "has_luggage": has_luggage,
        "walkspeed": {'aisle_begin' : size_aisle_begin/walkspeed,
                      'aisle_seats' : size_aisle_seats/walkspeed},
        "aisle_pos": None,
        "seated": False,
        "ID": ID,
        "timespent": {
            "board_time_cum": None,
            "time_entrance": None,
            "time_seated": None,
            "walking": walking_time,
            "waiting": 0,
            "luggage_stash": luggage_stash,
            "seating_time": 0,
            "standup": 0
        }
    }
    return dict_res



def New_arrival(passengers, aisle, current_time, events):
    # We find the first passenger on the passengers list that has not yet been assigned a spot in the aisle.
    for i in range(len(passengers)):
        if passengers[i]['aisle_pos']==None:
            while aisle[0] == 0:
                passengers[i]['aisle_pos'] = 0
                passengers[i]['timespent']['time_entrance'] = current_time[0]
                aisle[0] = 1
                # Make a new entry for the passenger in the event list
                pass_event = Passenger_event(passengers[i]['ID'], passengers[i]['walkspeed']['aisle_begin'])
                events.append(pass_event)



def Passenger_event(ID, timer):
    return{'ID': ID, 'timer' : timer,}



def BoardingSimulator9000(current_time, passengers, aisle, events, seating_matrix, mu_arrival):
    # Step 1: Identify the next event 
    min_timer = min(events, key = lambda x: x['timer'])['timer']    # Find the time to the next event
    # Retrieve the event list index of the next event (by matching the smallest timer)
    action_index = next((index for (index, e) in enumerate(events) if e["timer"] == min_timer), None) 
    
    current_time[0] += min_timer # Update current time
    for i in range(len(events)):          
        events[i]['timer'] -= min_timer     # Update the event timers.
        
    # Step 2: Execute the next event 
    # First we define what happens if the event is a new arrival
    if events[action_index]['ID'] == 'Arrival':
        # In case the first spot on the aisle is occupied, the arrival is delayed.
        if bool(aisle[0]):
            events[action_index]['timer'] = 1   # Setting the delay arbitrarily to 1 second, NEED TO DISCUSS AND UPDATE. 
            # Note that we are only counting waiting time on the plane. Waiting time in the tube is 
        else:
            events[action_index]['timer'] = random.expovariate(1/mu_arrival) # Otherwise draw from an exp. distribution with mean 2 ALSO NEED TO DISCUSS.
            New_arrival(passengers, aisle, current_time, events)    # And let the new passenger arrive.
        # If all passengers are in the plane, remove the arrival event from the event list.        
        if all(p['aisle_pos'] is not None for p in passengers):
            del events[action_index]
        
    else: # If we are not dealing with a new arrival, we must be dealing with a passenger.
        # Designate the passenger that is going to move.
        pass_index = next((index for (index, d) in enumerate(passengers) if d["ID"] == events[action_index]['ID']), None)
        # Retrieve the passenger's position on the aisle.
        aisle_position = passengers[pass_index]['aisle_pos']    
        
        # First we check if the passenger has already been seated. If so, we remove the passenger's actions from the event list.
        if passengers[pass_index]['seated'] == True:
            del events[action_index]
            # Uncomment line below for continuously keeping track of the simulation.
            # print(f" Passenger {passengers[pass_index]['ID']} is seated at time {current_time[0]}.")
            # And his space on the aisle is cleared.
            aisle[aisle_position] = 0
        
        # If not yet seated, the passenger will try to move up a space on the aisle.
        else:   
            # If the next space on the aisle is occupied, the passenger needs to wait.
            # The way it is coded here, the waiting time is basically the movement resetting. I think this is the right way to go.
            if bool(aisle[aisle_position + 1]):
                # Taking into account smaller tiles at the beginning of the aisle.
                if aisle_position < l_aisle - nrows:    
                    # Schedule the passenger's next movement.
                    events[action_index]['timer'] = passengers[pass_index]['walkspeed']['aisle_begin']
                    # Update waiting time.
                    passengers[pass_index]['timespent']['waiting'] += passengers[pass_index]['walkspeed']['aisle_begin']
                # Now for the larger aisle tiles at the seats.
                else:           
                    events[action_index]['timer'] = passengers[pass_index]['walkspeed']['aisle_seats']
                    passengers[pass_index]['timespent']['waiting'] += passengers[pass_index]['walkspeed']['aisle_seats']
            
            # If the next space on the aisle is vacant, the passenger moves.
            else:
                aisle[aisle_position] = 0
                aisle[aisle_position + 1] = 1
                passengers[pass_index]['aisle_pos'] = aisle_position + 1
                
                # If he has not arrived at his row number yet, we simply schedule his next movement.
                if not passengers[pass_index]['aisle_pos'] == passengers[pass_index]['row'] + l_aisle - nrows - 1:
                    # Again accounting for different tile sizes. Some duplicate code cannot be avoided here.
                    if aisle_position < l_aisle - nrows:    
                        events[action_index]['timer'] = passengers[pass_index]['walkspeed']['aisle_begin']
                    else: 
                        events[action_index]['timer'] = passengers[pass_index]['walkspeed']['aisle_seats']
                
                # Now for something very important: if the passenger arrives at the row of his seat he needs to sit!
                else:
                    # Next time this passenger's event occurs, the passenger is seated and removed from the event list.
                    # Here we already mark this passenger as seated. 
                    passengers[pass_index]['seated'] = True
                    res_seating = SeatingManager(passengers[pass_index], seating_matrix, current_time)
                    seating_matrix = res_seating[0]
                    events[action_index]['timer'] = res_seating[1]
                    


def SimulationExecutor(strategy, nrows, nseats, l_aisle, size_aisle_seats, size_aisle_begin, mu_arrival):
    # Initialization
    current_time = [0]                                # Initializing global time. For some reason this cannot simply be an integer, otherwise it does not update (took me forever to find this).
    aisle = np.zeros(l_aisle)                       # The aisle is a binary vector, where '1' designates occupancy and '0' vacancy.
    seating_matrix = np.full((nrows,nseats), {})    # Initializing seating matrix
    passengers = Queue_former(strategy)             # Initializing the passengers according to the boarding strategy.
    events = [{'ID': 'Arrival', 'timer' : random.expovariate(1/mu_arrival)}]    # Initializing the event list. 
    # At first the only active event is that of the next arrival.
    # As passengers enter the plane, their next action gets its separate event.
    
    # Execution
    while len(events)>0:                            # Once everybody is seated there are no more scheduled events and the simulation is done. 
        BoardingSimulator9000(current_time, passengers, aisle, events, seating_matrix, mu_arrival)
    
    return seating_matrix
        


def Simulation_analysis(strategies, nrows, nseats, l_aisle, size_aisle_seats, size_aisle_begin, mu_arrival):
    
    # For each strategy and for each time measure, this function returns a list with all the measured times for all passengers in all simulations. 
    # Also, for each strategy and all time measures it returns a dictionary with all the means and a dictionary with all the 95th-percentiles of each simulation.
    # Lastly, it also returns a list with the total boarding times of each simulation for all strategies.
    
    # Setting up output structure
    # Running simulation once for setting up the output structure.
    seating_matrix = SimulationExecutor(strategies[0], nrows, nseats, l_aisle, size_aisle_seats, size_aisle_begin, mu_arrival)
    passenger_list = seating_matrix.reshape(-1)
    time_measures = []
    for strat in enumerate(strategies):
        keys = [key for key in passenger_list[0]['timespent']]
        
        
        stats = ['95thpercentile', 'mean']
        keys0 = stats + ['Total_boarding_time']
        keys.extend(keys0)
        time_measures.append({strat[1] : {key : [] for key in keys}})
        # Now for the dictionaries with means and 95th-percentiles.
        for stat in stats:
                time_measures[strat[0]][strat[1]][stat] = {key: [] for key in keys if key not in keys0}
    
        
        # Saving output to output structure.
        start = time.time()
        for i in range(nsims):
            seating_matrix = SimulationExecutor(strat[1], nrows, nseats, l_aisle, size_aisle_seats, size_aisle_begin, mu_arrival)
            passenger_list = seating_matrix.reshape(-1)
            for key in keys:
                if key not in keys0:
                    key_list = [p['timespent'][key] for p in passenger_list]
                    time_measures[strat[0]][strat[1]][key].extend(key_list)
                    
                    key_mean = mean(key_list)
                    key_95percentile = np.percentile(key_list, 95)
                    
                    time_measures[strat[0]][strat[1]]['mean'][key].extend([key_mean]) 
                    time_measures[strat[0]][strat[1]]['95thpercentile'][key].extend([key_95percentile]) 
            time_measures[strat[0]][strat[1]]['Total_boarding_time'].append(max(passenger_list, key = lambda x: x['timespent']['time_seated'])['timespent']['time_seated'])
            
        finish = time.time() - start
        print(f" Running {nsims} simulations of boarding strategy {strat[1]} was completed in {finish} seconds.")
        
    return time_measures

def Plot_distributions(strategies, time_measures):
    
    for key in pltkeys:
        plt.clf()
        # Define histogram parameters
        binwidth = 0.5
        bins = []
        for i, strat in enumerate(strategies):
            bins.append(np.arange(min(time_measures[i][strat][key]), max(time_measures[i][strat][key]) + binwidth, binwidth))
        
        # Plot histograms
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'grey']
        handles = []
        for i, strat in enumerate(strategies):
            n, bins, patches = plt.hist(time_measures[i][strat][key], bins=bins[i], alpha=0.5, color=colors[i], label=f'Strategy {strat}')
        
            # Calculate and plot means and 95th-percentiles
            mean, percentile = np.mean(time_measures[i][strat][key]), np.percentile(time_measures[i][strat][key], 95)
            handles.append(patches.Patch(color=colors[i], label=f'Strategy {strat}'))
        
        # Add legend and labels
        plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1))
        for i, strat in enumerate(strategies):
            mean, percentile = np.mean(time_measures[i][strat][key]), np.percentile(time_measures[i][strat][key], 95)
            plt.axvline(mean, color=colors[i], linestyle='--', label=f'Mean: {mean:.2f}, 95th percentile: {percentile:.2f}')
        
        plt.xlabel('Time in seconds')
        plt.ylabel('Frequency')
        plt.title(f'Histograms of {key} time')
        plt.savefig(f'histograms_{key}.pdf', bbox_inches='tight')

###############################################################################
### Magic numbers

# Set randomization seed
random.seed(987654321)
rnd.seed(987654321)
# Number of simulations per boarding strategy
nsims = 100

# All analyzed boarding strategies
strategies = ["backtofront", "outsidein", "rotatingzone", "optimal", "pracoptimal", "revpyramid"]
# strategies = ["rotatingzone"]

# Plane attributes
# Number of seat rows on the plane
nrows = 30
# Seats per row
nseats = 6
# Total length of the aisle
l_aisle = 35

# The space occupied on the aisle is larger next to the seats than on the first part.
size_aisle_seats = 0.7112
size_aisle_begin = 0.4572

# Mean arrival delay (in seconds)
mu_arrival = 2

# Time measures for which the distributions are to be plotted
pltkeys = ['board_time_cum', 'waiting', 'seating_time', 'standup', 'Total_boarding_time']

###############################################################################
### Simulation      (initialization is done in the SimulationExecutor function)

# Run the simulation!
time_measures = Simulation_analysis(strategies, nrows, nseats, l_aisle, size_aisle_seats, size_aisle_begin, mu_arrival)





###############################################################################
### Plotting

Plot_distributions(strategies, time_measures)

















            
             


















