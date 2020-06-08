from collections import OrderedDict as od
from collections import namedtuple
DEBUGGING = False
FIND_LEAD_AGENT_THRESHOLD = 15 #TODO: compute this automatically
CAR_COLORS = ['blue', 'brown', 'gray', 'green', 'light_green', 'orange', 'red', 'yellow']
IOMap = namedtuple('IOMap', ['sources','sinks','map'])
TLNode = namedtuple('TLNode', ['xy', 'green', 'yellow', 'red'])
Intersection = namedtuple('Intersection', ['tiles', 'pcorner', 'mcorner', 'height', 'width'])
Neighbor = namedtuple('Neighbor', ['xyt', 'weight', 'name'])
DIRECTION_TO_VECTOR  = od()
DIRECTION_TO_VECTOR['east'] = [0, 1]
DIRECTION_TO_VECTOR['west'] = [0, -1]
DIRECTION_TO_VECTOR['north'] = [-1, 0]
DIRECTION_TO_VECTOR['south'] = [1, 0]
AGENT_CHAR = [str(i) for i in range(10)]
CHOSEN_IDs = []

