import copy

class OR_Node(object) :

    def __init__(self, s, terminal, depth, parent=None) :
        self._state = copy.deepcopy(s)
        self._parent = parent
        self._children = {}
        self._terminal = terminal
        self._visits = 0
        self._depth = depth
        self.stale = False

    @property
    def state(self) :
        return self._state

    @property
    def children(self) :
        return self._children

    @property
    def parent(self) :
        return self._parent

    @property
    def terminal(self) :
        return self._terminal

    @property
    def depth(self):
        return self._depth

    @property
    def num_visits(self):
        return self._visits

    def increment_visits(self):
        self._visits += 1

    def decrement_visits(self, value):
        self._visits -= value

class AND_Node(object) :
    def __init__(self, a, parent: OR_Node) :
        self._parent = parent
        self._parent.children[a] = self
        self._action = a
        self._children = set()
        self._visits = 0

    @property
    def state(self) :
        return self._parent.state

    @property
    def action(self) :
        return self._action

    @property
    def children(self) :
        return self._children

    @property
    def parent(self) :
        return self._parent

    @property
    def num_visits(self) :
        return self._visits

    def increment_visits(self) :
        self._visits += 1

    def decrement_visits(self, value):
        self._visits -= value

    def add_child(self, succ: OR_Node, R):
        self._children.add((succ, R))