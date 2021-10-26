# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from src.memory.andOrGraph import OR_Node, AND_Node
import copy
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import ttest_ind

# From https://keras.io/examples/rl/deep_q_network_breakout/
def deepmind_model_classifier(num_classes):
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(84, 84, 4,))
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu", trainable=True)(x )
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu", trainable=True)(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu", trainable=True)(layer2)

    layer4 = layers.Flatten(trainable=True)(layer3)

    layer5 = layers.Dense(512, activation="relu", trainable=True)(layer4)
    outputs = layers.Dense(num_classes, activation="softmax", trainable=True)(layer5) #EDIT: Changed from linear layer to softmax for classification

    return keras.Model(inputs=inputs, outputs=outputs)

def deepmind_model_V():
    # Network defined by the Deepmind paper but with single linear output
    inputs = layers.Input(shape=(84, 84, 4,))
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(x)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    outputs = layers.Dense(1, activation="linear")(layer5) #EDIT: Changed from linear layer to softmax for classification

    return keras.Model(inputs=inputs, outputs=outputs)

# base class for base policies
class BasePolicy(object):

    def __init__(self, numActions):
        self.numActions = numActions

    def train_with_batch(self):
        pass

    def predict_probs(self, state):
        pass

    def state_action_pair_to_train(self, state, action):
        pass


# Random base policy
class UniformPolicy(BasePolicy):

    def __init__(self, numActions):
        super(UniformPolicy, self).__init__(numActions)

    def predict_probs(self, state):
        return np.ones(self.numActions) / self.numActions  # Uniform prob distribution

    def value_func(self, state):
        return 0.0

    def train_with_batch(self, model_save_dir, rewards):
        return

    def state_action_pair_to_train(self, state, action, state_next, done, reward):
        return

    def state_action_pair_to_classify(self, state, action):
        return

def hasRewardsImproved(rewardsSinceLastUpdate, prevSuccRewards):
    t, p = ttest_ind(rewardsSinceLastUpdate, prevSuccRewards, equal_var=False)
    print("t {} p {} rewards this time {} and last time {}".format(t, p, rewardsSinceLastUpdate, prevSuccRewards))
    if t < 0 and p < 0.1:
        return False
    else:
        return True

class NN_V_Policy(BasePolicy): # Adapted from https://keras.io/examples/rl/deep_q_network_breakout/

    def __init__(self, numActions, useValueFunction, useAll):
        super(NN_V_Policy, self).__init__(numActions)
        self.useValueFunction = useValueFunction

        # Networks used for training
        self.model = deepmind_model_V()
        self.model_target = deepmind_model_V()
        self.policyModel = deepmind_model_classifier(numActions)

        # Deployed models being used in lookahead
        self.model_dep = deepmind_model_V()
        self.policyModel_dep = deepmind_model_classifier(numActions)

        self.optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
        self.loss_function = keras.losses.huber

        self.batch = ([], [])
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []

        self.hasTrained = False
        self.frame_count = 0

        self.update_after_actions = 32
        self.batch_size = 128
        self.gamma = 0.99
        self.num_actions = numActions

        # Max length of batch buffer.
        self.max_memory_length = 20480
        # How often to update the target network.
        self.update_target_network = 10000
        self.epsilon = 0.1
        if useAll: # If use all transitions in lookahead for learning value function.
            self.minFramesBeforeUsing = 1000000
            self.minFramesBeforeUsingcosttogo = 1000000
            self.frame_train_interval = 1000000
        else:
            self.minFramesBeforeUsing = 10000 # Equal to train every 1M sim calls as only 1 out of every 100 sims added to train set
            self.minFramesBeforeUsingcosttogo = 10000
            self.frame_train_interval = 10000


        self.numberOfEpsWhenLastLearningIt = 0
        self.numberOfFramesWhenLastLearning = 0

        self.prevSuccRewards = None
        self.prevpolicyRewards = None
        self.prevQRewards = None
        self.prevPolicyWeights = None
        self.prevModelWeights = None
        self.useRandomPolicy = False
        self.useNoCostToGo = False

        self.lastTrainedAt = 0
        self.numberOfTrainings = 0

    def train_with_batch(self, model_save_dir, rewards):
        # Train policy function, value function has been training as been collecting transitions as if we use
        # all transitions within the lookahead it is too much memory to store 1M transitions.
        # Only train/update the deployed networks being used in the lookahead if the previous network used with the
        # lookahead is accepted according to the learning schedule.
        trained = False
        if self.frame_count > self.minFramesBeforeUsing + self.numberOfFramesWhenLastLearning: # at least 2 eps complete #If buffer has min number of examples then train
            rewardsSinceLastUpdate = rewards[self.numberOfEpsWhenLastLearningIt:]
            self.numberOfEpsWhenLastLearningIt = len(rewards)
            self.numberOfFramesWhenLastLearning = self.frame_count

            if self.prevSuccRewards is not None and not hasRewardsImproved(rewardsSinceLastUpdate, self.prevSuccRewards):
                # Do not train and reinit networks back to previous network.
                self.model.set_weights(self.prevModelWeights)
                self.model_dep.set_weights(self.prevModelWeights)
                self.model_target.set_weights(self.prevModelWeights)
                self.policyModel.set_weights(self.prevPolicyWeights)
                self.policyModel_dep.set_weights(self.prevPolicyWeights)
                self.prevSuccRewards = None # Always train next as by definition performance should not get worse.
            else:
                trained = True
                self.prevModelWeights = self.model_dep.get_weights()
                self.prevPolicyWeights = self.policyModel_dep.get_weights()
                self.prevSuccRewards = rewardsSinceLastUpdate
                # Train policy.
                states, labels = self.batch
                if len(labels) > 0:
                    self.hasTrained = True
                    optimizer = keras.optimizers.Adam(learning_rate=0.00025)
                    loss_fn = keras.losses.SparseCategoricalCrossentropy()

                    # Prepare the training dataset.
                    train_dataset = tf.data.Dataset.from_tensor_slices(self.batch)
                    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)

                    epochs = 8
                    for epoch in range(epochs):
                        print("\nStart of epoch %d" % (epoch,))

                        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                            with tf.GradientTape() as tape:
                                logits = self.policyModel(x_batch_train)

                                loss_value = loss_fn(y_batch_train, logits)

                            grads = tape.gradient(loss_value, self.policyModel.trainable_weights)

                            optimizer.apply_gradients(zip(grads, self.policyModel.trainable_weights))

            self.policyModel.save(model_save_dir + "_Policy_{}".format(self.numberOfTrainings))
            self.model.save(model_save_dir + "{}".format(self.numberOfTrainings))

            self.numberOfTrainings += 1
            # Update deployed networks used in lookahead.
            self.model_dep.set_weights(self.model.get_weights())
            self.policyModel_dep.set_weights(self.policyModel.get_weights())

            # Clear all collected train data.
            del self.rewards_history[:]
            del self.state_history[:]
            del self.state_next_history[:]
            del self.action_history[:]
            del self.done_history[:]
            assert len(self.rewards_history) == 0

            del self.batch[0][:]
            del self.batch[1][:]
            assert len(self.batch[1]) == 0

            return trained


    def predict_probs(self, state, useTrainingModel=False):
        if self.useRandomPolicy or not self.hasTrained or self.frame_count < self.minFramesBeforeUsing:
            return np.ones(self.numActions) / self.numActions  # Uniform prob distribution before network has been trained
        # Predict action probs
        # From environment state
        state_tensor = tf.convert_to_tensor(np.moveaxis(np.array(state), 0, -1))
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.policyModel_dep(state_tensor, training=False).numpy()
        return action_probs[0]  # Uniform prob distribution



    def value_func(self, state):
        if not self.hasTrained or self.frame_count < self.minFramesBeforeUsingcosttogo:
            return 0.0
        state_tensor = tf.convert_to_tensor(np.moveaxis(np.array(state), 0, -1))
        state_tensor = tf.expand_dims(state_tensor, 0)
        value = self.model_dep(state_tensor, training=False).numpy()
        return value[0][0]



    def state_action_pair_to_train(self, state, action, state_next, done, reward):
        # Transition sent from lookahead.
        self.frame_count += 1
        # Save actions and states in replay buffer

        if self.useValueFunction:
            self.action_history.append(action)
            self.state_history.append(np.moveaxis(np.array(state), 0, -1))
            self.state_next_history.append(np.moveaxis(np.array(state_next), 0, -1))
            self.done_history.append(done)
            self.rewards_history.append(reward)


            trainedOn = 0
            if self.frame_count % self.update_after_actions == 0 and len(self.done_history) > self.update_after_actions * 8:
                # Get indices of samples for replay buffers
                while trainedOn < self.update_after_actions * 8: # For every x transitions added to buffer train on 8 times number of transitions.
                    trainedOn += self.batch_size
                    indices = np.random.choice(range(len(self.done_history)), size=self.batch_size)

                    state_sample = np.array([self.state_history[i] for i in indices])
                    state_next_sample = np.array([self.state_next_history[i] for i in indices])
                    rewards_sample = [self.rewards_history[i] for i in indices]
                    done_sample = tf.convert_to_tensor(
                        [float(self.done_history[i]) for i in indices]
                    )

                    future_value = self.model_target(tf.convert_to_tensor(state_next_sample)).numpy()
                    updated_values = rewards_sample + self.gamma * tf.reduce_max(
                    future_value, axis=1)

                    # If terminal set the value to 0.
                    updated_values = updated_values * (1 - done_sample) #- done_sample
                    with tf.GradientTape() as tape:
                        values = self.model(tf.convert_to_tensor(state_sample))
                        values = tf.reduce_sum(values, axis=1)

                        loss = self.loss_function(tf.convert_to_tensor(updated_values), values)
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            if self.frame_count % self.update_target_network == 0:
                # Update the the target network with new weights.
                self.model_target.set_weights(self.model.get_weights())

            if len(self.rewards_history) > self.max_memory_length:
                del self.rewards_history[:1]
                del self.state_history[:1]
                del self.state_next_history[:1]
                del self.action_history[:1]
                del self.done_history[:1]

    def state_action_pair_to_classify(self, state, action):
        # Save actions and states from critical path to the buffer.
        self.batch[0].append(np.moveaxis(np.array(state), 0, -1))
        self.batch[1].append([action])

        if len(self.batch[1]) > self.max_memory_length:
            del self.batch[0][:1]
            del self.batch[1][:1]

class IWRollout(object):

    def __init__(self, **kwargs):
        self.sim_budget = int(kwargs.get('sim_budget', 100))
        self._gamma = float(kwargs.get('gamma', 1.00))
        self.rolloutHorizon = int(kwargs.get('rolloutHorizon', 10))
        self.cache = bool(kwargs.get('cache', False))
        self.noveltyType = kwargs.get('noveltyType', 'depth')
        self.stop_on_non_novel = bool(kwargs.get('stop_on_non_novel', True))
        self.backup_from_non_novel = bool(kwargs.get('backup_from_non_novel', True))
        self.num_actions = int(kwargs.get('num_actions', 4))
        self.model_save_dir = kwargs.get('model_save_dir', None)

        # Use critical path to learn policy network.
        self.useCritPath = kwargs.get('useCritPath', False)

        # Use critical path or all transitions to learn value function.
        self.useAll = kwargs.get('useAll', False)
        self.useValueFunction = kwargs.get('useValueFunction', False)

        self.root = None
        self.sim_calls = 0
        self.num_rollouts = 0
        self.start_depth = 0

        if self.useAll or self.useCritPath:
            self.base_policy = NN_V_Policy(self.num_actions, self.useValueFunction, self.useAll)
        else:
            self.base_policy = UniformPolicy(self.num_actions)

        if self.noveltyType == "classic":
            self.classicNoveltyTable = np.array([])
            self.preComputedArrayUsedForAtariClassicTable = np.arange(0, 1806336, step=256)
        else:
            self.depthNoveltyTable = np.empty(1806336)
            self.preComputedArrayUsedForAtariDepthTable = np.arange(0, 1806336, step=256)
            self.depthNoveltyTable[:] = self.rolloutHorizon + 2 # Initialise depth table.

    def resetEp(self, rewards):
        if self.root is not None:
            rootTemp = self.root
            self.root = None
            self.clearGraphMem(rootTemp)
        trained = self.base_policy.train_with_batch(self.model_save_dir, rewards) # Check if time to accept/reject network updates and train a new policy network.
        return trained

    def checkAndUpdateNoveltyTableDepth(self, node, depth):  # Depth novelty definition.
        enteriesToAdd = (np.array(node.state[-1]).flatten() + self.preComputedArrayUsedForAtariDepthTable)
        enteriesToAdd = enteriesToAdd.astype('int32')
        # Check for better
        currentVals = np.take(self.depthNoveltyTable, enteriesToAdd)
        sumBefore = np.sum(currentVals)
        np.putmask(currentVals, currentVals > depth, depth)
        sumAfter = np.sum(currentVals)
        if sumAfter < sumBefore:
            node.novel = True
            np.put(self.depthNoveltyTable, enteriesToAdd, currentVals)
        elif hasattr(node, 'novel') and node.novel:
        # Check for same if already classified as novel
            result = currentVals == depth
            if np.sum(result) == 0:
                node.novel = False
        else:
            node.novel = False
        return

    def checkAndUpdateNoveltyTableClassic(self, node): # Classic novelty definition.
        if hasattr(node, 'novel'):
            return
        enteriesToAdd = np.array(node.state[-1]).flatten() + self.preComputedArrayUsedForAtariClassicTable
        lenBefore = len(self.classicNoveltyTable)

        self.classicNoveltyTable = np.append(self.classicNoveltyTable,
                                             np.setdiff1d(enteriesToAdd, self.classicNoveltyTable, assume_unique=True))

        lenafter = len(self.classicNoveltyTable)

        if lenBefore < lenafter:
            node.novel = True
        else:
            node.novel = False
        return

    def checkAndUpdateNoveltyTable(self, node, depth, accumReward):  # Only deals with states with 1 state var at
        if self.noveltyType == "depth":
            return self.checkAndUpdateNoveltyTableDepth(node, depth)
        elif self.noveltyType == "classic":
            return self.checkAndUpdateNoveltyTableClassic(node)
        else:
            assert False

    def select_root(self, state):
        if self.root is None:
            self.start_depth = 0
            self.root = OR_Node(state, False, 0)
            return True
        else:
            found_new = False
            self.start_depth += 1  # select new root node from next depth
            checkAll = False
            for i in range(2):
                for act, child in self.root.children.items():
                    if not checkAll and self.selected_action != act:
                        continue
                    elif checkAll and self.selected_action == act:
                        continue

                    for succ, r in child.children:
                        if not found_new and np.array_equal(state, succ.state):
                            new_root = succ
                            found_new = True
                        else:
                            self.clearGraphMem(succ)
                    del child
                checkAll = True

            if not found_new:  # If no state at level 1 then do not cache
                assert False # Should of found new node
            else:
                del self.root
                self.root = new_root
                self.removeNoveltyMarks(self.root)
                assert not self.root.terminal
                return False

    def get_action(self, env, state):
        # Reinit novelty table.
        if self.noveltyType == "classic":
            self.classicNoveltyTable = np.array([])
        else:
            self.depthNoveltyTable[:] = self.rolloutHorizon + 2

        self.init_sim_calls = self.sim_calls
        new_root = self.select_root(state)
        accumReward = 0
        if not self.root.stale:
            self.checkAndUpdateNoveltyTable(self.root, 0, accumReward)
            self.root.cloned_state = env.unwrapped.clone_full_state()
            self.root.currentFramesHist = copy.deepcopy(env.frames)
        else:
            self.root.novel = True # Consider cached nodes novel.

        rollouts = 0
        while self.sim_calls - self.init_sim_calls < self.sim_budget and (not hasattr(self.root, 'solved') or not
        self.root.solved):
            rollouts += 1
            trace = self.rollout(env)
            self.num_rollouts += 1

        self.backup(self._gamma)
        action, expected, reward, done, next_state = self.select_best(self.root, env)
        self.selected_action = action
        return action, expected, reward, done, next_state

    def clearGraphMem(self, root):
        open = deque()
        open.append(root)
        # Delete reference from parent
        if root.parent is not None:
            if isinstance(root, OR_Node):
                newChildren = set()
                for succ, R in root.parent.children:
                    if succ != root:
                        newChildren.add((succ, R))
                root.parent._children = newChildren
            elif isinstance(root, AND_Node):
                newChildren = {}
                for indx, succ in root.parent.children.items():
                    if succ != root:
                        newChildren[succ.action] = succ
                root.parent._children = newChildren
        while len(open) > 0:
            n = open.pop()  # top of the stack
            if isinstance(n, OR_Node):
                for act, child in n.children.items():
                    child._parent = None
                    open.append(child)
                n._children = {}
                del n
                continue
            elif isinstance(n, AND_Node):
                for succ, r in n.children:
                    succ._parent = None
                    open.append(succ)
                n._children = set()
                del n
                continue
            else:
                assert False

    def removeNoveltyMarks(self, root):
        open = deque()
        open.append(root)

        while len(open) > 0:
            n = open.pop()  # top of the stack
            if isinstance(n, OR_Node):
                for act, child in n.children.items():
                    open.append(child)
                if hasattr(n, "novel"):
                    del n.novel
                if hasattr(n, "solved"):
                    del n.solved
                continue
            elif isinstance(n, AND_Node):
                for succ, r in n.children:
                    open.append(succ)
                if hasattr(n, "solved"):
                    del n.solved
                continue
            else:
                assert False

    def rollout(self, env):
        depth = 0
        accReward = 0
        done = False
        currentOR = self.root
        if not self.cache or currentOR.num_visits < 1:
            currentOR.increment_visits()

        trace = [currentOR]
        Ors_to_solve_from = []
        self.needToRestore = True
        while self.sim_calls - self.init_sim_calls < self.sim_budget and depth < self.rolloutHorizon and not done and (currentOR.novel or not self.stop_on_non_novel) and (not hasattr(currentOR, 'solved') or not
        currentOR.solved):

            depth += 1

            action = self.select_action(env, currentOR, self.cache)

            if self.cache:
                try:
                    executedAND = currentOR.children[action]
                    assert len(executedAND.children) == 1
                    for currentOR, reward in executedAND.children:
                        break
                    accReward += reward
                    self.needToRestore = True
                    done = currentOR.terminal
                    trace.append(executedAND)
                    trace.append(currentOR)
                    if not currentOR.stale:
                        self.checkAndUpdateNoveltyTable(currentOR, depth, accReward) #Keep any successful runs
                    else:
                        currentOR.novel = True  # Treat cached nodes from previous search as novel

                    if not currentOR.novel and self.stop_on_non_novel:
                        Ors_to_solve_from.append(currentOR)
                        self.updateSolvedLabels(currentOR, env)
                    continue
                except KeyError:
                    assert True
            if self.needToRestore:
                assert hasattr(currentOR, 'cloned_state')
                env.unwrapped.restore_full_state(currentOR.cloned_state)
                env.frames = copy.deepcopy(currentOR.currentFramesHist)
                self.needToRestore = False

            state, reward, done, info = env.step(action)

            if self.useAll:
                self.base_policy.state_action_pair_to_train(currentOR.state, action, state, done, reward)
            accReward += reward
            self.sim_calls += 1
            currentOR, previousAND = self.updateLookaheadWithTransition(currentOR, action, state, reward, done,
                                                                        depth + self.start_depth)
            currentOR.cloned_state = env.unwrapped.clone_full_state()
            currentOR.currentFramesHist = copy.deepcopy(env.frames)
            trace.append(previousAND)
            trace.append(currentOR)
            self.checkAndUpdateNoveltyTable(currentOR, depth, accReward)
            if not currentOR.novel and self.stop_on_non_novel:
                Ors_to_solve_from.append(currentOR)
                self.updateSolvedLabels(currentOR, env)

        if currentOR.novel and (depth == self.rolloutHorizon or done) and (not hasattr(currentOR, 'solved') or not currentOR.solved):
            self.updateSolvedLabels(currentOR, env)
        return trace

    def updateSolvedLabels(self, solvedOR, env):
        open = deque()
        open.append(solvedOR)
        while len(open) > 0:
            n = open.pop()  # top of the stack

            if isinstance(n, OR_Node):
                n.solved = True
                if n.novel:
                    if len(n.children) > 0 and len(n.children) < env.action_space.n:  # If has no children is
                        # solved, but if has children has to have all children.
                        n.solved = False
                        continue
                    for act, child in n.children.items():
                        if not hasattr(child, 'solved') or not child.solved:
                            # Not solved.
                            n.solved = False
                            continue
                if n is not self.root:
                    open.append(n.parent)
                continue
            elif isinstance(n, AND_Node):
                n.solved = True
                assert len(n.children) == 1
                for child, reward in n.children:
                    if not hasattr(child, 'solved') or not child.solved:
                        # Not solved.
                        n.solved = False
                        continue
                open.append(n.parent)
                continue
            else:
                assert False

    def updateLookaheadWithTransition(self, currentOR, action, state, reward, done, depth):
        # Find or expand out AND from current OR node.
        try:
            executedAND = currentOR.children[action]
        except KeyError:
            executedAND = AND_Node(action, currentOR)
        executedAND.increment_visits()
        # Check for resulting OR node or add.
        nextOR = None
        for succ, R in executedAND.children:
            if (state == succ.state).all() and done == succ.terminal and R == reward:
                nextOR = succ
        if nextOR is None:
            nextOR = OR_Node(state, done, depth, executedAND)
            executedAND.add_child(nextOR, reward)
        nextOR.increment_visits()
        return nextOR, executedAND

    def select_action(self, env, currentOR, caching=False):
        try:
            base_policy_probs = currentOR.base_policy
        except:
            base_policy_probs = self.base_policy.predict_probs(currentOR.state)
            currentOR.base_policy = base_policy_probs
        if caching:
            unsolvedActions = []
            actionWeights = []
            for action in range(env.action_space.n):
                if action not in currentOR.children.keys() or not hasattr(currentOR.children[action], 'solved') or not \
                        currentOR.children[action].solved:
                    unsolvedActions.append(action)
                    actionWeights.append(base_policy_probs[action])

            assert len(unsolvedActions) > 0
        else:
            unsolvedActions = range(env.action_space.n )
            actionWeights = base_policy_probs
        selectedAction = random.choices(unsolvedActions, weights=actionWeights)[0]
        return selectedAction


    def backup(self, gamma):
        open = deque()
        backed_up_OR = set()
        backed_up_AND = set()
        open.append(self.root)
        while len(open) > 0:
            n = open[-1]  # top of the stack
            if isinstance(n, OR_Node):
                n.stale = True
                if n.terminal or n.depth == self.rolloutHorizon + self.start_depth or len(n.children) == 0:
                    if n.terminal:
                        n.V = 0.0
                    elif not hasattr(n, 'V'):
                        if self.useValueFunction:
                            n.V = self.base_policy.value_func(n.state)
                        else:
                            n.V = 0.0
                    backed_up_OR.add(n)
                    open.pop()
                    continue
                assert len(n.children) > 0 or n is self.root
                all = True
                for act, child in n.children.items():
                    if not child in backed_up_AND:
                        all = False
                        open.append(child)
                        break
                if not all:
                    continue
                best_child_value = float('-inf')
                for act, child in n.children.items():
                    if child.Q > best_child_value:
                        best_child_value = child.Q
                n.V = best_child_value
                backed_up_OR.add(n)
                open.pop()
                continue
            elif isinstance(n, AND_Node):
                all = True
                for succ, r in n.children:
                    if not succ in backed_up_OR:
                        all = False
                        open.append(succ)
                        break
                if not all: continue
                numberOfVisitsToChild = 0
                n.Q = 0
                assert len(n.children) > 0
                for succ, r in n.children:
                    q = r + gamma * succ.V
                    numberOfVisitsToChild += succ.num_visits
                    assert succ.num_visits > 0
                    n.Q += (q * succ.num_visits) / n.num_visits
                assert n.num_visits == numberOfVisitsToChild
                backed_up_AND.add(n)
                open.pop()
                continue
            else:
                assert False

    def select_best(self, n: OR_Node, env):
        best_Q = None
        candidates = []
        state_nexts = []
        dones = []
        rewards = []
        if len(n.children) == 0:
            best_action = self.select_action(env, None, caching=False) # Just follow base policy if current node has no children.
            expected = None
            print("no valid trajectories")
        else:
            expected = np.ones(env.action_space.n) * np.NaN
            for act, child in n.children.items():
                expected[act] = child.Q
                if best_Q is None or child.Q > best_Q:
                    candidates = [act]
                    assert len(child.children) == 1 # Only implemented for deterministic problems.
                    for nextState, reward in child.children:
                        break
                    rewards = [reward]
                    dones = [nextState.terminal]
                    state_nexts = [nextState.state]
                    best_Q = child.Q
                elif abs(child.Q - best_Q) < 0.0000001:  # For random breaking of ties.
                    for nextState, reward in child.children:
                        break
                    candidates.append(act)
                    rewards.append(reward)
                    dones.append(nextState.terminal)
                    state_nexts.append(nextState.state)

            best_action = random.choice(candidates)
            for i in range(len(candidates)): # Train on all transitions which have the same Qs.
                if candidates[i] == best_action:
                    if self.useCritPath:
                        self.base_policy.state_action_pair_to_train(n.state, candidates[i], state_nexts[i], dones[i], rewards[i])
                    done, reward, next_state = dones[i], rewards[i], state_nexts[i]
                    break
            if self.useCritPath or self.useAll:
                self.base_policy.state_action_pair_to_classify(n.state, best_action)

        return best_action, expected, reward, done, next_state
