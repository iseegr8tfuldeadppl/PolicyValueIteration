import gym
import time
import numpy


def evaluatePolicyOnce(env, policy, render=False, gamma=1.0, eps=1e-20):

    # Step 1: - Intialize total reward value.
    #         - StepIndex to help execute
    #           the decaying test formula, gamma goes higher in the power
    #           for each time we get farther in greedy actions.
    #         - Reset environment at a random starting point.
    totalReward = 0
    stepIndex = 0
    start = env.reset()

    # Step 2: Start greedily taking actions according to current policy till done
    while True:

        # Step 3: Render?
        if render:
            env.render()

        # Step 4: extract the action
        action = policy[start]

        # Step 5: Greedily step according to that policy
        start, reward, done, _ = env.step(action)

        # Step 6: Apply decaying formula for evaluating the policy
        totalReward += reward * gamma ** stepIndex
        stepIndex += 1

        # Step 7: Break if done
        if done:
            break
    return totalReward

def evaluatePolicy(env, policy, gamma=1.0, evaluationIterations=100, render=False, eps=1e-20):

    # Step 1: Extract multiple  scores using the policy
    scores = [
    evaluatePolicyOnce(env=env, policy=policy, gamma=gamma, render=render, eps=eps)
        for _ in range(evaluationIterations)
    ]

    # Step 2: Return the average of those scores
    return numpy.mean(scores)

def calculatePolicy(env, v, gamma=1.0):

    # Step 1: Initialize the policy holder
    policy = numpy.zeros(env.nS)

    # Step 2: Loop through all states
    for state in range(env.nS):

        # Step 3: Reset Q
        Q = [] #Q = numpy.zeros(env.nA)

        # Step 4: Loop through all actions
        for action in range(env.nA):

            # Step 5: Reset all outcomes of this particular action
            all_outcomes_of_that_action = []

            # Step 6: Loop through all outcomes of performing that action
            for probability, nextstate, reward, done in env.P[state][action]:
                all_outcomes_of_that_action.append(probability * (reward + gamma * v[nextstate]) )

            # Step 7: Sum all_outcomes_of_that_action into a score on a list in Q
            Q.append(numpy.sum(all_outcomes_of_that_action))

        # Step 8: Extract policy = choose best action for that state
        policy[state] = numpy.argmax(Q)

    # Step 9: Return policy
    return policy

def valueIteration(env, gamma=1.0, improvementIterations=10000, eps=1e-20):

    # Step 1: Initialize Value-function
    value = numpy.zeros(env.nS)

    # Step 2: Start Iteration
    for i in range(improvementIterations):

        # Step 3: Save a Reference to the current value function
        previous_value = numpy.copy(value)

        # Step 4: Loop through all states
        for state in range(env.nS):

            # Step 5: Reset Q function on each state
            Q = []

            # Step 6: Loop through all actions
            for action in range(env.nA):

                # Step 7: Reset all outcomes of this particular action
                all_outcomes_of_that_action = []

                # Step 8: Loop through all outcomes of performing that action
                for probability, nextstate, reward, done in env.P[state][action]:
                    all_outcomes_of_that_action.append(probability * (reward + gamma * previous_value[nextstate]) )

                # Step 9: Sum all_outcomes_of_that_action into a score on a list in Q
                Q.append(numpy.sum(all_outcomes_of_that_action))

            # Step 10: Choose the best action for that state
            value[state] = numpy.max(Q)

        # Step 11: Check if improvment is smaller than epsilon, if so break
        if(numpy.sum(numpy.fabs(previous_value - value)) <= eps):
            #print('Value-Iteration converged at # %d.'%(i+1))
            return value

    # Step 12: Handle the case of non-convergence
    print("Iterated over %d Iterations and couldn't converge"%(improvementIterations))
    return value

def get_value_function(env, policy, eps=1e-20):

    # Step 1: Initialize random value-function
    value = numpy.zeros(env.nS)

    while True:

        # Step 2: Keep Reference of older value-function to use in formula
        previous_value = numpy.copy(value)

        # Step 3: Start looping through all states
        for state in range(policy.size):
            all_outcomes_of_that_action = []

            # Step 4: Extract the action from policy
            action = policy[state]

            # Step 5: Get all outcomes of that action
            for probability, next_state, reward, is_done in env.P[state][action]:
                all_outcomes_of_that_action.append(probability * (reward + previous_value[next_state]))

            # Step 6: Extract the new value
            value[state] = numpy.sum(all_outcomes_of_that_action)

        # Step 7: if no convergence then break
        if numpy.sum(numpy.fabs(previous_value - value)) <= eps:
            return value

    return value

def policyIteration(env, gamma=1.0, improvementIterations=1000, eps=1e-20):

    # Step 1: Declare a random policy
    policy = numpy.random.choice(env.nA, size=(env.nS))

    # Step 2: Start iterations
    for i in range(improvementIterations):

        # Step 3: Get the value-function
        oldPolicyValue = get_value_function(env=env, policy=policy, eps=eps)

        # Step 4: improve value-function until it converges then get policy
        newPolicy = calculatePolicy(env=env, v=oldPolicyValue, gamma=gamma)

        # Step 5: If policy doesn't change then stop looping
        if(numpy.all(policy == newPolicy)):
            #print('Policy Iteration converged at %d.'%(i+1))
            return policy
        policy = newPolicy

    # Step 6: Handle the case of non-convergence
    print("Iterated over %d Iterations and couldn't converge"%(improvementIterations))
    return policy

def ValueIterator(env, gamma=1.0, improvementIterations=1000, evaluationIterations=100, eps=1e-20):
    optimalValue = valueIteration(env=env, gamma=gamma, improvementIterations=improvementIterations, eps=eps)
    startTime = time.time()
    policy = calculatePolicy(env=env, v=optimalValue, gamma=gamma)
    policy_score = evaluatePolicy(env=env, policy=policy, render=False, gamma=gamma, evaluationIterations=evaluationIterations, eps=eps)
    endTime = time.time()
    print('Best score = %0.2f. Time taken = %4.4f seconds'%(numpy.max(policy_score), endTime - startTime))

def PolicyIterator(env, gamma=1.0, improvementIterations=1000, eps=1e-20, evaluationIterations=100):
    optimalPolicy = policyIteration(env=env, gamma=gamma, improvementIterations=improvementIterations, eps=eps)
    startTime = time.time()
    policy_score = evaluatePolicy(env=env, policy=optimalPolicy, render=False, gamma=gamma, evaluationIterations=evaluationIterations, eps=eps)
    endTime = time.time()
    print('Best score = %0.2f. Time taken = %4.4f seconds'%(numpy.max(policy_score), endTime - startTime))
