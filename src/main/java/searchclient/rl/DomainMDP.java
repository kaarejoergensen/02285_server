package searchclient.rl;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.nd4j.linalg.api.ops.impl.scalar.Step;
import searchclient.State;
import shared.Action;

public class DomainMDP implements MDP<State, Integer, DiscreteSpace> {
    private State state;

    private Action[] actions;

    private DiscreteSpace actionSpace;
    private ObservationSpace<State> observationSpace;

    public DomainMDP(State state) {
        this.state = state;

        this.actions = Action.values();
        actionSpace = new DiscreteSpace(actions.length);

        int[] shape = {state.walls.length * state.walls[0].length * 6};
        observationSpace = new ArrayObservationSpace<>(shape);
    }

    @Override
    public ObservationSpace<State> getObservationSpace() {
        return observationSpace;
    }

    @Override
    public DiscreteSpace getActionSpace() {
        return actionSpace;
    }

    @Override
    public State reset() {
        this.state = this.state.reset();
        return this.state;
    }

    @Override
    public void close() {
    }

    @Override
    public StepReply<State> step(Integer actionIndex) {
        Action action = this.actions[actionIndex];
        if (this.state.isApplicable(0, action)) {
            this.state = new State(this.state, new Action[] {action});
            return new StepReply<>(this.state, this.state.reward(), this.state.isGoalState(), null);
        }
        return new StepReply<>(this.state, 0, this.state.isGoalState(), null);
    }

    @Override
    public boolean isDone() {
        return this.state.isGoalState();
    }

    @Override
    public MDP<State, Integer, DiscreteSpace> newInstance() {
        return new DomainMDP(this.state);
    }
}
