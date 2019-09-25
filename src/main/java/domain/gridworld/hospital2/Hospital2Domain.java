package domain.gridworld.hospital2;

import client.Timeout;
import domain.Domain;
import domain.ParseException;
import domain.gridworld.hospital2.state.State;
import domain.gridworld.hospital2.state.StaticState;
import domain.gridworld.hospital2.state.parser.StateParser;

import java.awt.*;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Hospital2Domain implements Domain {
    private boolean allowDiscardingPastStates;

    private StaticState staticState;
    private List<State> states;

    private String level;
    private Hospital2Runner runner;

    public Hospital2Domain(Path domainFile, boolean isReplay) {
        StateParser stateParser = new StateParser(domainFile, isReplay);
        try {
            stateParser.parse();
        } catch (IOException | ParseException e) {
            e.printStackTrace();
        }
        this.staticState = stateParser.getStaticState();
        this.states = Collections.synchronizedList(new ArrayList<>());
        this.states.add(stateParser.getState());
        this.level = stateParser.getLevel();
    }

    byte getNumAgents() {
        return this.staticState.getNumAgents();
    }

    /**
     * Execute a joint action.
     * Returns a boolean array with success for each agent.
     */
    boolean[] execute(Action[] jointAction, long actionTime) {
        State state = this.getLatestState();

        // Create new state with applicable and non-conflicting actions.
        State newState = state.apply(jointAction);
        newState.setStateTime(actionTime);

        if (this.allowDiscardingPastStates) {
            this.states.clear();
        }
        this.states.add(newState);

        return state.getApplicable();
    }
    @Override
    public void runProtocol(Timeout timeout, long timeoutNS, BufferedInputStream clientIn, BufferedOutputStream clientOut, OutputStream logOut) {
        this.runner = new Hospital2Runner(timeout, timeoutNS, this.level, clientIn, clientOut, logOut, this);
        this.runner.run();
    }

    @Override
    public void allowDiscardingPastStates() {
        this.allowDiscardingPastStates = true;
    }

    @Override
    public String getLevelName() {
        return this.staticState.getLevelName();
    }

    @Override
    public String getClientName() {
        return this.runner.getClientName();
    }

    State getLatestState() {
        return this.states.get(this.states.size() - 1);
    }

    boolean isSolved(State state) {
        return this.staticState.isSolved(state);
    }

    @Override
    public String[] getStatus() {
        State state = this.getLatestState();

        String solved = this.isSolved(state) ? "Yes" : "No";

        String[] status = new String[3];
        status[0] = String.format("Level solved: %s.", solved);
        status[1] = String.format("Actions used: %d.", this.getNumAgents());
        status[2] = String.format("Last action time: %.3f seconds.", state.getStateTime() / 1_000_000_000d);

        return status;
    }

    @Override
    public int getNumStates() {
        return this.states.size();
    }

    @Override
    public long getStateTime(int stateID) {
        return this.states.get(stateID).getStateTime();
    }

    @Override
    public void renderDomainBackground(Graphics2D g, int width, int height) {

    }

    @Override
    public void renderStateBackground(Graphics2D g, int stateID) {

    }

    @Override
    public void renderStateTransition(Graphics2D g, int stateID, double interpolation) {

    }
}
