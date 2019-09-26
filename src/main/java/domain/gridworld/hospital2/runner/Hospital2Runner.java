package domain.gridworld.hospital2.runner;

import client.Timeout;
import domain.ParseException;
import domain.gridworld.hospital2.Action;
import domain.gridworld.hospital2.state.State;
import domain.gridworld.hospital2.state.objects.StaticState;
import lombok.Getter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class Hospital2Runner {
    private Logger clientLogger = LogManager.getLogger("client");

    private String level;

    @Getter private String clientName;

    private boolean allowDiscardingPastStates;

    private List<State> states;
    private StaticState staticState;

    public Hospital2Runner(String level, State initialState, StaticState staticState) {
        this.level = level;

        this.states = Collections.synchronizedList(new ArrayList<>());
        this.states.add(initialState);
        this.staticState = staticState;
    }

    public void allowDiscardingPastStates() {
        this.allowDiscardingPastStates = true;
    }

    public String getLevelName() {
        return this.staticState.getLevelName();
    }

    public int getNumStates() {
        return this.states.size();
    }

    public long getStateTime(int stateID) {
        return this.states.get(stateID).getStateTime();
    }

    public String[] getStatus() {
        State state = this.getLatestState();

        String solved = this.isSolved(state) ? "Yes" : "No";

        String[] status = new String[3];
        status[0] = String.format("Level solved: %s.", solved);
        status[1] = String.format("Actions used: %d.", this.getNumActions());
        status[2] = String.format("Last action time: %.3f seconds.", state.getStateTime() / 1_000_000_000d);

        return status;
    }

    public void run(Timeout timeout, long timeoutNS,
                    BufferedInputStream clientIn, BufferedOutputStream clientOut, OutputStream logOut) throws RunException {

        BufferedReader clientReader = new BufferedReader(new InputStreamReader(clientIn, StandardCharsets.US_ASCII.newDecoder()));
        BufferedWriter clientWriter = new BufferedWriter(new OutputStreamWriter(clientOut, StandardCharsets.US_ASCII.newEncoder()));
        BufferedWriter logWriter = new BufferedWriter(new OutputStreamWriter(logOut, StandardCharsets.US_ASCII.newEncoder()));

        long startNS = System.nanoTime();
        timeout.reset(startNS, TimeUnit.SECONDS.toNanos(10));


        this.clientName = RunnerHelper.readClientName(clientReader, timeout, startNS, timeoutNS);
        RunnerHelper.writeLevelToClientAndLog(clientWriter, logWriter, level, clientName, timeout);
        RunnerHelper.exchangeActions(clientReader, clientWriter, logWriter, this.staticState.getNumAgents(), timeout,
                startNS, states, allowDiscardingPastStates);
        RunnerHelper.writeLogSummary(logWriter, this.isSolved(this.getLatestState()),
                this.getNumStates(), this.getLatestState().getStateTime());

        clientLogger.debug("Protocol finished.");
    }

    public void executeReplay(List<Action[]> jointActions, List<Long> actionTimes, boolean logSolved) throws ParseException {
        for (int i = 0; i < jointActions.size(); i++) {
            Action[] jointAction = jointActions.get(i);
            Long actionTime = actionTimes.get(i);
            RunnerHelper.execute(jointAction, actionTime, this.states, this.allowDiscardingPastStates);
        }
        if (this.staticState.isSolved(this.getLatestState()) != logSolved) {
            if (logSolved) {
                throw new ParseException("Log summary claims level is solved, but the actions don't solve the level.");
            } else {
                throw new ParseException("Log summary claims level is not solved, but the actions solve the level.");
            }
        }
    }

    private int getNumActions() {
        return this.getNumStates() - 1;
    }

    private State getLatestState() {
        return this.states.get(this.states.size() - 1);
    }

    private boolean isSolved(State state) {
        return this.staticState.isSolved(state);
    }
}
