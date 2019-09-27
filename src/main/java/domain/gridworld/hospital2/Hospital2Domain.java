package domain.gridworld.hospital2;

import client.Timeout;
import domain.Domain;
import domain.ParseException;
import domain.gridworld.hospital2.runner.Hospital2Runner;
import domain.gridworld.hospital2.runner.RunException;
import domain.gridworld.hospital2.state.State;
import domain.gridworld.hospital2.state.objects.StaticState;
import domain.gridworld.hospital2.state.objects.ui.GUIState;
import domain.gridworld.hospital2.state.parser.StateParser;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.awt.*;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Path;

public class Hospital2Domain implements Domain {
    private static Logger clientLogger = LogManager.getLogger("client");
    private Hospital2Runner runner;
    private GUIState guiState;

    public Hospital2Domain(Path domainFile, boolean isReplay) throws IOException, ParseException {
        StateParser stateParser = new StateParser(domainFile, isReplay);
        stateParser.parse();
        State state = stateParser.getState();
        StaticState staticState = stateParser.getStaticState();
        this.runner = new Hospital2Runner(stateParser.getLevel(), state, staticState);

        if (isReplay) {
            runner.executeReplay(stateParser.getActions(), stateParser.getActionTimes(), stateParser.isLogSolved());
        }

        this.guiState = new GUIState(staticState.getMap(), state.getAgents(), state.getBoxes(), staticState.getAllGoals());
    }

    @Override
    public void runProtocol(Timeout timeout, long timeoutNS, BufferedInputStream clientIn, BufferedOutputStream clientOut, OutputStream logOut) {
        try {
            this.runner.run(timeout, timeoutNS, clientIn, clientOut, logOut);
        } catch (RunException e) {
            clientLogger.error(e.getMessage());
        }
    }

    @Override
    public void allowDiscardingPastStates() {
        this.runner.allowDiscardingPastStates();
    }

    @Override
    public String getLevelName() {
        return this.runner.getLevelName();
    }

    @Override
    public String getClientName() {
        return this.runner.getClientName();
    }

    @Override
    public String[] getStatus() {
        return runner.getStatus();
    }

    @Override
    public int getNumStates() {
        return this.runner.getNumStates();
    }

    @Override
    public long getStateTime(int stateID) {
        return this.runner.getStateTime(stateID);
    }

    @Override
    public void renderDomainBackground(Graphics2D g, int width, int height) {
        short numCols = this.runner.getStaticState().getNumCols();
        short numRows = this.runner.getStaticState().getNumRows();
        this.guiState.drawBackground(g, width, numCols, height, numRows);
    }

    @Override
    public void renderStateBackground(Graphics2D g, int stateID) {
        StaticState staticState = this.runner.getStaticState();
        State currentState = this.runner.getState(stateID);
        State nextState = this.runner.getState(stateID + 1);
        guiState.drawStateBackground(g, staticState, currentState, nextState);
    }

    @Override
    public void renderStateTransition(Graphics2D g, int stateID, double interpolation) {

    }
}
