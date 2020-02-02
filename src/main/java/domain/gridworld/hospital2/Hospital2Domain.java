package domain.gridworld.hospital2;

import client.Timeout;
import domain.Domain;
import domain.ParseException;
import domain.gridworld.hospital2.runner.Hospital2Runner;
import domain.gridworld.hospital2.runner.RunException;
import domain.gridworld.hospital2.parser.StateParser;
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
    private static Logger serverLogger = LogManager.getLogger("server");

    private Hospital2Runner runner;
    private GUIHandler guiHandler;

    public Hospital2Domain(Path domainFile, boolean isReplay) throws IOException, ParseException {
        var stateParser = new StateParser(domainFile, isReplay);
        stateParser.parse();
        var staticState = stateParser.getStaticState();
        this.runner = new Hospital2Runner(stateParser.getLevel(), stateParser.getState(), staticState);

        if (isReplay) {
            runner.executeReplay(stateParser.getActions(), stateParser.getActionTimes(), stateParser.isLogSolved());
        }

        this.guiHandler = new GUIHandler(staticState);
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

    public double getNumActions() {
        return runner.getNumActions();
    }

    public double getTime() {
        return runner.getTimeTaken();
    }

    public double getMaxMemoryUsage() {
        return runner.getMaxMemoryUsage();
    }

    public boolean isSolved() {
        return runner.isLatestStateSolved();
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
        var currentState = this.runner.getState(0);
        this.guiHandler.drawBackground(g, width, height, currentState);
    }

    @Override
    public void renderStateBackground(Graphics2D g, int stateID) {
        var currentState = this.runner.getState(stateID);
        guiHandler.drawStateBackground(g, currentState);
    }

    @Override
    public void renderStateTransition(Graphics2D g, int stateID, double interpolation) {
        if (interpolation < 0.0 || interpolation >= 1.0) {
            serverLogger.error("Bad interpolation: " + interpolation);
            return;
        }
        var currentState = this.runner.getState(stateID);
        var nextState = interpolation == 0.0 ? currentState : this.runner.getState(stateID + 1);
        guiHandler.drawStateTransition(g, currentState, nextState, interpolation);
    }
}
