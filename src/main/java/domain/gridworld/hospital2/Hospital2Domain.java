package domain.gridworld.hospital2;

import client.Timeout;
import domain.Domain;
import domain.gridworld.hospital2.state.StateSequence;

import java.awt.*;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.OutputStream;
import java.nio.file.Path;

public class Hospital2Domain implements Domain {
    public Hospital2Domain(Path domainFile, boolean isReplay) {
        StateSequence stateSequence = new StateSequence(domainFile, isReplay);
    }

    @Override
    public void runProtocol(Timeout timeout, long timeoutNS, BufferedInputStream clientIn, BufferedOutputStream clientOut, OutputStream logOut) {

    }

    @Override
    public void allowDiscardingPastStates() {

    }

    @Override
    public String getLevelName() {
        return null;
    }

    @Override
    public String getClientName() {
        return null;
    }

    @Override
    public String[] getStatus() {
        return new String[0];
    }

    @Override
    public int getNumStates() {
        return 0;
    }

    @Override
    public long getStateTime(int stateID) {
        return 0;
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
