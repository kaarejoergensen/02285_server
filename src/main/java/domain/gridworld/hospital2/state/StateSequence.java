package domain.gridworld.hospital2.state;

import domain.ParseException;
import domain.gridworld.hospital2.state.parser.StateParser;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class StateSequence {
    private Logger serverLogger = LogManager.getLogger("server");

    private boolean allowDiscardingPastStates;

    private StaticState staticState;

    private List<State> states;

    public StateSequence(Path domainFile, boolean isReplay) {
        StateParser stateParser = new StateParser(domainFile, isReplay);
        try {
            stateParser.parse();
        } catch (IOException | ParseException e) {
            e.printStackTrace();
        }
        this.staticState = stateParser.getStaticState();
        this.states = Collections.synchronizedList(new ArrayList<>());
        this.states.add(stateParser.getState());

        System.out.println();
    }

    void allowDiscardingPastStates() {
        this.allowDiscardingPastStates = true;
    }

    String getLevelName() {
        return this.staticState.getLevelName();
    }

    int getNumStates() {
        return this.states.size();
    }

    long getStateTime(int state) {
        return this.states.get(state).getStateTime();
    }
}
