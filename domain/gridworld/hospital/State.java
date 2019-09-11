package domain.gridworld.hospital;

import java.util.Arrays;

/**
 * Compact data class for storing non-static state information.
 * Every instance belongs to a StateSequence, which tracks the static state information.
 */
class State
{
    /**
     * Box locations.
     * Indexed by box id (0 .. numBoxes-1).
     */
    short[] boxRows;
    short[] boxCols;

    /**
     * Agent locations.
     * Indexed by agent id (0 .. numAgents-1).
     */
    short[] agentRows;
    short[] agentCols;

    State(short[] boxRows, short[] boxCols, short[] agentRows, short[] agentCols)
    {
        this.boxRows = boxRows;
        this.boxCols = boxCols;
        this.agentRows = agentRows;
        this.agentCols = agentCols;
    }

    State(State copy)
    {
        this.boxRows = Arrays.copyOf(copy.boxRows, copy.boxRows.length);
        this.boxCols = Arrays.copyOf(copy.boxCols, copy.boxCols.length);
        this.agentRows = Arrays.copyOf(copy.agentRows, copy.agentRows.length);
        this.agentCols = Arrays.copyOf(copy.agentCols, copy.agentCols.length);
    }
}
