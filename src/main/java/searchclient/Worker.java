package searchclient;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.apache.commons.lang3.tuple.Triple;
import org.javatuples.Triplet;
import shared.Action;

import java.util.HashSet;
import java.util.concurrent.Callable;

@AllArgsConstructor
public class Worker implements Callable<WorkerReturn> {
    private int threadId;
    private State initialState;

    @Override
    public WorkerReturn call() throws Exception {
        System.err.println("Thread " + this.threadId + " starting");

        Frontier frontier = new FrontierBFS();
        HashSet<State> explored = new HashSet<>();
        frontier.add(initialState);
        while (true) {
            if (frontier.isEmpty()) {
                return new WorkerReturn(null, explored.size(), frontier.size(), initialState);
            }

            State leafState = frontier.pop();

            if (leafState.isGoalState()) {
                return new WorkerReturn(leafState.extractPlan(), explored.size(), frontier.size(), initialState);
            }

            explored.add(leafState);
            for (State s : leafState.getExpandedStates()) {
                if (!explored.contains(s) && !frontier.contains(s)) {
                    frontier.add(s);
                }
            }
        }
    }
}

@Data
class WorkerReturn {
    private final Action[][] plan;
    private final int exploredSize;
    private final int frontierSize;
    private final State state;
}
