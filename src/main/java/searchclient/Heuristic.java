package searchclient;

import searchclient.level.Coordinate;
import shared.Action;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

public abstract class Heuristic implements Comparator<State> {
    private HashMap<Character, List<Coordinate>> goals;
    public Heuristic(State initialState) {
        goals = new HashMap<>();
        for (int row = 1; row < initialState.goals.length - 1; row ++) {
            for (int col = 1; col < initialState.goals[row].length - 1; col++) {
                char g = initialState.goals[row][col];
                if (g > 0) {
                    Coordinate goal = new Coordinate(row, col);
                    if (goals.containsKey(g)) {
                        goals.get(g).add(goal);
                    } else {
                        List<Coordinate> goalsList = new ArrayList<>();
                        goalsList.add(goal);
                        goals.put(g, goalsList);
                    }
                }
            }
        }
    }

    public int h(State n) {
        int result = 0;
        for (int row = 1; row < n.boxes.length - 1; row ++) {
            for (int col = 1; col < n.boxes[row].length - 1; col++) {
                char b = n.boxes[row][col];
                if (b > 0) {
                    if (goals.containsKey(b)) {
                        List<Coordinate> goalsList = goals.get(b);
                        Coordinate box = new Coordinate(row, col);
                        int finalDistance = Integer.MAX_VALUE;
                        for (Coordinate goal : goalsList) {
                            if (goal.getRow() == box.getRow() && goal.getCol() == box.getCol()) {
                                break;
                            } else {
                                int distance = n.distanceMap.getDistance(box, goal);
                                if (distance < finalDistance) {
                                    finalDistance = distance;
                                }
                            }
                        }
                        if (finalDistance != Integer.MAX_VALUE) {
                            result += 2 * finalDistance;
                            result += n.distanceMap.getDistance(new Coordinate(n.agentRows[0], n.agentCols[0]), box);
                        }
                    }
                }
            }
        }

        if (result == 0) {
            for (Character g : goals.keySet()) {
                if ('0' <= g && g <= '9') {
                    int agentRow = n.agentRows[g - '0'];
                    int agentCol = n.agentCols[g - '0'];
                    Coordinate goal = goals.get(g).get(0);
                    if (agentRow != goal.getRow() || agentCol != goal.getCol()) {
                        int distance = n.distanceMap.getDistance(new Coordinate(agentRow, agentCol), goal);
                        result += distance;
                    }
                }
            }
        }
        for (Action a : n.jointAction) {
            result += a.getType().equals(Action.ActionType.NoOp) ? 0 : 1;
        }
        return result;
    }

    public abstract int f(State n);

    @Override
    public int compare(State n1, State n2) {
        return this.f(n1) - this.f(n2);
    }

}

class HeuristicAStar extends Heuristic {
    public HeuristicAStar(State initialState) {
        super(initialState);
    }

    @Override
    public int f(State n) {
        return n.g() + this.h(n);
    }

    @Override
    public String toString() {
        return "A* evaluation";
    }
}

class HeuristicWeightedAStar extends Heuristic {
    private int w;

    public HeuristicWeightedAStar(State initialState, int w) {
        super(initialState);
        this.w = w;
    }

    @Override
    public int f(State n) {
        return n.g() + this.w * this.h(n);
    }

    @Override
    public String toString() {
        return String.format("WA*(%d) evaluation", this.w);
    }
}

class HeuristicGreedy extends Heuristic {
    public HeuristicGreedy(State initialState) {
        super(initialState);
    }

    @Override
    public int f(State n) {
        return this.h(n);
    }

    @Override
    public String toString() {
        return "greedy evaluation";
    }
}

