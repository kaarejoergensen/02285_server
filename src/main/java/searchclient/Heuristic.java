package searchclient;

import searchclient.level.Box;
import searchclient.level.Coordinate;
import shared.Action;

import java.util.*;

public abstract class Heuristic implements Comparator<State> {
    private HashMap<Character, List<Coordinate>> goals;
    public Heuristic(State initialState) {
        goals = new HashMap<>();
        for (Map.Entry<Coordinate, Character> goal : initialState.goals.entrySet()) {
            if (goals.containsKey(goal.getValue())) {
                goals.get(goal.getValue()).add(goal.getKey());
            } else {
                List<Coordinate> goalsList = new ArrayList<>();
                goalsList.add(goal.getKey());
                goals.put(goal.getValue(), goalsList);
            }
        }
    }

    public int h(State n) {
        if (n.h == -1) {
            int result = 0;
            for (Map.Entry<Coordinate, Box> boxEntry : n.boxMap.entrySet()) {
                Box box = boxEntry.getValue();
                List<Coordinate> goalsList = goals.get(box.getCharacter());
                int finalDistance = Integer.MAX_VALUE;
                for (Coordinate goal : goalsList) {
                    if (goal.equals(boxEntry.getKey())) {
                        break;
                    } else {
                        int distance = n.distanceMap.getDistance(boxEntry.getKey(), goal);
                        if (distance < finalDistance) {
                            finalDistance = distance;
                        }
                    }
                }
                if (finalDistance != Integer.MAX_VALUE) {
                    result += 20 * finalDistance;
                    result += 10 * n.distanceMap.getDistance(new Coordinate(n.agentRows[0], n.agentCols[0]), boxEntry.getKey());
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
            if (n.jointAction != null) {
                for (Action a : n.jointAction) {
                    result += a.getType().equals(Action.ActionType.NoOp) ? 0 : 1;
                }
            }
            n.h = result;
        }
        return n.h;
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

