package searchclient;

import java.nio.charset.CharsetDecoder;
import java.util.*;

public abstract class Heuristic implements Comparator<State> {
    private HashMap<Character, List<Location>> goals;
    public Heuristic(State initialState) {
        goals = new HashMap<>();
        for (int row = 1; row < initialState.goals.length - 1; row ++) {
            for (int col = 1; col < initialState.goals[row].length - 1; col++) {
                char g = initialState.goals[row][col];
                if (g > 0) {
                    Location goal = new Location(row, col, 0);
                    if (goals.containsKey(g)) {
                        goals.get(g).add(goal);
                    } else {
                        List<Location> goalsList = new ArrayList<>();
                        goalsList.add(goal);
                        goals.put(g, goalsList);
                    }
                }
            }
        }
    }

    public int h(State n) {
        int result = 0;

        List<Location> boxes = new LinkedList<>();
        for (int row = 1; row < n.boxes.length - 1; row ++) {
            for (int col = 1; col < n.boxes[row].length - 1; col++) {
                char b = n.boxes[row][col];
                if (b > 0) {
                    if (goals.containsKey(b)) {
                        List<Location> goalsList = goals.get(b);
                        Location box = new Location(row, col, Integer.MAX_VALUE);
                        boolean success = false;
                        for (Location goal : goalsList) {
                            if (goal.x == box.x && goal.y == box.y) {
                                success = true;
                                break;
                            } else {
                                double distance = Math.sqrt(Math.pow(box.x - goal.x, 2) + Math.pow(box.y - goal.y, 2));
                                if (distance < box.distance) {
                                    box.distance = (int) distance;
                                }
                            }
                        }
                        if (!success) boxes.add(box);
                    }
                }
            }
        }

        if (boxes.isEmpty()) {
            for (Character g : goals.keySet()) {
                if (g <= 9) {
                    int agentRow = n.agentRows[g];
                    int agentCol = n.agentCols[g];
                    Location goal = goals.get(g).get(0);
                    result += Math.abs((Math.sqrt(Math.pow(agentRow - goal.x, 2) + Math.pow(agentCol - goal.y, 2))) - 1);
                }
            }
            return result;
        }
        for (Location box : boxes) {
            result += 100 * box.distance;
            result += Math.abs((Math.sqrt(Math.pow(n.agentRows[0] - box.x, 2) + Math.pow(n.agentCols[0] - box.y, 2))) - 1);
        }
        return result;
    }

    public abstract int f(State n);

    @Override
    public int compare(State n1, State n2) {
        return this.f(n1) - this.f(n2);
    }

    public class Location {
        public final int x;
        public final int y;
        public int distance;

        public Location(int x, int y, int distance) {
            this.x = x;
            this.y = y;
            this.distance = distance;
        }
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

