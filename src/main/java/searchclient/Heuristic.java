package searchclient;

import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public abstract class Heuristic implements Comparator<State> {
    public Heuristic(State initialState) {
        // Here's a chance to pre-process the static parts of the level.
    }

    public int h(State n) {
        int result = 0;

        List<Location> boxes = new LinkedList<>();
        List<Location> goals = new LinkedList<>();

        for (int row = 1; row < n.boxes.length - 1; row ++) {
            for (int col = 1; col < n.boxes[row].length - 1; col++) {
                char g = n.goals[row][col];
                char b = Character.toLowerCase(n.boxes[row][col]);
                if (g > 0) {
                    goals.add(new Location(row, col, g, Integer.MAX_VALUE));
                }
                if (b > 0) {
                    boxes.add(new Location(row, col, b, Integer.MAX_VALUE));
                }
            }
        }

        for (Location goal : goals) {
            Iterator<Location> itr = boxes.iterator();
            while (itr.hasNext()) {
                Location box = itr.next();
                if (goal.character == box.character) {
                    if (goal.x == box.x && goal.y == box.y) {
                        itr.remove();
                        break;
                    } else {
                        double distance = Math.sqrt(Math.pow(box.x - goal.x, 2) + Math.pow(box.y - goal.y, 2));
                        if (distance < box.distance) {
                            box.distance = (int) distance;
                        }
                    }
                }
            }
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
        public final char character;
        public int distance;

        public Location(int x, int y, char character, int distance) {
            this.x = x;
            this.y = y;
            this.character = character;
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

