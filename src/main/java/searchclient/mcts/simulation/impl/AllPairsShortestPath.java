package searchclient.mcts.simulation.impl;

import searchclient.Heuristic;
import searchclient.State;
import searchclient.level.Coordinate;
import searchclient.mcts.Node;
import searchclient.mcts.simulation.Simulation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

public class AllPairsShortestPath implements Simulation {
    private HashMap<Character, List<Location>> goals;

    public AllPairsShortestPath(State initialState) {
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

    @Override
    public int simulatePlayout(Node node) {
        State n = node.getState();
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
                                int distance = n.distanceMap.getDistance(new Coordinate(box.x, box.y), new Coordinate(goal.x, goal.y));
                                if (distance < box.distance) {
                                    box.distance = distance;
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
                    if (agentRow != goal.x || agentCol != goal.y) {
                        result += n.distanceMap.getDistance(new Coordinate(agentRow, agentCol), new Coordinate(goal.x, goal.y));
                    }
                }
            }
            return result;
        }
        for (Location box : boxes) {
            result += 100 * box.distance;
        }
        return result;
    }

    private static class Location {
        public final int x;
        public final int y;
        int distance;

        Location(int x, int y, int distance) {
            this.x = x;
            this.y = y;
            this.distance = distance;
        }
    }

}
