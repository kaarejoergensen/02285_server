package searchclient.mcts.simulation.impl;

import searchclient.State;
import searchclient.level.Coordinate;
import searchclient.mcts.model.Node;
import searchclient.mcts.simulation.Simulation;
import shared.Action;

import java.util.*;
import java.util.stream.Collectors;

public class AllPairsShortestPath implements Simulation {
    private static final int MAX_SCORE = 100;

    private HashMap<Character, List<Coordinate>> goals;

    public AllPairsShortestPath(State initialState) {
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

    @Override
    public float simulatePlayout(Node node) {
        State n = node.getState();
        if (n.isGoalState()) {
            return MAX_SCORE;
        }
        float result = 0;
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
                            result += 10 * finalDistance;
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
        if (n.jointAction != null) {
            for (Action a : Arrays.stream(n.jointAction).filter(Objects::nonNull).collect(Collectors.toList())) {
                result += a.getType().equals(Action.ActionType.NoOp) ? 0 : 1;
            }
        }
//        System.err.println(result + "|" + MAX_SCORE / result);
        return MAX_SCORE / result;
    }
}
