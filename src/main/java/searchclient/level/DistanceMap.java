package searchclient.level;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class DistanceMap {
    private Map<Coordinate, Map<Coordinate, Integer>> allGoalsDistanceMap;

    public DistanceMap(List<LevelNode> goalNodes) {
        this.allGoalsDistanceMap = new HashMap<>(goalNodes.size());
        for (LevelNode goalNode : goalNodes) {
            Map<Coordinate, Integer> distanceMap = new HashMap<>();
            int distance = 1;
            Deque<LevelNode> neighbourStack = Stream.of(goalNode).collect(Collectors.toCollection(ArrayDeque::new));
            while (!neighbourStack.isEmpty()) {
                Deque<LevelNode> newLevelStack = new ArrayDeque<>();
                LevelNode tempNode = neighbourStack.pollFirst();
                while (tempNode != null) {
                    for (LevelNode neighbour : tempNode.getEdges()) {
                        if (!distanceMap.containsKey(neighbour.getCoordinate()) && !neighbour.equals(goalNode)) {
                            distanceMap.put(neighbour.getCoordinate(), distance);
                            newLevelStack.add(neighbour);
                        }
                    }
                    tempNode = neighbourStack.pollFirst();
                }
                neighbourStack = newLevelStack;
                distance++;
            }
            this.allGoalsDistanceMap.put(goalNode.getCoordinate(), distanceMap);
        }
    }

    public int getDistance(Coordinate from, Coordinate goal)  {
        if (!this.allGoalsDistanceMap.containsKey(goal)) throw new IllegalArgumentException("Invalid goal coordinate");
        Map<Coordinate, Integer> distanceMap = this.allGoalsDistanceMap.get(goal);
        if (!distanceMap.containsKey(from)) throw new IllegalArgumentException("Invalid from coordinate");
        return distanceMap.get(from);
    }
}
