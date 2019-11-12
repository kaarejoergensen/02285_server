package levelgenerator.pgl;

import levelgenerator.Complexity;
import shared.Action.MoveDirection;

import java.awt.*;
import java.util.ArrayList;

interface PGL {

    Point getRandomCoordinate();
    int getCellCount();

    boolean isFrame(Point p);
    boolean isWall(Point p);
    Point getNewPoint(Point p, MoveDirection direction);

    char[] elementsToArray();
    char[] agentsToArray();
    ArrayList<Character> agentsToArrayList();
    ArrayList<Character> boxesToArrayList();

    String toString();
    String stateToString(char[][] state);
    String getName();
    String getAlgorithmName();
}
