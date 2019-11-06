package levelgenerator.pgl;

import levelgenerator.Complexity;
import shared.Action.MoveDirection;

import java.awt.*;

interface PGL {

    Point getRandomCoordinate();
    int getCellCount();

    boolean isFrame(Point p);
    boolean isWall(Point p);
    Point getNewPoint(Point p, MoveDirection direction);

    char[] elementsToArray();

    String toString();
    String stateToString(char[][] state);
    String getName();
    String getAlgorithmName();
}
