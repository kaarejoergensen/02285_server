package levelgenerator.pgl;

import levelgenerator.Complexity;

import java.awt.*;

interface PGL {

    public Point getRandomCoordinate();
    public int getCellCount();

    public String toString();
    public String stateToString(char[][] state);
    public String getName();
    public String getAlgorithmName();
}
