package levelgenerator.pgl.Dungeon;

import java.awt.*;

public class Edge{

    public Room src; //Rommet den starter fra
    public Room dest;//Rommet den ender
    public int distance; //Hvor langt det er

    public Edge(Room src, Room dest, int distance){
        this.dest = dest;
        this.distance = distance;
        this.src = src;
    }

}
