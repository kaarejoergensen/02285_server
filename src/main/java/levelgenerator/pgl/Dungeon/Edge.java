package levelgenerator.pgl.Dungeon;

import java.awt.*;

public class Edge{

    public Room src; //Rommet den starter fra
    public Room dest;//Rommet den ender
    public int distance; //Hvor langt det er

    public Edge(Room src, Room dest){
        this.dest = dest;
        this.distance = distance;
        this.src = src;
        distance = src.getDistance(dest);
    }


    public boolean equals(Edge obj) {
        //Om både src og dest er like
        if(obj.src == src && obj.dest == dest) return true;
        //Eller om de er snudd om
        if(obj.src == dest && obj.dest == src) return true;
        return false;
    }
}
