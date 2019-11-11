package levelgenerator.pgl.Dungeon;

import java.awt.*;
import java.util.ArrayList;
import java.util.UUID;

import lombok.Getter;

public class Room {
    @Getter
    private String id;

    public Point centre;
    public Rectangle rect;

    public ArrayList<Edge> edges;

    public Room (Rectangle rect){
        this.rect = rect;
        edges = new ArrayList();
        setCentroid();
    }

    private void setCentroid(){
        int x = (int)Math.floor(rect.width / 2);
        int y = (int) Math.ceil(rect.height/2);
        centre = new Point(rect.x + x, rect.y + y);
        id = UUID.randomUUID().toString();
    }

    public void addEdge(Edge edge){
        edges.add(edge);
    }

    public ArrayList<Edge> getEdges(){
        return edges;
    }

    public boolean hasEdge(Room dest){
        for(Edge e : edges){
            if(e.equals(new Edge(this, dest))){
                return true;
            }
        }
        return false;
    }

    public int getArea(){
        return rect.width * rect.height;
    }

    //Manhattan distance
    public int getDistance(Room destination){
        return Math.abs(centre.x - destination.centre.x) + Math.abs(centre.y - destination.centre.y);
    }


    public boolean equals(Room obj) {
        return obj.id.equals(this.id);
    }

    @Override
    public String toString() {
        return this.getId().substring(0,3);
    }

}
