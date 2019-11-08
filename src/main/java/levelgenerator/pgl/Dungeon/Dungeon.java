package levelgenerator.pgl.Dungeon;

import levelgenerator.Complexity;
import levelgenerator.pgl.RandomLevel;

import java.awt.*;
import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;

public class Dungeon extends RandomLevel {

    /*
        Logic for the Duengon generator.
        Article: https://www.gamasutra.com/blogs/AAdonaac/20150903/252889/Procedural_Dungeon_Generation_Algorithm.php?fbclid=IwAR19vvpMo35FS7klCGEqduwE4QiTF6UJ0UfmuFasvnU1kNgzcH-ZDiS5m0s

     */

    //
    private Point mapCentre;
    //Hvor mange rom skal lages?
    private final double MINIMUM_SPACE = 0.3;
    private final double MAX_ROOMSIZE_FACTOR = 3;
    private final int MAX_ROOMSIZE = 15;
    //Hvor mange tiles som finnes
    private int tiles;

    //Min størrelse på høyde / bredde på boksene
    private final int MIN_SIZE = (width < 10 || height < 10) ? 2 :3;
    private int maxHeight;
    private int maxWidth;

    private ArrayList<Room> rooms;
    private ArrayList<Edge> edges;

    //MST Stuff


    public Dungeon(Complexity c, int levelNumber) {
        super(c, levelNumber);
        rooms = new ArrayList<>();
        edges = new ArrayList<>();

        determineVariables();
        fillLevelWithWalls();
        //Create rooms
        while(tiles < (getTotalSpace() * MINIMUM_SPACE)){
            //Create a room
            Room room = generateRoom();
            //If this room intersects with another, scrap it
            if(isIntersecting(room)) continue;

            rooms.add(room);
            tiles += room.getArea();
        }
        System.out.println("Rooms Created");
        //Convert rooms to tiles
        convertRoomToTiles();
        //Find an edge (hallway) for every room (centroid)
        generateEdges();
        convertEdgesToTiles();


        System.out.println(wallsToString());

        System.out.println("STOP");
        //Make a path to every room?
    }

    public void generateEdges(){
        for(int i = 0; i < rooms.size(); i++){
            Room dest = (i == (rooms.size()-1)) ? rooms.get(0) : rooms.get(i+1);
            Edge temp = new Edge(rooms.get(i),dest, rooms.get(i).getDistance(dest));
            edges.add(temp);
        }
    }


    private boolean isIntersecting(Room room){
        for(Room r : rooms){
            var tmp = r.rect;
            var rect = new Rectangle(tmp.x-1,tmp.y-1,tmp.width+2,tmp.height+2);
            if(rect.intersects(room.rect)){
                return true;
            }
        }
        return false;
    }

    private void determineVariables(){
        tiles = 0;
        mapCentre = new Point(width/2,height/2);
        //Setting rect bounds
        maxWidth = (int) Math.floor(width / MAX_ROOMSIZE_FACTOR);
        if(maxWidth < MIN_SIZE) maxWidth = MIN_SIZE;
        if(maxWidth > MAX_ROOMSIZE) maxWidth = MAX_ROOMSIZE;
        maxHeight= (int) Math.floor(height / MAX_ROOMSIZE_FACTOR);
        if(maxHeight < MIN_SIZE) maxHeight = MIN_SIZE;
        if(maxHeight > MAX_ROOMSIZE) maxHeight = MAX_ROOMSIZE;
    }

    private Room generateRoom(){
        Rectangle r = new Rectangle();

        //Minimum burde være 4x4 og maks?
        r.width = maxWidth == MIN_SIZE ? MIN_SIZE : ThreadLocalRandom.current().nextInt(MIN_SIZE, maxWidth);
        r.height = maxHeight == MIN_SIZE ? MIN_SIZE : ThreadLocalRandom.current().nextInt(MIN_SIZE, maxHeight);
        //Random placement
        r.x = ThreadLocalRandom.current().nextInt(1, width - r.width);
        r.y = ThreadLocalRandom.current().nextInt(1, height - r.height);

        Room room = new Room(r);
        return room;
    }

    private void convertRoomToTiles(){
        for(Room room : rooms){
            var rect = room.rect;
            for(int y = rect.y ; y < (rect.y + rect.height); y++){
                for(int x = rect.x ; x < (rect.x + rect.width); x++){
                    walls[y][x] = ' ';
                }
            }
        }
    }

    private void convertEdgesToTiles(){
        for(Edge e : edges){
            Point dest = e.dest.centre;
            Point tmp = new Point(e.src.centre.x, e.src.centre.y);

            while(!tmp.equals(dest)){
                walls[tmp.y][tmp.x] = ' ';

                if(tmp.x != dest.x){
                    if(tmp.x > dest.x) tmp.x--;
                    if(tmp.x < dest.x) tmp.x++;
                    continue;
                }
                if(tmp.y != dest.y){
                    if(tmp.y > dest.y) tmp.y--;
                    if(tmp.y < dest.y) tmp.y++;
                    continue;
                }
            }

            System.out.println("Done? " + tmp + " vs " + dest);

        }
    }

    private void printCentroids(){
        for(Room r : rooms){
            System.out.println("Room: " + r + " Centroid: " + r.centre);
        }
    }

    private void printEdges(){
        for(Edge e : edges){
            System.out.println("Edge; Src: " + e.src + "  - Dest: " + e.dest);
        }
    }








    @Override
    public String getAlgorithmName() {
        return "Dungeon";
    }
}
