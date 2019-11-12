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

    private int tiles;

    //Min størrelse på høyde / bredde på boksene
    private final int MIN_SIZE = (width < 10 || height < 10) ? 2 :3;
    private int maxHeight;
    private int maxWidth;

    private ArrayList<Room> rooms;
    private ArrayList<Edge> edges;

    public int MAX_TRIES = 10000;

    public Dungeon(Complexity c, int levelNumber) {
        super(c, levelNumber);
        rooms = new ArrayList<>();
        edges = new ArrayList<>();

        determineVariables();
        fillLevelWithWalls();
        //Create rooms
        int tries = this.MAX_TRIES;
        while(tries > 0 && tiles < (getTotalSpace() * MINIMUM_SPACE)){
            tries--;
            //Create a room
            Room room = generateRoom();
            //If this room intersects with another, scrap it
            if(isIntersecting(room)) continue;

            rooms.add(room);
            tiles += room.getArea();

        }

        convertRoomToTiles();
        buildEdges();
        convertEdgesToTiles();


        totallyRandomElementDistribution(initStateElements);
        totallyRandomElementDistribution(goalStateElements);

    }

    private void totallyRandomElementDistribution(char[][] state){
        var agents = agentsToArrayList();
        var boxes = boxesToArrayList();
        var rooms_copy = new ArrayList<>(rooms);

        //Agents
        while(agents.size() > 0){
            Room agentRoom = getBiggestRoom(rooms_copy);
            int loop_indexer = (int) Math.floor(agentRoom.getArea() / 2);
            while(agents.size() > 0 && loop_indexer > 0){
                int rng = ThreadLocalRandom.current().nextInt(0,agents.size());
                char temp = agents.get(rng);
                agents.remove(rng);
                agentRoom.addElement(temp, state);
            }
            rooms_copy.remove(agentRoom);
        }

        //Om banen er veldig liten, så fjernes alle rommene til agenter. Må legge de tilbake dersom dette skjer
        if(rooms_copy.size() == 0) rooms_copy = new ArrayList<>(rooms);

        //Box time
        while(boxes.size() > 0){
            int rng = ThreadLocalRandom.current().nextInt(0,rooms_copy.size());
            Room picked_room = rooms_copy.get(rng);
            char temp = boxes.get(0);
            boxes.remove(0);
            picked_room.addElement(temp,state);
        }
    }

    private Room getBiggestRoom(ArrayList<Room> rooms){
        Room temp = null;
        for(Room r  : rooms){
            if(temp == null || temp.getArea() < r.getArea()){
                temp = r;
            }
        }
        return temp;
    }



    private void buildEdges(){
        var rooms_copy = new ArrayList<>(rooms);
        Room chosen = rooms_copy.get(0);
        rooms_copy.remove(0);
        connectClosest(chosen, rooms_copy);
    }

    private void connectClosest(Room r, ArrayList<Room> rooms_copy){
        if(rooms_copy.size() == 0) return;
        Room closestRoom = null;
        for(Room temp : rooms_copy){
            if(!temp.equals(r) && (closestRoom == null || r.getDistance(temp) < r.getDistance(closestRoom))){
                closestRoom = temp;
            }
        }
        rooms_copy.remove(r);

        if(closestRoom != null){
            edges.add(new Edge(findClosestRoom(r, closestRoom), closestRoom));
            connectClosest(closestRoom,rooms_copy);

        }
    }

    private Room findClosestRoom(Room r, Room closest) {
        Room temp = r;
        while(true){
            ArrayList<Edge> edges = new ArrayList<>(r.getEdges());
            Room candidate = null;
            for(Edge e : edges){
                Room dest = e.src == r ? e.dest : e.src;
                if(dest.getDistance(closest) < r.getDistance(closest)){
                    candidate = dest;
                }
            }
            if(candidate == null) break;
            r = candidate;
        }
        //if(!r.equals(temp)) System.out.println("New closest to " +  closest + " found! -> OG: " + r + " vs: " + temp);
        return temp;
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

            boolean prioritizeX = distanceBetweenPoints(dest.x, tmp.x) > distanceBetweenPoints(dest.y, tmp.y) ? true : false;

            while(!tmp.equals(dest)){
                walls[tmp.y][tmp.x] = ' ';

                if(prioritizeX){
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
                }else{
                    if(tmp.y != dest.y){
                        if(tmp.y > dest.y) tmp.y--;
                        if(tmp.y < dest.y) tmp.y++;
                        continue;
                    }
                    if(tmp.x != dest.x){
                        if(tmp.x > dest.x) tmp.x--;
                        if(tmp.x < dest.x) tmp.x++;
                        continue;
                    }
                }

            }
        }
    }



    private int distanceBetweenPoints(int a, int b){
        return Math.abs(a-b);
    }

    /*
    Debug Functions
     */

    private void printDebug(){
        addRoomNameInTiles(); //Debug
        System.out.println(wallsDebugString()); //Debug
        printEdges(); //Debug

        //System.out.println("Dungeon Generated!"+ System.lineSeparator() + "Number of rooms: " + rooms.size() + System.lineSeparator() + "Edge count: " + edges.size());

    }


    private void printCentroids(){
        for(Room r : rooms){
            System.out.println("Room: " + r + " Centroid: " + r.centre + " Area: " + r.getArea());
        }
    }

    private void printEdges(){
        for(Edge e : edges){
            System.out.println("Edge: Src: " + e.src + "  - Dest: " + e.dest);
        }
    }

    private void addRoomNameInTiles(){
        for(Room r : rooms){
            walls[r.centre.y][r.centre.x-1] = r.toString().charAt(0);
            walls[r.centre.y][r.centre.x] = r.toString().charAt(1);
            walls[r.centre.y][r.centre.x+1] = r.toString().charAt(2);
        }
    }



    @Override
    public String getAlgorithmName() {
        return "Dungeon";
    }
}
