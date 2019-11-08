package levelgenerator.pgl.Dungeon;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class MST {

    //Lagret som HashSet for å enklere
    private HashSet<HashSet<String>> allRooms;

    private ArrayList<Edge> primEdges;
    private ArrayList<Edge> tempEdges;
    private ArrayList<Edge> operationalEdges;

    private String selectedRoom;
    public MST(){

        allRooms = new HashSet<>();
        primEdges = new ArrayList<>();
        tempEdges = new ArrayList<>();


    }

    public ArrayList<Edge> primMST(){
        operationalEdges = new ArrayList<>();
        ArrayList<Edge> MSTEdges = new ArrayList<>();

        return MSTEdges;
    }

    public void addEdge(Room src, Room dest, int distance){
        allRooms.add(Stream.of(src.toString()).collect(Collectors.toCollection(HashSet::new)));
        allRooms.add(Stream.of(dest.toString()).collect(Collectors.toCollection(HashSet::new)));
        primEdges.add(new Edge(src, dest, src.getDistance(dest)));
        selectedRoom = src.getId();
    }


    //Henter en liste over alle edges som er koblet til 'Selected'. Som 'Selected' er 'dest',
    //så byttes src og dest
    private void getSelectedEdges(){
        for(int i = 0 ; i < tempEdges.size(); i++){
            Edge target =  tempEdges.get(i);
            if(target.src.getId().equals(selectedRoom)){

            }
        }
    }





}
