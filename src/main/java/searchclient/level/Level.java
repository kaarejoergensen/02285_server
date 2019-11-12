package searchclient.level;

import searchclient.State;
import shared.Farge;

import java.io.BufferedReader;

public class Level {
    public DistanceMap distanceMap;
    //Map Data
    public int[] agentRows;
    public int[] agentCols;
    public Farge[] agentColors;

    //TOOD: Finne en annen løsning på dette
    public int numAgents;

    public char[][] boxes;
    public Farge[] boxColors;

    public boolean[][] walls;

    public char[][] goals;

    //Map Details
    public int height;
    public int width;

    //Level Credentials
    public String name;
    public String domain;

    public LevelParser parse;

    //Rules
    final static int MAX_AGENTS = 10;
    final static int BOX_COLORS = 26;

    public Level(BufferedReader file){
        parse = new LevelParser(file,this);

        agentColors = new Farge[MAX_AGENTS];
        boxColors = new Farge[BOX_COLORS];

        agentRows = new int[MAX_AGENTS];
        agentCols = new int[MAX_AGENTS];

        height = 0;
        width = 0;

        numAgents = 0;

    }

    public void setMapDetails(int width, int height){
        this.height = height;
        this.width = width;
    }


    public void initiateMapDependentArrays(){
        goals = new char[width][height];
        boxes = new char[width][height];
        walls = new boolean[width][height];

    }

    public State toState(){
        return new State(distanceMap, agentRows, agentCols, agentColors, walls, boxes, boxColors, goals);
    }
}
