package searchclient.level;

import searchclient.State;
import shared.Farge;

import java.io.BufferedReader;
import java.util.Arrays;

public class Level {
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

    //TODO: Faktisk legge inn width og height
    public void initateMapDependentArrays(){
        goals = new char[75][75];
        boxes = new char[75][75];
        walls = new boolean[75][75];

    }

    public State toState(){
        return new State(agentRows, agentCols, agentColors, walls, boxes, boxColors, goals);
    }

    @Override
    public String toString() {
        String output;
        System.err.println("Walls: " + goals);
        System.err.println("AgentRows: " + Arrays.toString(agentRows));
        System.err.println("AgentCols: " + Arrays.toString(agentCols));
        System.err.println("Goals " + Arrays.toString(goals));

        return super.toString();
    }


}
