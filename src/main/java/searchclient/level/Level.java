package searchclient.level;

import shared.Farge;

import java.io.BufferedReader;

public class Level {
    //Map Data
    public int[] agentRows;
    public int[] agentCols;
    public Farge[] agentColors;

    public char[][] boxes;
    public Farge[] boxColors;

    public boolean[] walls;

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
    }


}
