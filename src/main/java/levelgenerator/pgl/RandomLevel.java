package levelgenerator.pgl;

import levelgenerator.Complexity;
import levelgenerator.pgl.Basic;
import shared.Farge;

import java.awt.*;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public abstract class RandomLevel implements PGL{
    //Info
    private int levelNumber;
    public Complexity complexity;


    //Canvas
    public int width;
    public int height;

    //Level
    public char[][] walls;
    public char[][] initStateElements;
    public char[][] goalStateElements;

    //Colors
    public char[][] colors;



    public RandomLevel(Complexity c, int levelNumber){
        this.levelNumber = levelNumber;
        this.complexity = c;
        this.width = c.width;
        this.height = c.height;


        walls = new char[c.width][c.height];
        initStateElements = new char[c.width][c.height];
        goalStateElements = new char[c.width][c.height];

    }




    public Point getRandomCoordinate(){
        int x = ThreadLocalRandom.current().nextInt(1, height );
        int y = ThreadLocalRandom.current().nextInt(1,  width );
        return new Point(x,y);
    }

    public int getCellCount(){
        int cellCount = 0;
        for(int x = 0; x < width; x++){
            for(int y = 0; y< height; y++){
                if(walls[x][y] == '+');
            }
        }
        return cellCount;
    }

    public String toString(){
        String out = "#domain" + System.lineSeparator() + "hospital2" + System.lineSeparator() + getName() + System.lineSeparator();
        out += "#Initial" + System.lineSeparator();
        out += stateToString(initStateElements);
        out += "#Goal" + System.lineSeparator();
        out += stateToString(goalStateElements);
        return out;
    }

    public String stateToString(char[][] state){
        String out = "";
        for(int x = 0; x < width; x++){
            for(int y = 0; y< height; y++){
                if(walls[x][y] == '+'){
                    out += walls[x][y];
                    continue;
                }else if(state[x][y] != 0){
                    out += state[x][y];
                }
                else{
                    out += ' ';
                }
            }
            out += System.lineSeparator();
        }
        return out;
    }

    public String getName(){
        return levelNumber + "_" + getAlgorithmName() + "_" + complexity.label;
    }


}
