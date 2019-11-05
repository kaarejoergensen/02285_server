package levelgenerator.pgl;

import levelgenerator.Complexity;
import levelgenerator.pgl.Basic;
import org.apache.logging.log4j.util.Strings;
import shared.Farge;

import java.awt.*;
import java.util.ArrayList;
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
    public char[][] elementColors;
    public ArrayList<Farge> fargeList;




    public RandomLevel(Complexity c, int levelNumber){
        this.levelNumber = levelNumber;
        this.complexity = c;
        this.width = c.width;
        this.height = c.height;


        walls = new char[c.width][c.height];
        initStateElements = new char[c.width][c.height];
        goalStateElements = new char[c.width][c.height];

        fargeList = Farge.clientFargerToList();
        elementColors = new char[Farge.getClientFarger().length][c.boxes + c.agents];

        randomAssignAvailableColors();
        System.out.println(fargeList);
        assignAgentsToColors();
        assignBoxesToColors();
        System.out.println(Arrays.deepToString(elementColors));
    }
    /*
    public randomlyAssignElementsToColors(){
        int index = ThreadLocalRandom.current().nextInt(0, fargeList.size());
        for(int i = 0; i < complexity.colors; i++){

        }
    }
    */

    public void assignAgentsToColors(){
        //First make sure every color have at least one agent
        int i = 0;
        for(; i < fargeList.size(); i++){
            int indexInFarge = fargeList.get(i).ordinal();
            elementColors[indexInFarge][0] = (char)(i + '0');
        }
        //Then random distribute the rest
        //Pick a random color, and proceed to add next agent into that one
        for(; i < complexity.agents; i++){
            randomAllocateElementToColor((char)(i + '0'));
        }
    }


    public void assignBoxesToColors(){
        for(int i = 65; i <  65 + complexity.boxes;i++){
            randomAllocateElementToColor((char)i);
        }
    }

    private void randomAllocateElementToColor(char c){
        int indexInFarge = fargeList.get(ThreadLocalRandom.current().nextInt(0, fargeList.size())).ordinal();
        for(int j = 0; j < (complexity.agents + complexity.boxes); j++){
            if(elementColors[indexInFarge][j] == '\0'){
                elementColors[indexInFarge][j] = (c);
                break;
            }
        }
    }


    public void randomAssignAvailableColors(){
        int maxColors = fargeList.size();
        //TODO: Dette skal fikses når wizard baner skal kunne genereres
        int colorCount = complexity.colors > complexity.agents ? complexity.agents : complexity.colors;

        for(int i = colorCount; i < maxColors; i++){
            int removeIndex = ThreadLocalRandom.current().nextInt(0, fargeList.size());
            fargeList.remove(removeIndex);
        }
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
        //Color Section
        out += "#Colors" + System.lineSeparator();
        for(Farge f: fargeList){
            out += f.name() + ":" ;
            int indexFarge = fargeList.get(ThreadLocalRandom.current().nextInt(0, fargeList.size())).ordinal();
            StringBuilder sb = new StringBuilder();
            for(char c : elementColors[indexFarge]){
                if(c == '\0') break;
                sb.append(c).append(',');
            }
            out += sb.toString().substring(0, sb.length() - 1) + System.lineSeparator();
        }


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
