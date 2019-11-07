package levelgenerator.pgl;

import levelgenerator.Complexity;
import shared.Action;
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

        walls = new char[c.height][c.width];
        initStateElements = new char[c.height][c.width];
        goalStateElements = new char[c.height][c.width];

        fargeList = Farge.clientFargerToList();
        randomAssignAvailableColors();
        elementColors = new char[fargeList.size()][c.boxes + c.agents];

        assignAgentsToColors();
        assignBoxesToColors();
    }

    //Må bli kalt før colorlisten
    public void randomAssignAvailableColors(){
        int maxColors = fargeList.size();
        //TODO: Dette skal fikses når wizard baner skal kunne genereres
        int colorCount = complexity.colors > complexity.agents ? complexity.agents : complexity.colors;

        for(int i = colorCount; i < maxColors; i++){
            int removeIndex = ThreadLocalRandom.current().nextInt(0, fargeList.size());
            fargeList.remove(removeIndex);
        }
    }

    public void assignAgentsToColors(){
        //First make sure every color have at least one agent
        int i = 0;
        for(; i < fargeList.size(); i++){
            elementColors[i][0] = (char)(i + '0');
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
        int indexInFarge = ThreadLocalRandom.current().nextInt(0, fargeList.size());
        for(int j = 0; j < (complexity.agents + complexity.boxes); j++){
            if(elementColors[indexInFarge][j] == '\0'){
                elementColors[indexInFarge][j] = (c);
                break;
            }
        }
    }




    public void fillLevelWithWalls(){
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width ; x++){
                walls[y][x] = '+';
            }
        }
    }


    public Point getRandomCoordinate(){
        int y = ThreadLocalRandom.current().nextInt(1, height-1 );
        int x = ThreadLocalRandom.current().nextInt(1,  width-1 );
        return new Point(x,y);
    }

    public boolean isWall(Point p){
        try {
            return walls[p.y][p.x] == '+';
        }catch(ArrayIndexOutOfBoundsException e){
            System.out.println(e + System.lineSeparator());
            System.out.println(wallsToString());
        }
        return true;
    }

    public boolean isFrame(Point p){
        return p.x == 0 || p.x == (width-1) || p.y == 0 || p.y == (height-1);
    }

    public Point getNewPoint(Point p, Action.MoveDirection direction){
        return new Point(p.x + direction.getDeltaCol(), p.y + direction.getDeltaRow());
    }

    public char[] elementsToArray(){
        char[] elements = new char[complexity.agents + complexity.boxes];
        int i = 0;
        for(char[] a : elementColors){
            for(char c : a){
                if(c != '\0'){
                    elements[i] = c;
                    i++;
                }
            }
        }
        return elements;
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

    public int getTotalSpace(){
        return (width-2) * (height-2);
    }

    public String toString(){
        String out = "#domain" + System.lineSeparator() + "hospital2" + System.lineSeparator() + "#levelname" + System.lineSeparator() +  getName() + System.lineSeparator();
        //Color Section
        out += "#colors" + System.lineSeparator();
        for(int i = 0; i < fargeList.size(); i++){
            Farge temp = fargeList.get(i);
            out += temp.name() + ":" ;
            StringBuilder sb = new StringBuilder();
            for(char c : elementColors[i]){
                if(c == '\0') break;
                sb.append(c).append(',');
            }
            out += sb.toString().substring(0, sb.length() - 1) + System.lineSeparator();
        }


        out += "#initial" + System.lineSeparator();
        out += stateToString(initStateElements);
        out += "#goal" + System.lineSeparator();
        out += stateToString(goalStateElements);
        out += "#end";
        return out;
    }



    public String stateToString(char[][] state){
        String out = "";
        for(int x = 0; x < width; x++){
            for(int y = 0; y< height; y++){
                if(walls[y][x] == '+'){
                    out += walls[y][x];
                    continue;
                }else if(state[y][x] != 0){
                    out += state[y][x];
                }
                else{
                    out += ' ';
                }
            }
            out += System.lineSeparator();
        }
        return out;
    }

    public String wallsToString(){
        return stateToString(walls);
    }

    public String getName(){
        return levelNumber + "_" + getAlgorithmName() + "_" + complexity.label;
    }


}
