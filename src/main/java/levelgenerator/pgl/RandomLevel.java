package levelgenerator.pgl;

import levelgenerator.Complexity;
import searchclient.State;
import searchclient.level.*;
import shared.Action;
import shared.Farge;

import java.awt.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

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

        validateInput();

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
        int colorCount = Math.min(complexity.colors, complexity.agents);

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

    public char[] agentsToArray(){
        char[] agents = new char[complexity.agents];
        int index = 0;
        for(char e : elementsToArray()){
            if(Character.isDigit(e)){
                agents[index] = e;
                index++;
            }
        }
        return agents;
    }

    public ArrayList<Character> agentsToArrayList(){
        ArrayList<Character> temp = new ArrayList<>();
        for(char c : elementsToArray()){
            if(Character.isDigit(c)){
                temp.add(c);
            }
        }
        return temp;
    }

    public ArrayList<Character> boxesToArrayList(){
        ArrayList<Character> temp = new ArrayList<>();
        for(char c : elementsToArray()){
            if(Character.isLetter(c)){
                temp.add(c);
            }
        }
        return temp;
    }


    public int getCellCount(){
        int cellCount = 0;
        for(int y = 0; y < height; y++){
            for(int x = 0; y< width; x++){
                if(walls[y][x] == '+');
            }
        }
        return cellCount;
    }

    public State toState() {
        Level level = new Level();
        level.domain = "hospital2";
        level.name = getName();
        for (int i = 0; i < fargeList.size(); i++) {
            for (char c : elementColors[i]) {
                if (c == '\0') break;
                if ('0' <= c && c <= '9') {
                    level.agentColors[c - '0'] = fargeList.get(i);
                } else if ('A' <= c && c <= 'Z') {
                    level.boxColors[c - 'A'] = fargeList.get(i);
                }
            }
        }
        level.setMapDetails(width, height);
        level.initiateMapDependentArrays();
        LevelNode[][] tiles = new LevelNode[level.height][level.width];
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                char c = initStateElements[row][col];
                if (walls[row][col] == '+') {
                    level.walls[row][col] = true;
                } else if ('0' <= c && c <= '9') {
                    level.agentRows[c - '0'] = row;
                    level.agentCols[c - '0'] = col;
                    ++level.numAgents;
                } else if ('A' <= c && c <= 'Z') {
                    Box box = new Box(c, level.boxColors[c - 'A']);
                    level.boxMap.put(new Coordinate(row, col), box);
                }
                if (walls[row][col] != '+') {
                    LevelNode node = new LevelNode(new Coordinate(row, col));
                    tiles[row][col] = node;
                    if (row > 0 && tiles[row - 1][col] != null) {
                        node.addEdge(tiles[row - 1][col]);
                        tiles[row - 1][col].addEdge(node);
                    }
                    if (col > 0 && tiles[row][col - 1] != null) {
                        node.addEdge(tiles[row][col - 1]);
                        tiles[row][col - 1].addEdge(node);
                    }
                }
            }
        }
        List<LevelNode> levelNodes = Arrays.stream(tiles).flatMap(Arrays::stream)
                .filter(Objects::nonNull).collect(Collectors.toList());
        level.distanceMap = new DistanceMap(levelNodes);
        level.agentRows = Arrays.copyOf(level.agentRows, level.numAgents);
        level.agentCols = Arrays.copyOf(level.agentCols, level.numAgents);

        for (int row = 0; row < goalStateElements.length; row++) {
            for (int col = 0; col < goalStateElements[row].length; col++) {
                char c = goalStateElements[row][col];
                if (('0' <= c && c <= '9') || ('A' <= c && c <= 'Z')) {
                    level.goals.put(new Coordinate(row, col), c);
                }
            }
        }
        return level.toState();
    }

    public int getTotalSpace(){
        return (width-2) * (height-2);
    }

    public String toString(){
        StringBuilder out = new StringBuilder("#domain" + System.lineSeparator() + "hospital2" + System.lineSeparator() + "#levelname" + System.lineSeparator() + getName() + System.lineSeparator());
        //Color Section
        out.append("#colors").append(System.lineSeparator());
        for(int i = 0; i < fargeList.size(); i++){
            Farge temp = fargeList.get(i);
            out.append(temp.name()).append(":");
            StringBuilder sb = new StringBuilder();
            for(char c : elementColors[i]){
                if(c == '\0') break;
                sb.append(c).append(',');
            }
            out.append(sb.toString(), 0, sb.length() - 1).append(System.lineSeparator());
        }


        out.append("#initial").append(System.lineSeparator());
        out.append(stateToString(initStateElements));
        out.append("#goal").append(System.lineSeparator());
        out.append(stateToString(goalStateElements));
        out.append("#end");
        return out.toString();
    }



    public String stateToString(char[][] state){
        StringBuilder out = new StringBuilder();
        for(int y = 0; y < height; y++){
            for(int x = 0; x< width; x++){
                if(walls[y][x] == '+'){
                    out.append(walls[y][x]);
                }else if(state[y][x] != 0){
                    out.append(state[y][x]);
                }
                else{
                    out.append(' ');
                }
            }
            out.append(System.lineSeparator());
        }
        return out.toString();
    }

    public String wallsToString(){
        return stateToString(walls);
    }

    public String wallsDebugString(){
        StringBuilder out = new StringBuilder();
        for(int y = 0; y < height; y++){
            for(int x = 0; x< width; x++){
                out.append(walls[y][x]);
            }
            out.append(System.lineSeparator());
        }
        return out.toString();
    }

    public String getName(){
        return levelNumber + "_" + getAlgorithmName() + "_" + complexity.label;
    }

    public void validateInput(){
        if(height < 3 || width < 3) throw new IllegalArgumentException("Height and Width have to be larger than 4");
        if(complexity.boxes > 26) throw new IllegalArgumentException("Box limit is 26");
        if(complexity.agents > 9) throw new IllegalArgumentException("Agent limit is 9");
        if(complexity.agents > Farge.getClientFarger().length) throw new IllegalArgumentException("Max colors are" + Farge.getClientFarger().length);
    }
}
