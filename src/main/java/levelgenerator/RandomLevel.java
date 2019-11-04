package levelgenerator;

import java.awt.*;
import java.util.concurrent.ThreadLocalRandom;

public class RandomLevel {
    //Info
    private int count;

    private Complexity c;

    //Level
    public char[][] walls;

    public char[][] initStateElements;
    public char[][] goalStateElements;




    public RandomLevel(int complexityCode, int count){
        this.count = count;

        switch (complexityCode){
            case 1:
                c = Complexity.Basic;
                generateBasicLevel();
                break;
            default:
               throw new IllegalArgumentException(complexityCode + " is not a valid code. Input require a number between 1-10");
        }
        generateBasicLevel();
    }




    private void generateBasicLevel(){
        walls = new char[c.width][c.height];
        initStateElements = new char[c.width][c.height];
        goalStateElements = new char[c.width][c.height];
        createFrame();
        //Create the intital state
        randomlyPlaceAgentAndBoxes(initStateElements);
        //Create the inital state
        randomlyPlaceAgentAndBoxes(goalStateElements);
    }

    public void createFrame(){
        for(int y = 0; y < c.height; y++){
            if(y == 0 || y == c.height -1){
                for(int x = 0; x < c.width ; x++){
                    walls[x][y] = '+';
                }
            }else{
                walls[0][y] = '+';
                walls[c.width-1][y] = '+';
                for(int x = 1 ; x < c.width-1; x++) walls[x][y] = ' ';
            }
        }
    }

    public void randomlyPlaceAgentAndBoxes(char[][] state){
        for(int i = 48; i < (48 + c.agents) ; i++){
            while(true){
                Point rndPoint = getRandomCoordinate();
                if(state[rndPoint.x][rndPoint.y] == 0 && walls[rndPoint.x][rndPoint.y] == ' '){
                    state[rndPoint.x][rndPoint.y] = (char)i;
                    break;
                }

            }
        }

        for(int i = 65; i < (65 + c.boxes); i++){
            while(true){
                Point rndPoint = getRandomCoordinate();
                if(state[rndPoint.x][rndPoint.y] == 0 && walls[rndPoint.x][rndPoint.y] == ' '){
                    state[rndPoint.x][rndPoint.y] = (char)i;
                    break;
                }

            }
        }
    }

    public Point getRandomCoordinate(){
        int x = ThreadLocalRandom.current().nextInt(1, c.height );
        int y = ThreadLocalRandom.current().nextInt(1,  c.width );
        return new Point(x,y);
    }

    public int getCellCount(){
        int cellCount = 0;
        for(int x = 0; x < c.width; x++){
            for(int y = 0; y< c.height; y++){
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
        for(int x = 0; x < c.width; x++){
            for(int y = 0; y< c.height; y++){
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
        return c.label + "_" + count;
    }


}
