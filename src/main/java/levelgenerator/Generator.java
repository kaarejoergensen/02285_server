package levelgenerator;

import searchclient.level.Level;

import java.awt.*;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

import java.io.InputStream;
import java.io.InputStreamReader;

public class Generator {

    //Verdier som senere skal bli definert av complexity
    public final int AGENTS = 1;
    public final int BOXES = 1;

    public final int WIDTH = 5;
    public final int HEIGHT = 5;

    //Level
    public char[][] walls;

    public char[][] initStateElements;
    public char[][] goalStateElements;

    //Modules
    private LevelTester levelTester;
    private Writer writer;

    public Generator(int amountOfLevels){
        writer = new Writer("generated");

        for(int i = 0 ; i < amountOfLevels; i++){
            generateLevel();
            //LevelTester t = new LevelTester("greedy", "120", "./src/main/resources/levels/fred/");
            writer.toFile(levelToString());
        }
    }

    private void generateLevel(){
        walls = new char[WIDTH][HEIGHT];
        initStateElements = new char[WIDTH][HEIGHT];
        goalStateElements = new char[WIDTH][HEIGHT];
        createFrame();
        //Create the intital state
        randomlyPlaceAgentAndBoxes(initStateElements);
        //Create the inital state
        randomlyPlaceAgentAndBoxes(goalStateElements);
    }

    public void createFrame(){
        for(int y = 0; y < HEIGHT; y++){
            if(y == 0 || y == HEIGHT-1){
                for(int x = 0; x < WIDTH ; x++){
                    walls[x][y] = '+';
                }
            }else{
                walls[0][y] = '+';
                walls[WIDTH-1][y] = '+';
                for(int x = 1 ; x < WIDTH-1; x++) walls[x][y] = ' ';
            }
        }
    }


    //TODO: Smart måte å lage dette på
    public boolean ensureSolveable(){
        return true;
    }

    public int getCellCount(){
        int cellCount = 0;
        for(int x = 0; x < WIDTH; x++){
            for(int y = 0; y< HEIGHT; y++){
                if(walls[x][y] == '+');
            }
        }
        return cellCount;
    }

    public void randomlyPlaceAgentAndBoxes(char[][] state){
        for(int i = 48; i < (48 + AGENTS) ; i++){
            while(true){
                Point rndPoint = getRandomCoordinate();
                if(state[rndPoint.x][rndPoint.y] == 0 && walls[rndPoint.x][rndPoint.y] == ' '){
                    state[rndPoint.x][rndPoint.y] = (char)i;
                    break;
                }

            }
        }

        for(int i = 65; i < (65 + BOXES); i++){
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
        int x = ThreadLocalRandom.current().nextInt(1, HEIGHT );
        int y = ThreadLocalRandom.current().nextInt(1,  WIDTH );
        return new Point(x,y);
    }


    public static void main(String[] args)throws IOException {
        System.out.println("Generator Initated");
        Generator g = new Generator(10);
    }

    public void printWalls(){
        for(int x = 0; x < WIDTH; x++){
            for(int y = 0; y< HEIGHT; y++){
                System.out.print(walls[x][y]);
            }
            System.out.println();
        }
    }

    public String levelToString(){
        String out = "";
        out += "#Initial" + System.lineSeparator();
        out += stateToString(initStateElements);
        out += "#Goal" + System.lineSeparator();
        out += stateToString(goalStateElements);
        return out;
    }

    public String stateToString(char[][] state){
        String out = "";
        for(int x = 0; x < WIDTH; x++){
            for(int y = 0; y< HEIGHT; y++){
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




}
