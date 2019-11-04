package levelgenerator.pglAlgorithms;

import levelgenerator.Complexity;
import levelgenerator.RandomLevel;

import java.awt.*;

public class Basic {

    /*
        Logic for the basic generator.
        No additional walls, just agents and boxes in an open space
     */

    private RandomLevel l;
    private Complexity c;

    public Basic(RandomLevel level) {
        this.l = level;
        this.c = level.c;

        l.walls = new char[c.width][c.height];
        l.initStateElements = new char[c.width][c.height];
        l.goalStateElements = new char[c.width][c.height];
        createFrame();
        //Create the intital state
        randomlyPlaceAgentAndBoxes(l.initStateElements);
        //Create the inital state
        randomlyPlaceAgentAndBoxes(l.goalStateElements);
    }

    public void createFrame(){
        for(int y = 0; y < l.c.height; y++){
            if(y == 0 || y == l.c.height -1){
                for(int x = 0; x < c.width ; x++){
                    l.walls[x][y] = '+';
                }
            }else{
                l.walls[0][y] = '+';
                l.walls[l.c.width-1][y] = '+';
                for(int x = 1 ; x < l.c.width-1; x++) l.walls[x][y] = ' ';
            }
        }
    }

    public void randomlyPlaceAgentAndBoxes(char[][] state){
        for(int i = 48; i < (48 + l.c.agents) ; i++){
            while(true){
                Point rndPoint = l.getRandomCoordinate();
                if(state[rndPoint.x][rndPoint.y] == 0 && l.walls[rndPoint.x][rndPoint.y] == ' '){
                    state[rndPoint.x][rndPoint.y] = (char)i;
                    break;
                }

            }
        }

        for(int i = 65; i < (65 + l.c.boxes); i++){
            while(true){
                Point rndPoint = l.getRandomCoordinate();
                if(state[rndPoint.x][rndPoint.y] == 0 && l.walls[rndPoint.x][rndPoint.y] == ' '){
                    state[rndPoint.x][rndPoint.y] = (char)i;
                    break;
                }

            }
        }
    }

}
