package levelgenerator.pgl.Basic;

import levelgenerator.Complexity;
import levelgenerator.pgl.RandomLevel;

import java.awt.*;

public class Basic extends RandomLevel {

    /*
        Logic for the basic generator.
        No additional walls, just agents and boxes in an open space
     */


    public Basic(Complexity c, int levelNumber) {
        super(c, levelNumber);

        createFrame();
        //Create the intital state
        randomlyPlaceAgentAndBoxes(initStateElements);
        //Create the inital state
        randomlyPlaceAgentAndBoxes(goalStateElements);
    }

    public void createFrame(){
        for(int y = 0; y < height; y++){
            if(y == 0 || y == height -1){
                for(int x = 0; x < width ; x++){
                    walls[y][x] = '+';
                }
            }else{
                walls[y][0] = '+';
                walls[y][width-1] = '+';
                for(int x = 1 ; x < width-1; x++) walls[y][x] = ' ';
            }
        }
    }

    public void randomlyPlaceAgentAndBoxes(char[][] state){
        for(int i = 48; i < (48 + complexity.agents) ; i++){
            while(true){
                Point rndPoint = getRandomCoordinate();
                if(state[rndPoint.y][rndPoint.x] == 0 && walls[rndPoint.y][rndPoint.x] == ' '){
                    state[rndPoint.y][rndPoint.x] = (char)i;
                    break;
                }

            }
        }

        for(int i = 65; i < (65 + complexity.boxes); i++){
            while(true){
                Point rndPoint = getRandomCoordinate();
                if(state[rndPoint.y][rndPoint.x] == 0 && walls[rndPoint.y][rndPoint.x] == ' '){
                    state[rndPoint.y][rndPoint.x] = (char)i;
                    break;
                }

            }
        }
    }

    @Override
    public String getAlgorithmName() {
        return "Basic";
    }
}
