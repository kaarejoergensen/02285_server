package levelgenerator.pgl;

import levelgenerator.Complexity;

import java.awt.*;

public class Basic extends RandomLevel{

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
                    walls[x][y] = '+';
                }
            }else{
                walls[0][y] = '+';
                walls[width-1][y] = '+';
                for(int x = 1 ; x < width-1; x++) walls[x][y] = ' ';
            }
        }
    }

    public void randomlyPlaceAgentAndBoxes(char[][] state){
        for(int i = 48; i < (48 + complexity.agents) ; i++){
            while(true){
                Point rndPoint = getRandomCoordinate();
                if(state[rndPoint.x][rndPoint.y] == 0 && walls[rndPoint.x][rndPoint.y] == ' '){
                    state[rndPoint.x][rndPoint.y] = (char)i;
                    break;
                }

            }
        }

        for(int i = 65; i < (65 + complexity.boxes); i++){
            while(true){
                Point rndPoint = getRandomCoordinate();
                if(state[rndPoint.x][rndPoint.y] == 0 && walls[rndPoint.x][rndPoint.y] == ' '){
                    state[rndPoint.x][rndPoint.y] = (char)i;
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
