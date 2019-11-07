package levelgenerator.pgl;

import levelgenerator.Complexity;
import shared.Action.MoveDirection;

import java.awt.*;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;

public class RandomWalk extends RandomLevel{

    /*
        Logic for the basic generator.
        No additional walls, just agents and boxes in an open space
     */

    //At least MINIMUM_SPACE of the map have to be path
    private final double MINIMUM_SPACE = 0.5;

    private ArrayList<Point> paths;

    public RandomWalk(Complexity c, int levelNumber) {
        super(c, levelNumber);

        paths = new ArrayList<>();

        fillLevelWithWalls();
        //Pick a random point and start walking to create a room

        Point chosenPoint;

        while(paths.size() < (getTotalSpace()*MINIMUM_SPACE)){

            if(paths.size() == 0){
                chosenPoint = getRandomCoordinate();
                setPointToPath(chosenPoint);
            }else{
                chosenPoint = getRandomPointFromPath();
            }

            doTheWalk(chosenPoint);

        }

        //System.out.println(paths);

        distributeElements(initStateElements);
        distributeElements(goalStateElements);

    }


    private void doTheWalk(Point p){
        //Get random direction from point
        MoveDirection direction = MoveDirection.values()[ThreadLocalRandom.current().nextInt(0, MoveDirection.values().length-1)];
        //New point after direction applied
        Point newPoint = getNewPoint(p, direction);
        //If the pint is not in the edge, and a wall, then set it to wall
        if(!isFrame(newPoint) && isWall(newPoint)){
            setPointToPath(newPoint);
            doTheWalk(newPoint);
        }

    }

    private void setPointToPath(Point p){
        System.out.println(p);
        walls[p.y][p.x] = ' ';
        paths.add(p);
    }

    private Point getRandomPointFromPath(){
        return paths.get(ThreadLocalRandom.current().nextInt(0, paths.size()-1));
    }

    private void distributeElements(char[][] state){
        ArrayList<Point> pathsCopy = new ArrayList<>(paths);
        char[] elements = elementsToArray();
        for(char s : elements){
            int extractIndex = ThreadLocalRandom.current().nextInt(0, pathsCopy.size()-1);
            Point p = pathsCopy.get(extractIndex);
            pathsCopy.remove(extractIndex);

            state[p.y][p.x] = s;
        }
    }



    @Override
    public String getAlgorithmName() {
        return "RandomWalk";
    }
}
