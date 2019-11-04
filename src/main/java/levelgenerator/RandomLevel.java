package levelgenerator;

import levelgenerator.pglAlgorithms.Basic;
import shared.Farge;

import java.awt.*;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public class RandomLevel {
    //Info
    private int count;

    public Complexity c;

    //Level
    public char[][] walls;
    public char[][] initStateElements;
    public char[][] goalStateElements;

    //Colors
    public char[][] colors;



    public RandomLevel(int complexityCode, int count){
        this.count = count;

        switch (complexityCode){
            case 1:
                c = Complexity.Basic;
                colors = new char[Farge.getClientFarger().length][c.boxes + c.agents];
                System.out.println(Arrays.deepToString(colors));
                new Basic(this);
                break;
            default:
               throw new IllegalArgumentException(complexityCode + " is not a valid code. Input require a number between 1-10");
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
