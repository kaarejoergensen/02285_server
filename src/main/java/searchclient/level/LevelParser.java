package searchclient.level;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.InputMismatchException;

public class LevelParser {

    private BufferedReader file;
    private Level level;


    public LevelParser(BufferedReader file, Level level){
        this.level = level;
        this.file = file;
    }

    public void credentials() throws IOException {
        file.readLine(); // #domain
        level.domain = file.readLine(); // hospital

        // Read Level name.
        file.readLine(); // #levelname
        level.name =  file.readLine(); // <name>
    }

    public void colors() throws IOException{
        String colorCheck = file.readLine();
        if(colorCheck != "#colors") throw new InputMismatchException("Expected #colors");


    }




    public static <E> void printMatrix(E[][] input){
        String output = "[" + System.lineSeparator();
        for(int i = 0; i < input[0].length; i++){
            output += Arrays.toString(input[i]) + System.lineSeparator();
        }
        output += "]";
        System.err.println(output);
    }
}
