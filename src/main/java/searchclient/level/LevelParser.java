package searchclient.level;

import shared.Farge;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.InputMismatchException;

public class LevelParser {

    private BufferedReader file;
    private Level level;

    final int BUFFER_SIZE = 1000; //Brukes til å markere i filen for å refreransepunkter. Tallet representerer hvor mange linjer fremover den skal huske

    final boolean print_debugger = true;

    public LevelParser(BufferedReader file, Level level){
        this.level = level;
        this.file = file;
        if(print_debugger) System.err.println("Parser Initated");
    }

    public void credentials() throws IOException {
        file.readLine(); // #domain
        level.domain = file.readLine(); // hospital

        // Read Level name.
        file.readLine(); // #levelname
        level.name =  file.readLine(); // <name>
        if(print_debugger)System.err.println("Credentials extracted [Name: '" + level.name + "' -  Domain: '" + level.domain + "']");

    }

    public void colors() throws IOException{
        checkArgument("#colors", file.readLine());

        String line = file.readLine();
        String debug_log = "";

        while (!line.startsWith("#")) {
            file.mark(BUFFER_SIZE);
            debug_log += line + " ";

            String[] split = line.split(":");
            Farge colors = Farge.fromString(split[0].strip());
            String[] entities = split[1].split(",");
            for (String entity : entities) {
                char c = entity.strip().charAt(0);
                if ('0' <= c && c <= '9') {
                    level.agentColors[c - '0'] = colors;
                } else if ('A' <= c && c <= 'Z') {
                    level.boxColors[c - 'A'] = colors;
                }
            }
            line = file.readLine();
        }

        if(print_debugger){
            System.err.println("Colors Read: " +  debug_log );
        }
    }

    public void determineMapDetails() throws IOException {
        file.reset();
        file.mark(BUFFER_SIZE);
        checkArgument("#initial", file.readLine());

        int height = 0, width = 0;
        String line = file.readLine();
        while (!line.startsWith("#")) {
            ++height;
            if(line.length() > width) width = line.length();
            line = file.readLine();
        }
        if(print_debugger) System.err.println("Map Details Determined. Height:[" + height + "] Width:[" + width + "]");
        level.setMapDetails(height,width);
    }

    public void initialState() throws IOException{
        file.reset();
        checkArgument("#initial", file.readLine());

        level.numAgents = 0;

        //75 er maks størrelse på banen. Siden vi ikke allerede vet
        //Høyde eller bredde på kartet, så lagrer vi først i temp_boxes, for å så flytte verdiene over
        //til en approriate størrelse. Det samme gjelder walls

        char[][] tmp_boxes = new char[75][75];
        boolean[][] tmp_walls = new boolean[75][75];

        String line = file.readLine();
        if(print_debugger)System.err.println(line);
        int row = 0;
        while (!line.startsWith("#")) {
            file.mark(BUFFER_SIZE);
            for (int col = 0 ; col < line.length(); ++col) {
                char c = line.charAt(col);
                if ('0' <= c && c <= '9') {
                    level.agentRows[c - '0'] = row;
                    level.agentCols[c - '0'] = col;
                    ++level.numAgents;
                } else if ('A' <= c && c <= 'Z') {
                    level.boxes[row][col] = c;
                } else if (c == '+') {
                    level.walls[row][col] = true;
                }
            }

            ++row;
            line = file.readLine();
            if(print_debugger) System.err.println(line);
        }


        level.setMapDetails(1,row);
    }

    public void goalState() throws IOException {
        file.reset();
        checkArgument("#goal",file.readLine());
        requireMapDetails();
        requireArraysInitialized();

        String line = file.readLine();
        System.err.println(line);
        int row = 0;
        while (!line.startsWith("#")) {
            file.mark(BUFFER_SIZE);
            for (int col = 0; col < line.length(); ++col) {
                char c = line.charAt(col);

                if (('0' <= c && c <= '9') || ('A' <= c && c <= 'Z')) {
                    level.goals[row][col] = c;
                }
            }

            ++row;
            line = file.readLine();
            if(print_debugger) System.err.println(line);

        }
    }


    /*
    HELP FUNCTIONS
     */

    private void checkArgument(String expectedString, String input){
        if(!input.equals(expectedString)) throw new InputMismatchException("Expected " + expectedString + ", got '" + input + "'");
    }

    public void requireMapDetails() throws InputMismatchException {
        if(level.height == 0 || level.width == 0) throw new InputMismatchException("Level Height and Width is required to run this function");
    }

    public void requireArraysInitialized() throws IndexOutOfBoundsException{
        if (level.walls == null || level.boxes == null || level.goals == null){
            throw new IndexOutOfBoundsException("Call 'initateMapDependentArrays' from 'Level' in order to initilize it's map dependent levels before calling this method");
        }
    }

    public static <E> void printMatrix(E[][] input){
        String output = "[" + System.lineSeparator();
        for(int i = 0; i < input[0].length; i++){
            output += Arrays.toString(input[i]) + System.lineSeparator();
        }
        output += "]";
        System.err.println(output);
    }

       /*
        char[][] boxes = new char[row][col];
        boolean[][] walls = new boolean[row][col];
        //Kopierer verdiene over til den ekte
        for(int i = row; i < 0; i++){
            for(int j = col; j < 0 ; j++){
                boxes[i][j] = tmp_boxes[i][j];
                walls[i][j] = tmp_walls[i][j];
            }
        }
        */
}
