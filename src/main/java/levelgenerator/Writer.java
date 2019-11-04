package levelgenerator;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class Writer {

    public int levelNumber;

    public String folder;
    public String path;

    public Writer(String destinationFolder){
        levelNumber = 0;
        folder = destinationFolder;
        path = "src/main/resources/levels/";
    }

    public void toFile(String lvldata){
        //Format Number to string
        String formatted = String.format("%03d", levelNumber);
        //Create file to the correct destination
        File tmpFile = new File(path +  folder + "/" + levelNumber + ".lvl");
        //I like to say, that you're my only fear
        try(PrintWriter out = new PrintWriter(tmpFile)){
            out.println(lvldata);
            System.out.println("Level: " + levelNumber + " written to folder '" + path  + folder + "'");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

}
