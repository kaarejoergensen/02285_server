package levelgenerator.modules;

import lombok.Getter;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class Writer {

    public String folder;
    @Getter public String path;

    public Writer(String destinationFolder){
        folder = destinationFolder;
        path = "src/main/resources/levels/";
    }

    public void toFile(String lvldata, String name){
        //Create file to the correct destination
        File tmpFile = new File(path +  folder + "/" + name + ".lvl");
        //I like to say, that you're my only fear
        try(PrintWriter out = new PrintWriter(tmpFile)){
            out.println(lvldata);
            System.out.println("Level: " + name + " written to folder '" + path  + folder + "'");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

}
