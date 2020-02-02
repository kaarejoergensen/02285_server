package levelgenerator;

import levelgenerator.modules.Writer;
import levelgenerator.pgl.Basic.Basic;
import levelgenerator.pgl.Dungeon.Dungeon;
import levelgenerator.pgl.RandomLevel;
import levelgenerator.pgl.RandomWalk.RandomWalk;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class Generator {
    private Writer writer;

    public Generator(int amountOfLevels, String algorithm, Complexity complexity){
        String folder = "generated/" + algorithm + "/" + complexity;
        writer = new Writer(folder);
        if (!Files.isDirectory(Path.of(writer.getPath() + folder))) {
            try {
                Files.createDirectories(Path.of(writer.getPath() + folder));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        RandomLevel pgl = null;
        for(int i = 0 ; i < amountOfLevels; i++){
            switch (algorithm.toLowerCase()){
                case "basic":
                    pgl = new Basic(complexity, i);
                    break;
                case "random walk":    
                case "randomwalk":
                    pgl = new RandomWalk(complexity, i);
                    break;
                case "dungeon":
                    pgl = new Dungeon(complexity, i);
                    break;
                    
            }
            writer.toFile(pgl.toString(), pgl.getName());
        }
    }


    public static void main(String[] args)throws IOException {
        System.out.println("Generator Initated");

        new Generator(10, "dungeon", Complexity.fromString("easy_1"));
        new Generator(10, "dungeon", Complexity.fromString("easy_2"));
        new Generator(10, "dungeon", Complexity.fromString("easy_3"));
        new Generator(10, "dungeon", Complexity.fromString("medium_1"));
        new Generator(10, "dungeon", Complexity.fromString("medium_2"));
    }

}
