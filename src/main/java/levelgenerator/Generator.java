package levelgenerator;

import levelgenerator.modules.Writer;
import levelgenerator.pgl.Basic.Basic;
import levelgenerator.pgl.Dungeon.Dungeon;
import levelgenerator.pgl.RandomLevel;
import levelgenerator.pgl.RandomWalk.RandomWalk;

import java.io.IOException;

public class Generator {



    private Writer writer;

    public Generator(int amountOfLevels, String algorithm, Complexity complexity){
        writer = new Writer("generated");
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
        new Generator(1, "dungeon", Complexity.fromString("easy_3"));
    }

}
