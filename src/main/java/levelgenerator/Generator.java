package levelgenerator;

import levelgenerator.modules.Writer;
import levelgenerator.pgl.Basic;
import levelgenerator.pgl.RandomLevel;
import levelgenerator.pgl.RandomWalk;
import shared.Farge;

import java.io.IOException;
import java.util.Arrays;

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
                    
            }
            writer.toFile(pgl.toString(), pgl.getName());
        }


    }


    public static void main(String[] args)throws IOException {
        System.out.println("Generator Initated");
        Generator g = new Generator(1, "randomwalk", Complexity.fromString("easy_3"));
    }

}
