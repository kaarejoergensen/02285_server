package levelgenerator;

import levelgenerator.modules.Writer;
import levelgenerator.pgl.Basic;
import levelgenerator.pgl.RandomLevel;
import shared.Farge;

import java.io.IOException;
import java.util.Arrays;

public class Generator {



    private Writer writer;

    public Generator(int amountOfLevels, String algorithm, Complexity complexity){
        writer = new Writer("generated");
        RandomLevel pgL = null;
        for(int i = 0 ; i < amountOfLevels; i++){
            switch (algorithm.toLowerCase()){
                case "basic":
                    pgL = new Basic(complexity, i);
                    break;
            }
            writer.toFile(pgL.toString(), pgL.getName());
        }


    }




    public static void main(String[] args)throws IOException {
        System.out.println("Generator Initated");
        Generator g = new Generator(1, "basic", Complexity.fromString("easy_1"));
    }

    /*
        Complexity Definiton
        1. Simple SA Level, Small Statespace, one box one agent
        2. Larger statespace
        3. More than 1 box
        4.
        5.

        6. Simple MA Level
        7. More boxes than agents (same color)
        8. More Colors
        9.
        10.
     */

}
