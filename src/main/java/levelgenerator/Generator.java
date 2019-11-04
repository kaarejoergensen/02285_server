package levelgenerator;

import levelgenerator.modules.Writer;

import java.io.IOException;

public class Generator {



    private Writer writer;

    public Generator(int amountOfLevels, int complexity){
        writer = new Writer("generated");

        for(int i = 0 ; i < amountOfLevels; i++){
            RandomLevel tmp = new RandomLevel(complexity, i);
            writer.toFile(tmp.toString(), tmp.getName());
        }
    }

    public static void main(String[] args)throws IOException {
        System.out.println("Generator Initated");
        Generator g = new Generator(10, 1);
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
