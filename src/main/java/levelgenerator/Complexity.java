package levelgenerator;

public enum Complexity {

    Easy_1("Easy_1", 5,5, 1, 1, 1),
    Easy_2("Easy_2", 8,8, 1, 3, 1),
    Easy_3("Easy_3", 10,10, 1, 5, 1);


    public final String label;
    public final int width;
    public final int height;

    public final int agents;
    public final int boxes;

    public final int colors;

    Complexity(String label, int width, int height, int agents, int boxes, int colors){
        this.label = label;
        this.width = width;
        this.height = height;
        this.agents = agents;
        this.boxes = boxes;
        this.colors = colors;
    }

    public String getLabel(){
        return this.label;
    }

    public static Complexity fromString(String string){
        for(Complexity c : Complexity.values()){
            if(c.getLabel().equalsIgnoreCase(string)) return c;
        }
        return null;
    }


}
