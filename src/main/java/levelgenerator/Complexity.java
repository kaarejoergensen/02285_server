package levelgenerator;

public enum Complexity {

    Basic("Basic", 5,5, 1, 1, 1);

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


}
