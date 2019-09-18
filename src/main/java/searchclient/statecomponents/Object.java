package searchclient.statecomponents;

import lombok.Getter;
import lombok.Setter;
import searchclient.Color;


@Getter
public abstract  class Object {

    protected int id;
    protected Color color;
    @Setter
    protected int X,Y;

    public Object(Color color, int init_x, int init_y){
        this.color = color;
        this.X = init_x;
        this.Y = init_y;
    }

    public Object(Color color){
        color = color;
        X = 0; Y = 0;
    }
}
