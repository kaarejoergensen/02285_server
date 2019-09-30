package searchclient.statecomponents;

import lombok.Getter;
import lombok.Setter;
import shared.Farge;


@Getter
public abstract  class Object {

    protected int id;
    protected Farge color;
    @Setter
    protected int X,Y;

    public Object(Farge colors, int init_x, int init_y){
        this.color = color;
        this.X = init_x;
        this.Y = init_y;
    }

    public Object(Farge color){
        color = color;
        X = 0; Y = 0;
    }
}
