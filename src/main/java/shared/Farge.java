package shared;


import java.awt.*;
import java.util.Locale;

public enum Farge {
    Blue (new Color(48, 80, 255)) ,
    Red (new Color(255, 0, 0)),
    Cyan (new Color(0, 255, 255)),
    Purple (new Color(96, 0, 176)),
    Green (new Color(0, 255, 0)),
    Orange (new Color(255, 128, 0)),
    Pink(new Color(240, 96, 192)),
    Lightblue(new Color(112, 192, 255)),
    Brown(new Color(96, 48, 0)),

    Grey(new Color(112, 112, 112)),
    UnresolvedGoal(new Color(223,223,0)),
    SolvedGoal(new Color(0,160,0)),
    LetterboxColor(new Color(0, 0, 0)),
    GridColor(new Color(64, 64, 64)),
    CellColor(new Color(192, 192, 192)),
    WallColor(new Color(0, 0, 0)),
    BoxAgentFontColor(new Color(0, 0, 0)),
    GoalColor(new Color(223, 223, 0)),
    GoalFontColor(new Color(66, 66, 0)),
    GoalSolvedColor(new Color(0, 160, 0));


    public Color color;

    Farge(Color color){
        this.color = color;
    }

    private static Farge[] farger = values();

    //TODO: Rename this class - doesn't make sense to call it colors. dankColors måske?
    public static Farge getFromRGB(Color color){
        for(Farge c : farger){
            if(c.color.equals(color)) return c;
        }
        return null;
    }

    public static Farge fromString(String s) {
        switch (s.toLowerCase(Locale.ROOT)) {
            case "blue":
                return Blue;
            case "red":
                return Red;
            case "cyan":
                return Cyan;
            case "purple":
                return Purple;
            case "green":
                return Green;
            case "orange":
                return Orange;
            case "pink":
                return Pink;
            case "grey":
                return Grey;
            case "lightblue":
                return Lightblue;
            case "brown":
                return Brown;
            default:
                return null;
        }
    }

    public static Farge next(Farge current){
        int nextIndex = current.ordinal() + 1;
        var nextColor = farger[nextIndex % farger.length];
        return nextColor != Grey || nextColor != UnresolvedGoal || nextColor != SolvedGoal? nextColor : next(nextColor);
    }

    public static Farge next(Color color){
        return next(getFromRGB(color));
    }

}
