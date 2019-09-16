package domain.gridworld.hospital;

import java.awt.*;

/**
 * Supported colors:
 * blue, red, cyan, purple, green, orange, pink, grey, lightblue, brown
 */
class Colors {
    private static final Color Blue = new Color(48, 80, 255);
    private static final Color Red = new Color(255, 0, 0);
    private static final Color Cyan = new Color(0, 255, 255);
    private static final Color Purple = new Color(96, 0, 176);
    private static final Color Green = new Color(0, 255, 0); // FIXME: Green not too good with the goal color.
    private static final Color Orange = new Color(255, 128, 0);
    private static final Color Pink = new Color(240, 96, 192);
    private static final Color Lightblue = new Color(112, 192, 255);
    private static final Color Brown = new Color(96, 48, 0);

    //Update: Gray now means "uncolored"
    private static final Color Grey = new Color(112, 112, 112);

    static final Color UnsolvedGoal = new Color(223, 223, 0);
    static final Color SolvedGoal = new Color(0, 160, 0);

    static Color fromString(String colorName) {
        switch (colorName) {
            case "blue":
                return Colors.Blue;
            case "red":
                return Colors.Red;
            case "cyan":
                return Colors.Cyan;
            case "purple":
                return Colors.Purple;
            case "green":
                return Colors.Green;
            case "orange":
                return Colors.Orange;
            case "pink":
                return Colors.Pink;
            case "grey":
                return Colors.Grey;
            case "lightblue":
                return Colors.Lightblue;
            case "brown":
                return Colors.Brown;
            default:
                return null;
        }
    }

    static Color nextFrom(Color current){
        int RGB = current.getRGB();
        if(RGB == Blue.getRGB()) return Red;
        if(RGB ==)
        switch(RGB){
            case Blue.getRGB():
                return Colors.Red;
            case current.getRGB() == Red.getRGB():
                return Colors.Cyan;
            case current.getRGB() == Cyan.getRGB():
                return Colors.Purple;
            case current.getRGB() == Purple.getRGB():
                return Colors.Green;
            case current.getRGB() == Green.getRGB():
                return Colors.Orange;
            case current.getRGB() == Orange.getRGB():
                return Colors.Pink;
            case current.getRGB() == Pink.getRGB():
                return Colors.Lightblue;
            case current.getRGB() == Grey.getRGB():
                return Colors.Grey;
            case current.getRGB() == Lightblue.getRGB():
                return Colors.Brown;
            case current.getRGB() == Brown.getRGB():
                return Colors.Blue;
            default:
                return null;
        }
    }
}
