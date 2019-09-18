package searchclient;

import java.util.Locale;

public enum Color {
    Blue,
    Red,
    Cyan,
    Purple,
    Green,
    Orange,
    Pink,
    Grey,
    Lightblue,
    Brown;


    private static Color[] colors = values();

    public static Color fromString(String s) {
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

    static Color next(Color current){
        int nextIndex = current.ordinal() + 1;
        nextIndex %= colors.length;
        return colors[nextIndex];
    }
}
