package domain.gridworld.hospital.gameobjects;

import java.awt.*;
import java.awt.font.TextLayout;

public abstract class GameObject {

    public byte id;

    private TextLayout letterText;

    private int letterTopOffset;
    private int letterLeftOffset;

    private Color outlineColor;
    private Color armColor;


    public GameObject(byte id, Color color){
        this.id = id;
        armColor = color.darker();
        outlineColor = color.darker().darker();
    }

    public void draw(Graphics2D g, short row, short col){


    }

    public TextLayout getLetterText() {
        return letterText;
    }

    public void setLetterText(TextLayout letterText) {
        this.letterText = letterText;
    }

    public int getLetterTopOffset() {
        return letterTopOffset;
    }

    public void setLetterTopOffset(int letterTopOffset) {
        this.letterTopOffset = letterTopOffset;
    }

    public int getLetterLeftOffset() {
        return letterLeftOffset;
    }

    public void setLetterLeftOffset(int letterLeftOffset) {
        this.letterLeftOffset = letterLeftOffset;
    }

    public Color getArmColor() {
        return armColor;
    }

    public Color getOutlineColor() {
        return outlineColor;
    }

}
