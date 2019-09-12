package domain.gridworld.hospital;

import java.awt.*;
import java.awt.font.TextLayout;
import java.awt.geom.AffineTransform;

public class Agent {

    public byte id;

    private TextLayout letterText;

    private int letterTopOffset;
    private int letterLeftOffset;

    private Color outlineColor;
    private Color armColor;


    public Agent(byte id, Color color){
        this.id = id;
        armColor = color.darker();
        outlineColor = color.darker().darker();
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
