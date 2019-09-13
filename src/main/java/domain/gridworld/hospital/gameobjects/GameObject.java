package domain.gridworld.hospital.gameobjects;

import domain.gridworld.hospital.components.CanvasDetails;

import java.awt.*;
import java.awt.font.FontRenderContext;
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

    public void letterTextUpdate(String codePoint, Font curFont, FontRenderContext fontRenderContext, CanvasDetails canvas){
        setLetterText(new TextLayout(codePoint, curFont, fontRenderContext));
        Rectangle bound = getLetterText().getPixelBounds(fontRenderContext, 0, 0);

        int size = canvas.cellSize - 2 * canvas.cellTextMargin;
        setLetterTopOffset(canvas.cellTextMargin + size - (size - bound.height) / 2);
        setLetterLeftOffset(canvas.cellTextMargin + (size - bound.width) / 2 - bound.x);
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
