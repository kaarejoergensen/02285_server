package domain.gridworld.hospital.gameobjects;

import domain.gridworld.hospital.HospitalDomain;
import domain.gridworld.hospital.components.CanvasDetails;

import java.awt.*;
import java.awt.font.FontRenderContext;
import java.awt.font.TextLayout;

public abstract class GameObject {

    private static final Color BOX_AGENT_FONT_COLOR = Color.BLACK;


    public byte id;

    private TextLayout letterText;

    private int letterTopOffset;
    private int letterLeftOffset;

    private Color color;
    private Color outlineColor;
    private Color armColor;

    //Method Variables
    protected int size;
    protected int cellBoxMargin;

    public GameObject(byte id, Color color){
        this.id = id;
        this.color = color;
        armColor = color.darker();
        outlineColor = color.darker().darker();
    }

    public void draw(Graphics2D g, int top, int left){
        var canvas = HospitalDomain.canvas;
        size = canvas.cellSize - 2 * canvas.cellBoxMargin;
        cellBoxMargin = canvas.cellBoxMargin;
        g.setColor(color);
    }

    public void drawLetter(Graphics2D g, int top, int left){
        g.setColor(BOX_AGENT_FONT_COLOR);
        getLetterText().draw(g, left + getLetterLeftOffset(), getLetterTopOffset() + top);
    }

    public void letterTextUpdate(String codePoint, Font curFont, FontRenderContext fontRenderContext){
        var canvas = HospitalDomain.canvas;
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
