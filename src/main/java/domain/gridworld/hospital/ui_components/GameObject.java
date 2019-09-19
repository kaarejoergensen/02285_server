package domain.gridworld.hospital.ui_components;

import domain.gridworld.hospital.HospitalDomain;

import java.awt.*;
import java.awt.font.FontRenderContext;
import java.awt.font.TextLayout;
import java.awt.geom.AffineTransform;

public abstract class GameObject {

    protected static final Color BOX_AGENT_FONT_COLOR = Color.BLACK;
    protected static final AffineTransform IDENTITY_TRANSFORM = new AffineTransform();
    protected static final Stroke OUTLINE_STROKE = new BasicStroke(2.0f);


    public byte id;

    private TextLayout letterText;

    private int letterTopOffset;
    private int letterLeftOffset;

    private Color color;
    private Color outlineColor;
    private Color armColor;

    public boolean solved;

    //Method Variables
    protected int size;
    protected int cellBoxMargin;

    public GameObject(byte id, Color color){
        this.id = id;
        this.color = color;
        armColor = color.darker();
        outlineColor = color.darker().darker();
        solved  = false;
    }

    public void draw(Graphics2D g, int top, int left, Color alternativeColor){
        var canvas = HospitalDomain.canvas;
        size = canvas.cellSize - 2 * canvas.cellBoxMargin;
        cellBoxMargin = canvas.cellBoxMargin;
        g.setColor(alternativeColor == null ? color : alternativeColor);
    }

    public void draw(Graphics2D g, int top, int left) {
        draw(g, top, left, null);
    }

    public void drawLetter(Graphics2D g, int top, int left){
        g.setColor(BOX_AGENT_FONT_COLOR);
        getLetterText().draw(g, left + getLetterLeftOffset(), getLetterTopOffset() + top);
    }


    public void letterTextUpdate(String codePoint, Font curFont, FontRenderContext fontRenderContext){
        var canvas = HospitalDomain.canvas;
        // FIXME: Holy shit, creating a TextLayout object is SLOW!
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
