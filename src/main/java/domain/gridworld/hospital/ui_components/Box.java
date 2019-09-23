package domain.gridworld.hospital.ui_components;

import java.awt.*;
import java.awt.font.FontRenderContext;

public class Box extends GameObject{


    public Box(byte id, Color color) {
        super(id, color);
    }


    @Override
    public void draw(Graphics2D g, int top, int left, Color color) {
        super.draw(g, top, left, color);
        g.fillRect(left + cellBoxMargin, top + cellBoxMargin, size,size);
        drawLetter(g,top,left);
    }


    public void letterTextUpdate(Font curFont, FontRenderContext fontRenderContext) {
        super.letterTextUpdate(Character.toString('A' + id), curFont, fontRenderContext);
    }

    @Override
    public void drawLetter(Graphics2D g, int top, int left) {
        super.drawLetter(g, top, left);
    }
}