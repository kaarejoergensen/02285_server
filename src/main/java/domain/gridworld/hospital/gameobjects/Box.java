package domain.gridworld.hospital.gameobjects;

import domain.gridworld.hospital.HospitalDomain;
import domain.gridworld.hospital.components.CanvasDetails;

import java.awt.*;
import java.awt.font.FontRenderContext;

public class Box extends GameObject{


    public boolean atGoal;

    public Box(byte id, Color color) {
        super(id, color);
    }

    @Override
    public void draw(Graphics2D g, int top, int left) {
        super.draw(g, top, left);
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
