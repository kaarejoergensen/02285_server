package domain.gridworld.hospital.gameobjects;

import domain.gridworld.hospital.components.CanvasDetails;

import java.awt.*;
import java.awt.font.FontRenderContext;
import java.awt.font.TextLayout;
import java.awt.geom.AffineTransform;

public class Agent extends GameObject {
    public Agent(byte id, Color color){
        super(id, color);
    }

    public void letterTextUpdate(Font curFont, FontRenderContext fontRenderContext) {
        super.letterTextUpdate(Character.toString('0' + this.id), curFont, fontRenderContext);
    }

    @Override
    public void drawLetter(Graphics2D g, int top, int left) {
        super.drawLetter(g,  top, left);
        g.drawString("W", 0, 0);

    }
}
