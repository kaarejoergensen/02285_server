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

    public void letterTextUpdate(Font curFont, FontRenderContext fontRenderContext, CanvasDetails canvas) {
        super.letterTextUpdate(Character.toString('0' + this.id), curFont, fontRenderContext, canvas);
    }
}
