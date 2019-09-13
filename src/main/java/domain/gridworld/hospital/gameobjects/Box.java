package domain.gridworld.hospital.gameobjects;

import domain.gridworld.hospital.components.CanvasDetails;

import java.awt.*;
import java.awt.font.FontRenderContext;

public class Box extends GameObject{

    public boolean atGoal;

    public Box(byte id, Color color) {
        super(id, color);
    }


    public void letterTextUpdate(Font curFont, FontRenderContext fontRenderContext, CanvasDetails canvas) {
        super.letterTextUpdate(Character.toString('A' + id), curFont, fontRenderContext, canvas);
    }
}
