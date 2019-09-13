package domain.gridworld.hospital.gameobjects;

import java.awt.font.TextLayout;

public interface IGameObject {
    public byte getId();

    public TextLayout getLetterText();

    public int getX();
    public void setX();
    public int getY();
    public void setY();

    public void draw();
}
