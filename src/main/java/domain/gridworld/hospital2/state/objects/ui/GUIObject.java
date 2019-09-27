package domain.gridworld.hospital2.state.objects.ui;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.awt.*;
import java.awt.font.TextLayout;

import static domain.gridworld.hospital2.state.Colors.BOX_AGENT_FONT_COLOR;

@RequiredArgsConstructor
@Getter
public abstract class GUIObject {
    final protected String id;

    final protected Character letter;
    final protected Color color;
    protected TextLayout letterText;
    protected int letterTopOffset, letterLeftOffset;

    void letterTextUpdate(CanvasDetails canvasDetails){
        this.letterText = new TextLayout(String.valueOf(this.letter), canvasDetails.getCurrentFont(), canvasDetails.getFontRenderContext());
        Rectangle bound = this.letterText.getPixelBounds(canvasDetails.getFontRenderContext(), 0, 0);

        int size = canvasDetails.getCellSize() - 2 * canvasDetails.getCellTextMargin();
        this.letterTopOffset = canvasDetails.getCellTextMargin() + size - (size - bound.height) / 2;
        this.letterLeftOffset = canvasDetails.getCellTextMargin() + (size - bound.width) / 2 - bound.x;
    }

    public void draw(Graphics2D g, CanvasDetails canvasDetails, short row, short col) {
        int top = canvasDetails.getOriginTop() + row * canvasDetails.getCellSize();
        int left = canvasDetails.getOriginLeft() + col * canvasDetails.getCellSize();
        int size = canvasDetails.getCellSize() - 2 * canvasDetails.getCellBoxMargin();
        g.setColor(this.color);
        if (this.isAgent()) {
            g.fillOval(left + canvasDetails.getCellBoxMargin(), top + canvasDetails.getCellBoxMargin(), size, size);
        } else {
            g.fillRect(left + canvasDetails.getCellBoxMargin(), top + canvasDetails.getCellBoxMargin(), size, size);
        }

        g.setColor(BOX_AGENT_FONT_COLOR);
        letterText.draw(g, left + this.letterLeftOffset, top + this.letterTopOffset);
    }

    protected boolean isAgent() {
        return '0' <= this.letter && this.letter <= '9';
    }
}
