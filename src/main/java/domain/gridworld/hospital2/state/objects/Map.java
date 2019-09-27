package domain.gridworld.hospital2.state.objects;

import domain.gridworld.hospital2.state.objects.ui.CanvasDetails;
import lombok.AllArgsConstructor;

import java.awt.*;
import java.util.List;

import static domain.gridworld.hospital2.state.Colors.*;

@AllArgsConstructor
public class Map {
    private List<List<Boolean>> map;

    public boolean isCell(int row, int col) {
        return this.isPartOfMap(row, col) && map.get(row).get(col);
    }

    public boolean isWall(int row, int col) {
        return this.isPartOfMap(row, col) && !map.get(row).get(col);
    }

    private boolean isPartOfMap(int row, int col) {
        return row < map.size() && col < map.get(row).size();
    }

    private int getNumRows() {
        return this.map.size();
    }

    private List<Boolean> getRow(int row) {
        return this.map.get(row);
    }

    private int getNumCols(int row) {
        return this.getRow(row).size();
    }

    public void draw(Graphics2D g, CanvasDetails canvasDetails, int width, int height) {
        this.drawLetterBox(g, width, height);
        this.drawCellBackground(g, canvasDetails);
        this.drawGridAndWalls(g, canvasDetails);
    }

    private void drawLetterBox(Graphics2D g, int width, int height) {
        g.setColor(LETTERBOX_COLOR);
        g.fillRect(0, 0, width, height);
    }

    private void drawCellBackground(Graphics2D g, CanvasDetails canvasDetails) {
        g.setColor(CELL_COLOR);
        g.fillRect(canvasDetails.getOriginLeft(), canvasDetails.getOriginTop(),
                canvasDetails.getWidth(), canvasDetails.getHeight());
    }

    private void drawGridAndWalls(Graphics2D g, CanvasDetails canvasDetails) {
        for (short row = 0; row < this.getNumRows(); ++row) {
            int top = canvasDetails.getOriginTop() + row * canvasDetails.getCellSize();
            for (short col = 0; col < this.getNumCols(row); ++col) {
                int left = canvasDetails.getOriginLeft() + col * canvasDetails.getCellSize();
                if (this.isWall(row, col)) {
                    g.setColor(WALL_COLOR);
                    g.fillRect(left, top, canvasDetails.getCellSize(), canvasDetails.getCellSize());
                } else {
                    g.setColor(GRID_COLOR);
                    g.drawRect(left, top, canvasDetails.getCellSize() - 1, canvasDetails.getCellSize() - 1);
                }
            }
        }
    }
}
